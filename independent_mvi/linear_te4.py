"""Standalone weak-form X/Y variational-iteration solver for linear CUF TE4.

This module deliberately DOES NOT import independent_mkr, cylinder_functional_
assembly, cylinder_variational_dynamic, or any shared solver/backend module.
Its own: cylindrical thickness integration, geometry, angular/axial derivative
matrices, discrete weak first variation, inertia, load, essential constraints,
rank-fixed alternating factor subproblems, and two-stage Bathe time stepping.

This is a NEW experimental implementation, not a claim of a historically
independently verified Krysko-Kirichenko solver. It supports linear homogeneous
isotropic elastic TE4, zero end displacement, no micro/electric physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class VariationalCase:
    length_m: float = 2.
    radius_m: float = 1.
    thickness_m: float = .05
    young_pa: float = 210.e9
    poisson: float = .3
    density_kg_m3: float = 7800.
    nx: int = 9
    ns: int = 9
    pressure_pa: float = 100000.
    frequency_hz: float = 25.
    pressure_kind: str = "UNIFORM"
    x_interval_m: tuple[float,float] = (.3,1.6)
    theta_interval_rad: tuple[float,float] = (.45,3.1)
    x_edge_m: float = .35
    theta_edge_rad: float = .6

    def __post_init__(self):
        if self.nx<9 or self.ns<5 or self.ns%2!=1:
            raise ValueError("MVI requires nx>=9, odd ns>=5")
        if not self.radius_m>self.thickness_m/2>0 or self.length_m<=0:
            raise ValueError("invalid MVI cylinder geometry")
        if not self.young_pa>0 or not -1<self.poisson<.5:
            raise ValueError("invalid MVI isotropic Hooke material")
        if not self.density_kg_m3>0 or self.frequency_hz<=0 or self.pressure_pa<=0:
            raise ValueError("invalid MVI physical input")
        if self.pressure_kind not in ("UNIFORM","PATCH"):
            raise ValueError("unsupported MVI pressure type")
        if (not 0<=self.x_interval_m[0]<self.x_interval_m[1]<=self.length_m or
                not 0<=self.theta_interval_rad[0]<
                    self.theta_interval_rad[1]<=2*pi):
            raise ValueError("MVI pressure patch outside cylinder")


def _sbp42(n,length):
    """Independent fourth-order interior / second-order SBP closure."""
    if n<9:
        raise ValueError("MVI SBP42 requires at least 9 axial nodes")
    dx=length/(n-1)
    d=np.zeros((n,n))
    closure=(
        ((0,-24/17),(1,59/34),(2,-4/17),(3,-3/34)),
        ((0,-.5),(2,.5)),
        ((0,4/43),(1,-59/86),(3,59/86),(4,-4/43)),
        ((0,3/98),(2,-59/98),(4,32/49),(5,-4/49)),
    )
    for i in range(n):
        if i<4: entries=closure[i]
        elif i>=n-4: entries=tuple((n-1-j,-c) for j,c in closure[n-1-i])
        else: entries=((i-2,1/12),(i-1,-2/3),(i+1,2/3),(i+2,-1/12))
        for j,c in entries:d[i,j]=c/dx
    weight=np.ones(n)
    weight[:4]=[17/48,59/48,43/48,49/48]
    weight[-4:]=weight[:4][::-1]
    return d,weight*dx


def _fourier_derivative(n,radius):
    """Independent spectral trigonometric derivative in angular arc length."""
    circumference=2*pi*radius
    wave=2*pi*np.fft.fftfreq(n,d=circumference/n)
    eye=np.eye(n)
    d=np.fft.ifft(1j*wave[:,None]*np.fft.fft(eye,axis=0),axis=0).real
    return .5*(d-d.T),np.full(n,circumference/n)


def _pressure(case: VariationalCase):
    shape=(case.nx,case.ns)
    window=np.ones(shape,float)
    if case.pressure_kind=="PATCH":
        def smooth(x,width):
            t=np.clip(x/width+.5,0.,1.)
            return np.sin(.5*pi*t)**2
        x=np.linspace(0.,case.length_m,case.nx)
        a,b=case.x_interval_m
        wx=smooth(x-a,case.x_edge_m)*smooth(b-x,case.x_edge_m)
        angles=2*pi*np.arange(case.ns)/case.ns
        lo,hi=case.theta_interval_rad
        distance=(angles-(lo+hi)/2+pi)%(2*pi)-pi
        wtheta=smooth((hi-lo)/2-np.abs(distance),case.theta_edge_rad)
        window=np.outer(wx,wtheta)
    load=np.zeros((15,)+shape)
    load[10]=-case.pressure_pa*window
    return load


class VariationalTE4:
    def __init__(self,case: VariationalCase,rank:int):
        if not isinstance(case,VariationalCase):
            raise TypeError("independent MVI requires VariationalCase")
        if type(rank) is not int or rank<1 or rank>8:
            raise ValueError("MVI rank must be 1..8")
        # Every factor of an independently active separated sum must vanish
        # at each clamped axial end. Only nx-2 independent axial directions.
        if rank>min(case.nx-2,case.ns):
            raise ValueError("MVI_RANK_EXCEEDS_CLAMPED_SPATIAL_CAPACITY")
        self.case,self.rank=case,rank
        self.dx,self.wx=_sbp42(case.nx,case.length_m)
        self.ds,self.ws=_fourier_derivative(case.ns,case.radius_m)
        self.surface_weight=np.outer(self.wx,self.ws)
        self.mass,self.blocks=self._integrate_te4()
        self.pressure=_pressure(case)

    def _integrate_te4(self):
        c=self.case
        lam=c.young_pa*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu=c.young_pa/(2*(1+c.poisson))
        hooke=np.diag([2*mu]*3+[mu]*3)
        hooke[:3,:3]+=lam
        mass=np.zeros((5,5))
        blocks=np.zeros((3,3,15,15))
        xi,wi=np.polynomial.legendre.leggauss(10)
        for u,weight in zip(xi,wi):
            z=.5*c.thickness_m*u
            weighted=.5*c.thickness_m*weight
            H=1+z/c.radius_m
            f=u**np.arange(5)
            dz=np.array([0. if i==0 else i*2/c.thickness_m*u**(i-1)
                         for i in range(5)])
            mass+=weighted*H*c.density_kg_m3*np.outer(f,f)
            # In-plane strain coordinates [xx,ss,zz, xs,xz,sz] with
            # engineering shear. Row is q, q_x, or q_s respectively.
            b=np.zeros((3,6,15))
            for k in range(5):
                b[1,0,k]=f[k]
                b[0,1,10+k]=f[k]/(c.radius_m*H)
                b[2,1,5+k]=f[k]/H
                b[0,2,10+k]=dz[k]
                b[1,3,5+k]=f[k]
                b[2,3,k]=f[k]/H
                b[0,4,k]=dz[k]
                b[1,4,10+k]=f[k]
                b[0,5,5+k]=dz[k]-f[k]/(c.radius_m*H)
                b[2,5,10+k]=f[k]/H
            for i in range(3):
                for j in range(3):
                    blocks[i,j]+=weighted*H*b[i].T@hooke@b[j]
        return mass,blocks

    def _x(self,u):
        return np.einsum("ij,cjs->cis",self.dx,u,optimize=True)

    def _s(self,u):
        return np.einsum("ij,cxj->cxi",self.ds,u,optimize=True)

    def weak_stiffness(self,q):
        c=self.case
        q=np.asarray(q,float)
        if q.shape!=(15,c.nx,c.ns):
            raise ValueError("MVI invalid generalized field shape")
        gradients=[q,self._x(q),self._s(q)]
        flux=[]
        for i in range(3):
            flux.append(sum(np.einsum("ab,bxs->axs",self.blocks[i,j],gradients[j],
                        optimize=True) for j in range(3)))
        wx=self.surface_weight[None]
        return (wx*flux[0]+
                np.einsum("ji,cjs->cis",self.dx,wx*flux[1],optimize=True)+
                np.einsum("ji,cxj->cxi",self.ds,wx*flux[2],optimize=True))

    def weak_mass(self,q):
        c=self.case
        a=np.asarray(q,float).reshape(3,5,c.nx,c.ns)
        return (np.einsum("ij,cjxs->cixs",self.mass,a,
                          optimize=True).reshape(q.shape)*self.surface_weight[None])

    def weak_load(self,time_s):
        return (self.pressure*self.surface_weight[None]*
                np.sin(2*pi*self.case.frequency_hz*time_s))

    def _initial(self):
        """Zero physical field; independent nonzero clamped axial partners."""
        c=self.case
        x=np.zeros((15,self.rank,c.nx))
        y=np.zeros((15,self.rank,c.ns))
        grid=np.linspace(0,1,c.nx)
        modes=np.array([np.sin((j+1)*pi*grid) for j in range(self.rank)])
        ortho=np.linalg.qr(modes[:,1:-1].T,mode="reduced")[0].T
        x[:,:,1:-1]=np.broadcast_to(ortho,(15,)+ortho.shape)
        return x,y

    @staticmethod
    def _field(x,y):
        return np.einsum("crx,crs->cxs",x,y,optimize=True)

    @staticmethod
    def _canonicalize(x,y,freeze):
        x,y=x.copy(),y.copy()
        for c in range(len(x)):
            if freeze=="X":
                # Orthogonalize the active axial partner in its admissible
                # interior, preserving both boundary zeros and full field.
                Q,R=np.linalg.qr(x[c,:,1:-1].T,mode="reduced")
                x[c,:,1:-1]=Q.T
                y[c]=R@y[c]
            elif freeze=="Y":
                Q,R=np.linalg.qr(y[c].T,mode="reduced")
                y[c]=Q.T
                x[c]=R@x[c]
            else:raise ValueError("MVI must freeze X or Y")
        return x,y

    def _directions(self,x,y,active):
        c=self.case
        n=c.nx*c.ns
        span=(c.ns if active=="Y" else c.nx-2)
        t=np.zeros((15*n,15*self.rank*span),float)
        for component in range(15):
            for j in range(self.rank):
                for k in range(span):
                    idx=(component*self.rank+j)*span+k
                    row=np.zeros((c.nx,c.ns))
                    if active=="Y":
                        row[:,k]=x[component,j]
                    else:
                        row[k+1,:]=y[component,j]
                    t[component*n:(component+1)*n,idx]=row.ravel()
        return t

    def _linear_endpoint(self,pred,time_s,coefficient,previous=None,
                         max_sweeps=35,relative_change=2.e-7,
                         krylov_rtol=1.e-7,max_krylov_matvec=1800):
        c=self.case
        if previous is None:x,y=self._initial()
        else:
            x,y=tuple(np.asarray(a,float).copy() for a in previous)
        if (x.shape!=(15,self.rank,c.nx) or
                y.shape!=(15,self.rank,c.ns) or
                np.any(x[:,:,[0,-1]])):
            raise ValueError("MVI previous separated profiles invalid")
        rhs=self.weak_load(time_s)+coefficient*self.weak_mass(pred)
        size=15*c.nx*c.ns

        def stiffness(q):
            return self.weak_stiffness(q)+coefficient*self.weak_mass(q)

        accepted=False
        loops=[]
        for sweep in range(1,max_sweeps+1):
            before=self._field(x,y)
            for freeze,active in (("X","Y"),("Y","X")):
                x,y=self._canonicalize(x,y,freeze)
                T=self._directions(x,y,active)
                old=self._field(x,y)
                b=T.T@(rhs-stiffness(old)).ravel()
                if np.linalg.norm(b)<1.e-18: continue
                count=[0]
                def projected_action(delta):
                    count[0]+=1
                    if count[0]>max_krylov_matvec:
                        raise RuntimeError("MVI_ONE_DIMENSIONAL_GMRES_BUDGET")
                    field=(T@delta).reshape(15,c.nx,c.ns)
                    return T.T@stiffness(field).ravel()
                operator=LinearOperator((len(b),len(b)),
                                        matvec=projected_action,dtype=float)
                solution,info=gmres(operator,b,rtol=krylov_rtol,atol=0.,
                                   restart=min(70,len(b)),
                                   maxiter=max(1,max_krylov_matvec//min(70,len(b))),
                                   callback_type="pr_norm")
                if info!=0 or not np.isfinite(solution).all():
                    raise RuntimeError("MVI_FROZEN_PARTNER_GMRES_NOT_CONVERGED")
                ncoord=self.rank*(c.ns if active=="Y" else c.nx-2)
                delta=solution.reshape(15,self.rank,-1)
                if delta.shape[1:]!=(self.rank,ncoord//self.rank):
                    raise ValueError("MVI_HALFSTEP_WRONG_DIMENSION")
                if active=="Y": y+=delta
                else:x[:,:,1:-1]+=delta
                loops.append((sweep,active,count[0]))
            current=self._field(x,y)
            diff=float(np.linalg.norm(current-before))
            denom=max(float(np.linalg.norm(current)),1.e-20)
            if diff<=relative_change*denom:
                accepted=True
                break
        if not accepted:
            raise RuntimeError("MVI_XY_ALTERNATION_NOT_CONVERGED")
        state=self._field(x,y)
        if np.any(state[:,[0,-1],:]):
            raise RuntimeError("MVI_CLAMPED_DISPLACEMENT_FAILED")
        return state,(x,y),tuple(loops)

    def step_bathe(self,q,v,a,time_s,dt_s=.002,
                   previous_profiles=None,max_sweeps=35):
        """MVI factors recomputed at both implicit Bathe displacement stages."""
        c=self.case
        shape=(15,c.nx,c.ns)
        q,v,a=tuple(np.asarray(z,float) for z in (q,v,a))
        if any(z.shape!=shape for z in (q,v,a)) or dt_s<=0:
            raise ValueError("MVI invalid Bathe initial state")
        if any(np.any(z[:,[0,-1],:]) for z in (q,v,a)):
            raise ValueError("MVI initial clamp violated")
        gamma=2-sqrt(2)
        delta=gamma*dt_s
        kappa=4/delta**2
        predictor=q+delta*v+delta*delta*a/4
        q1,fac1,loops1=self._linear_endpoint(
            predictor,time_s+delta,kappa,previous_profiles,max_sweeps)
        a1=kappa*(q1-predictor)
        v1=v+delta*(a+a1)/2
        c1=(1-gamma)/(gamma*dt_s)
        c2=-1/((1-gamma)*gamma*dt_s)
        c3=(2-gamma)/((1-gamma)*dt_s)
        predictor2=-(c3*(c1*q+c2*q1)+c1*v+c2*v1)/c3**2
        q2,fac2,loops2=self._linear_endpoint(
            predictor2,time_s+dt_s,c3**2,fac1,max_sweeps)
        v2=c1*q+c2*q1+c3*q2
        a2=c1*v+c2*v1+c3*v2
        if any(not np.isfinite(z).all() for z in (q1,v1,a1,q2,v2,a2)):
            raise RuntimeError("MVI nonfinite Bathe stage")
        return (q1,v1,a1),(q2,v2,a2),fac2,{"first":loops1,"second":loops2}


    def step_newmark(self,q,v,a,time_s,dt_s=.002, *,
                     previous_profiles=None,max_sweeps=35,beta=.25,gamma=.5):
        """Own implicit Newmark; spatial MVI cycles X/Y at each physical step.

        Newmark predictors are unrestricted physical fields; ONLY the
        displacement endpoint is kept on the rank-fixed spatial manifold.
        """
        c=self.case
        expect=(15,c.nx,c.ns)
        q,v,a=tuple(np.asarray(field,float) for field in (q,v,a))
        if any(field.shape!=expect or not np.isfinite(field).all()
               for field in (q,v,a)):
            raise ValueError("MVI_NEWMARK_INVALID_INITIAL_FIELDS")
        if (not np.isfinite(dt_s) or dt_s<=0 or
                not np.isclose(beta,.25) or not np.isclose(gamma,.5)):
            raise ValueError("MVI_NEWMARK_REQUIRES_AVERAGE_ACCELERATION")
        if any(np.any(field[:,[0,-1],:]) for field in (q,v,a)):
            raise ValueError("MVI_NEWMARK_CLAMPED_SUPPORT_VIOLATED")
        qp=q+dt_s*v+dt_s*dt_s*(.5-beta)*a
        vp=v+dt_s*(1-gamma)*a
        stiffness_factor=1/(beta*dt_s**2)
        qn,factors,history=self._linear_endpoint(
            qp,time_s+dt_s,stiffness_factor,
            previous=previous_profiles,max_sweeps=max_sweeps)
        an=stiffness_factor*(qn-qp)
        vn=vp+gamma*dt_s*an
        if not all(np.isfinite(arr).all() for arr in (qn,vn,an)):
            raise RuntimeError("MVI_NEWMARK_NONFINITE")
        return (qn,vn,an),factors,history

    def _recover_rank_factors(self,field,*,relative_tolerance=3.e-9):
        """Strict rank capacity certificate, never truncate an endpoint.

        A full endpoint made from multiple time-varying rank-R stages can
        have greater rank than R. Returning the stage-2 factors would be
        mathematically wrong. Reject unsupported endpoints instead.
        """
        q=np.asarray(field,float)
        c=self.case
        if q.shape!=(15,c.nx,c.ns) or not np.isfinite(q).all():
            raise ValueError("MVI_IRK4_INVALID_RANK_RECOVERY_FIELD")
        if np.max(np.abs(q[:,[0,-1],:]))>1.e-11:
            raise ValueError("MVI_IRK4_ENDPOINT_CLAMP_VIOLATED")
        x=np.zeros((15,self.rank,c.nx))
        y=np.zeros((15,self.rank,c.ns))
        for component in range(15):
            u,s,vh=np.linalg.svd(q[component],full_matrices=False)
            tail=float(np.linalg.norm(s[self.rank:]))
            total=float(np.linalg.norm(s))
            if tail>relative_tolerance*max(total,1.e-30):
                raise RuntimeError("MVI_IRK4_ENDPOINT_EXCEEDS_FIXED_RANK")
            count=min(self.rank,len(s))
            x[component,:count]=u[:,:count].T*np.sqrt(s[:count,None])
            y[component,:count]=vh[:count]*np.sqrt(s[:count,None])
        rec=self._field(x,y)
        if np.max(np.abs(rec-q))>relative_tolerance*max(
                float(np.max(np.abs(q))),1.e-30):
            raise RuntimeError("MVI_IRK4_ENDPOINT_NOT_REPRESENTABLE")
        return x,y

    def step_runge_kutta_gauss4(self,q,v,a,time_s,dt_s=.002, *,
                                previous_profiles=None,max_sweeps=45,
                                krylov_rtol=2.e-8,max_krylov_matvec=1800,
                                relative_change=2.e-7):
        """Own two-stage Gauss-Legendre IRK4 with X/Y on BOTH stage fields.

        For accelerations a_j, Gauss stages satisfy
          Q_i=q0+dt*c_i*v0+dt^2*sum_j (A^2)_ij*a_j.
        Invert A^2 to obtain a_j from Q_i, then enforce
          T_i(X_i,Y_i)^T [ M*a_i + K*Q_i - F_i ] = 0
        at each stage using frozen-partner variations; the two stages
        remain coupled in every linear half-step.

        End displacement is the exact Gauss endpoint combination.
        Never replace the physical endpoint with rank-R SVD truncation.
        If its rank exceeds R, fail explicitly without a partial result.
        """
        c=self.case
        shape=(15,c.nx,c.ns)
        q,v,a=tuple(np.asarray(z,float) for z in (q,v,a))
        if any(z.shape!=shape or not np.isfinite(z).all() for z in (q,v,a)):
            raise ValueError("MVI_IRK4_INVALID_INITIAL_FIELDS")
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MVI_IRK4_BAD_TIMESTEP")
        if any(np.any(z[:,[0,-1],:]) for z in (q,v,a)):
            raise ValueError("MVI_IRK4_CLAMPED_INITIAL_FIELDS")
        root3=np.sqrt(3.)
        nodes=np.array([.5-root3/6,.5+root3/6])
        A=np.array([[.25,.25-root3/6],[.25+root3/6,.25]])
        A2=A@A
        invA2=np.linalg.inv(A2)
        weights=np.array([.5,.5])
        if previous_profiles is None:
            factors=[self._initial(),self._initial()]
        else:
            factors=[tuple(z.copy() for z in previous_profiles)
                     for _ in range(2)]
        if any(x.shape!=(15,self.rank,c.nx) or
               y.shape!=(15,self.rank,c.ns) or
               np.any(x[:,:,[0,-1]])
               for x,y in factors):
            raise ValueError("MVI_IRK4_BAD_INITIAL_PROFILE_DIMENSIONS")
        initial=[q+dt_s*nodes[i]*v for i in range(2)]
        fstage=[self.weak_load(time_s+dt_s*nodes[i]) for i in range(2)]

        def stage_fields():
            return [self._field(x,y) for x,y in factors]

        def stage_accelerations(Q):
            return [sum(invA2[i,j]*(Q[j]-initial[j])
                        for j in range(2))/(dt_s**2)
                    for i in range(2)]

        def stage_equations(Q):
            acc=stage_accelerations(Q)
            return [self.weak_mass(acc[i])+self.weak_stiffness(Q[i])
                    -fstage[i] for i in range(2)]
        histories=[]
        converged=False
        for sweep in range(1,max_sweeps+1):
            baseline=stage_fields()
            for freeze,active in (("X","Y"),("Y","X")):
                for i in range(2):
                    factors[i]=self._canonicalize(
                        *factors[i],freeze)
                directions=[self._directions(*factors[i],active)
                            for i in range(2)]
                Q=stage_fields()
                eq=stage_equations(Q)
                g=np.concatenate([directions[i].T@eq[i].ravel()
                                  for i in range(2)])
                sizes=[mat.shape[1] for mat in directions]
                def action(z):
                    s0=z[:sizes[0]];s1=z[sizes[0]:]
                    dq=[(directions[0]@s0).reshape(shape),
                        (directions[1]@s1).reshape(shape)]
                    da=[sum(invA2[i,j]*dq[j] for j in range(2))/dt_s**2
                        for i in range(2)]
                    return np.concatenate([
                        directions[i].T@(
                            self.weak_mass(da[i])+
                            self.weak_stiffness(dq[i])).ravel()
                        for i in range(2)])
                if np.linalg.norm(g)<1.e-18:
                    continue
                n=len(g)
                operator=LinearOperator((n,n),matvec=action,dtype=float)
                restart=min(75,n)
                delta,info=gmres(operator,-g,rtol=krylov_rtol,atol=0.,
                                 restart=restart,
                                 maxiter=max(1,max_krylov_matvec//restart),
                                 callback_type="pr_norm")
                if info!=0 or not np.isfinite(delta).all():
                    raise RuntimeError("MVI_IRK4_STAGE_COUPLED_GMRES_FAILED")
                segments=(delta[:sizes[0]],delta[sizes[0]:])
                for i in range(2):
                    dx=segments[i].reshape(15,self.rank,-1)
                    x,y=factors[i]
                    if active=="Y": y=y+dx
                    else:
                        x=x.copy()
                        x[:,:,1:-1]+=dx
                    factors[i]=(x,y)
                histories.append((sweep,active))
            current=stage_fields()
            difference=sum(np.linalg.norm(current[i]-baseline[i])**2
                           for i in range(2))**.5
            magnitude=sum(np.linalg.norm(current[i])**2
                          for i in range(2))**.5
            if difference<=relative_change*max(magnitude,1.e-30):
                converged=True
                break
        if not converged:
            raise RuntimeError("MVI_IRK4_COUPLED_XY_NOT_CONVERGED")
        Q=stage_fields()
        acceleration=stage_accelerations(Q)
        V=[v+dt_s*sum(A[i,j]*acceleration[j] for j in range(2))
           for i in range(2)]
        endpoint=q+dt_s*v+dt_s**2*sum(
            sum(weights[i]*A[i,j] for i in range(2))*acceleration[j]
            for j in range(2))
        endpoint_factors=self._recover_rank_factors(endpoint)
        vend=v+dt_s*sum(weights[i]*acceleration[i] for i in range(2))
        # A(t_end) is full nodal and is not subject to spatial rank R.
        f_end=self.weak_load(time_s+dt_s)-self.weak_stiffness(endpoint)
        ea=f_end.reshape(3,5,c.nx,c.ns) / self.surface_weight[None,None]
        ea=np.linalg.solve(self.mass,ea.transpose(1,0,2,3).reshape(5,-1))
        aend=ea.reshape(5,3,c.nx,c.ns).transpose(1,0,2,3).reshape(shape)
        aend[:,[0,-1],:]=0.
        if any(not np.isfinite(z).all() for z in
               (endpoint,vend,aend,*Q,*V,*acceleration)):
            raise RuntimeError("MVI_IRK4_NONFINITE")
        return tuple(zip(Q,V,acceleration)),(
            endpoint,vend,aend),endpoint_factors,tuple(histories)


__all__=("VariationalCase","VariationalTE4")
