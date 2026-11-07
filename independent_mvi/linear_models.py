"""Separate seven-theory CUF, KL, FSDT and Reddy MVI weak operators.

This implementation is self-contained and independent of MVI and the legacy
shared variational functional. Its own physics, stiffness, mass, loads,
clamped traces and separated X/Y solve apply to seven LINEAR models.
It contains NO full-grid spatial inverse or MVI dispatch.
Experimental; all physical benchmarks and higher edge work remain pending.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, log1p, sqrt
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, gmres

MODELS=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
ORDERS=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
ORDER_MAP={v:i for i,v in enumerate(ORDERS)}

@dataclass(frozen=True)
class VariationalModelCase:
    theory: str = "CUF_TE4"
    length_m: float = 2.
    radius_m: float = 1.
    thickness_m: float = .05
    young_pa: float = 210.e9
    poisson: float = .3
    density_kg_m3: float = 7800.
    nx: int = 11
    ns: int = 9
    pressure_pa: float = 100000.
    frequency_hz: float = 25.
    pressure_kind: str = "UNIFORM"
    pressure_face: str = "OUTER"
    x_interval_m: tuple[float,float] = (.3,1.6)
    theta_interval_rad: tuple[float,float] = (.45,3.1)
    x_edge_m: float = .35
    theta_edge_rad: float = .6

    def __post_init__(self):
        if self.theory not in MODELS:
            raise ValueError("INDEPENDENT_MVI_UNSUPPORTED_THEORY")
        if (type(self.nx) is not int or self.nx<9 or
                type(self.ns) is not int or self.ns<5 or self.ns%2==0):
            raise ValueError("INDEPENDENT_MVI_GRID_REQUIRES_NX9_ODD_NS5")
        vals=(self.length_m,self.radius_m,self.thickness_m,self.young_pa,
              self.density_kg_m3,self.pressure_pa,self.frequency_hz)
        if not all(np.isfinite(v) and v>0 for v in vals):
            raise ValueError("INDEPENDENT_MVI_NONPOSITIVE_PHYSICS")
        if self.radius_m<=self.thickness_m/2 or not -1<self.poisson<.5:
            raise ValueError("INDEPENDENT_MVI_INVALID_PHYSICS")
        if self.pressure_kind not in ("UNIFORM","PATCH"):
            raise ValueError("INDEPENDENT_MVI_UNSUPPORTED_LOAD")
        if self.pressure_face not in ("OUTER","MIDDLE"):
            raise ValueError("INDEPENDENT_MVI_UNSUPPORTED_REFERENCE_FACE")
        if (not 0<=self.x_interval_m[0]<self.x_interval_m[1]<=self.length_m
                or not 0<=self.theta_interval_rad[0]<
                self.theta_interval_rad[1]<=2*pi):
            raise ValueError("INDEPENDENT_MVI_INVALID_PATCH")


def _mvi_axial_sbp(n,length):
    """Independent SBP42 weak-form axial derivative."""
    d=length/(n-1)
    out=np.zeros((n,n))
    closure=(
        ((0,-24/17),(1,59/34),(2,-4/17),(3,-3/34)),
        ((0,-.5),(2,.5)),
        ((0,4/43),(1,-59/86),(3,59/86),(4,-4/43)),
        ((0,3/98),(2,-59/98),(4,32/49),(5,-4/49)),
    )
    for i in range(n):
        entries=(closure[i] if i<4 else
                 tuple((n-1-j,-v) for j,v in closure[n-1-i])
                 if i>=n-4 else
                 ((i-2,1/12),(i-1,-2/3),(i+1,2/3),(i+2,-1/12)))
        for j,v in entries:out[i,j]=v/d
    wx=np.ones(n)
    wx[:4]=[17/48,59/48,43/48,49/48]
    wx[-4:]=wx[:4][::-1]
    return out,wx*d


def _mvi_fourier(n,radius):
    """Independent exact trigonometric derivative, weighted periodic weak form."""
    length=2*pi*radius
    freq=2*pi*np.fft.fftfreq(n,d=length/n)
    out=np.fft.ifft(1j*freq[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
    return .5*(out-out.T),np.full(n,length/n)


def _reddy_warps(z,R,h):
    """Independent curvature-corrected Reddy axial and tangent warping."""
    ax=-4*z**3/(3*h*h)
    axp=-4*z*z/(h*h)
    t=z/R
    if abs(t)<.08:
        series=sum(((-1)**(k+1))*(k-2)/k*t**k for k in range(3,25))
        antiderivative=R*series
    else:
        antiderivative=R*(t+t/(1+t)-2*log1p(t))
    bs=-4*R*(R+z)*antiderivative/(h*h)
    bsp=(bs/R-4*z*z/(h*h))/(1+z/R)
    return ax,axp,bs,bsp


def _ncomponents(theory):
    return 3 if theory=="KL" else 5 if theory in ("FSDT","REDDY") else (
        3*(int(theory[-1])+1))


def _kinematic_coefficients(theory,z,R,h):
    """Three physical displacement maps C0,Cx,Cs and z derivatives."""
    n=_ncomponents(theory)
    c=np.zeros((3,3,n))
    dz=np.zeros_like(c)
    if theory.startswith("CUF_TE"):
        m=int(theory[-1])+1
        for k in range(m):
            f=(2*z/h)**k
            fp=0. if k==0 else k*(2/h)*(2*z/h)**(k-1)
            for j in range(3):
                c[0,j,j*m+k]=f
                dz[0,j,j*m+k]=fp
    elif theory=="KL":
        # Linear KL: U=u-z*w_x, V=(1+z/R)v-z*w_s, W=w.
        c[0,0,0]=1
        c[0,1,1]=1+z/R
        c[0,2,2]=1
        c[1,0,2]=-z
        c[2,1,2]=-z
        dz[0,1,1]=1/R
        dz[1,0,2]=-1
        dz[2,1,2]=-1
    elif theory=="FSDT":
        c[0,0,0]=c[0,1,1]=c[0,2,2]=1
        c[0,0,3]=c[0,1,4]=z
        dz[0,0,3]=dz[0,1,4]=1
    elif theory=="REDDY":
        ax,axp,bs,bsp=_reddy_warps(z,R,h)
        c[0,0,0]=1
        c[0,1,1]=1-bs/R
        c[0,2,2]=1
        c[0,0,3]=z+ax
        c[0,1,4]=z+bs
        c[1,0,2]=ax
        c[2,1,2]=bs
        dz[0,1,1]=-bsp/R
        dz[0,0,3]=1+axp
        dz[0,1,4]=1+bsp
        dz[1,0,2]=axp
        dz[2,1,2]=bsp
    return c,dz


def _strain_matrices(c,dz,R,z):
    """Six independent engineering strains from cylindrical local geometry."""
    hfactor=1+z/R
    b=np.zeros((6,6,c.shape[-1]))
    for index,(ax,ang) in enumerate(ORDERS[:3]):
        coeff=c[index]
        deriv=dz[index]
        def insert(row,da,ds,v):
            b[ORDER_MAP[(da,ds)],row]+=v
        # Exx = d_x U
        insert(0,ax+1,ang,coeff[0])
        # Ess = (d_s V + W/R) / H
        insert(1,ax,ang+1,coeff[1]/hfactor)
        insert(1,ax,ang,coeff[2]/(R*hfactor))
        # Ezz = d_z W
        insert(2,ax,ang,deriv[2])
        # gamma_xs = d_x V + d_s U/H
        insert(3,ax+1,ang,coeff[1])
        insert(3,ax,ang+1,coeff[0]/hfactor)
        # gamma_xz = d_z U + d_x W
        insert(4,ax,ang,deriv[0])
        insert(4,ax+1,ang,coeff[2])
        # gamma_sz = d_z V + (d_s W-V/R)/H
        insert(5,ax,ang,deriv[1]-coeff[1]/(R*hfactor))
        insert(5,ax,ang+1,coeff[2]/hfactor)
    return b


def _elastic_tensor(theory, young_pa, poisson):
    """Reduced plane stress for classical shells; full 3D for CUF thickness stretch.

    The KL/FSDT/Reddy kinematics do not retain independent thickness strain.
    Eliminate it from 3D isotropic energy by the sigma_zz=0 condition.
    CUF TE1..4 retains independent z-dependent displacement components.
    """
    mu=young_pa/(2*(1+poisson))
    if theory in ("KL","FSDT","REDDY"):
        lam_ps=young_pa*poisson/(1-poisson**2)
        hooke=np.diag((2*mu,2*mu,0.,mu,mu,mu))
        hooke[:2,:2]+=lam_ps
    else:
        lam=young_pa*poisson/((1+poisson)*(1-2*poisson))
        hooke=np.diag((2*mu,2*mu,2*mu,mu,mu,mu))
        hooke[:3,:3]+=lam
    return hooke


class IndependentVariationalModel:
    def __init__(self,case:VariationalModelCase,rank:int):
        if not isinstance(case,VariationalModelCase):
            raise TypeError("MVI model requires standalone VariationalModelCase")
        if type(rank) is not int or not 1<=rank<=8:
            raise ValueError("MVI_RANK_OUTSIDE_1_TO_8")
        self.case=case
        self.rank=rank
        self.n=_ncomponents(case.theory)
        self.dx,self.wx=_mvi_axial_sbp(case.nx,case.length_m)
        self.ds,self.ws=_mvi_fourier(case.ns,case.radius_m)
        self.area=np.outer(self.wx,self.ws)
        self.deriv_x=(np.eye(case.nx),self.dx,self.dx@self.dx)
        self.deriv_s=(np.eye(case.ns),self.ds,self.ds@self.ds)
        self.stiffness=np.zeros((6,6,self.n,self.n))
        self.mass=np.zeros((3,3,self.n,self.n))
        self._integrate_thickness()
        self.basis=self._boundary_basis()
        if rank>min(case.ns,*(b.shape[1] for b in self.basis)):
            raise ValueError("MVI_RANK_EXCEEDS_CLAMPED_FIELD_CAPACITY")
        self.dofs=tuple(b.shape[1]*case.ns for b in self.basis)
        self.force=self._loading()

    def _integrate_thickness(self):
        c=self.case
        hooke=_elastic_tensor(c.theory,c.young_pa,c.poisson)
        zeta,wts=np.polynomial.legendre.leggauss(12)
        for xi,wi in zip(zeta,wts):
            z=.5*c.thickness_m*xi
            H=1+z/c.radius_m
            scale=.5*c.thickness_m*wi*H
            displacement,dz=_kinematic_coefficients(
                c.theory,z,c.radius_m,c.thickness_m)
            b=_strain_matrices(displacement,dz,c.radius_m,z)
            for i in range(6):
                for j in range(6):
                    self.stiffness[i,j]+=scale*b[i].T@hooke@b[j]
            for i in range(3):
                for j in range(3):
                    self.mass[i,j]+=scale*c.density_kg_m3*(
                        displacement[i].T@displacement[j])

    def _boundary_basis(self):
        n=self.case.nx
        standard=np.eye(n)[:,1:-1]
        basis=[]
        for index in range(self.n):
            if self.case.theory=="KL" and index==2:
                # Full CLAMPED KL: w=0 and w_x=0 at each axial end.
                edge=self.dx[[0,-1],:]@standard
                ns=linalg.null_space(edge, rcond=1.e-12)
                if ns.shape[1]!=n-4:
                    raise ValueError("MVI_KL_EDGE_SLOPE_SUBSPACE_INVALID")
                basis.append(standard@ns)
            else:
                basis.append(standard.copy())
        return tuple(basis)

    def _loading(self):
        c=self.case
        window=np.ones((c.nx,c.ns))
        if c.pressure_kind=="PATCH":
            def front(d,width):
                t=np.clip(d/width+.5,0.,1.)
                return np.sin(.5*pi*t)**2
            x=np.linspace(0,c.length_m,c.nx)
            s=2*pi*np.arange(c.ns)/c.ns
            a,b=c.x_interval_m
            lo,hi=c.theta_interval_rad
            axial=front(x-a,c.x_edge_m)*front(b-x,c.x_edge_m)
            d=(s-(lo+hi)/2+pi)%(2*pi)-pi
            angular=front((hi-lo)/2-np.abs(d),c.theta_edge_rad)
            window=np.outer(axial,angular)
        q=np.zeros((self.n,c.nx,c.ns))
        # Preserve the actual nominal reference load face. For CUF:
        # F_k(h/2)=1 on OUTER, F_k(0)=delta_{k0} on MIDDLE.
        # These are distinct physical loads, never interchangeable.
        face_factor=(1.+c.thickness_m/(2.*c.radius_m)
                     if c.pressure_face=="OUTER" else 1.)
        face_force=-c.pressure_pa*face_factor*window
        if c.theory in ("KL","FSDT","REDDY"):
            q[2]=face_force
        else:
            order=int(c.theory[-1])+1
            if c.pressure_face=="OUTER":
                q[2*order:3*order]=face_force[None,:,:]
            else:
                q[2*order]=face_force
        return q

    def _differentiate(self,q,order):
        a,b=ORDERS[order]
        return np.einsum("ij,cjk,lk->cil",self.deriv_x[a],q,
                         self.deriv_s[b],optimize=True)

    def _weak_form(self,q,tensor,terms):
        """Own exact-discrete 1st variation, weighted adjoints at each order."""
        q=np.asarray(q,float)
        field=[self._differentiate(q,k) for k in range(terms)]
        result=np.zeros_like(q)
        for i in range(terms):
            flux=sum(np.einsum("ab,bxs->axs",tensor[i,j],field[j],
                              optimize=True) for j in range(terms))
            a,b=ORDERS[i]
            result+=np.einsum("ji,cjk,kl->cil",self.deriv_x[a],
                              self.area[None]*flux,self.deriv_s[b],optimize=True)
        return result

    def stiffness_action(self,q):
        return self._weak_form(q,self.stiffness,6)

    def mass_action(self,q):
        return self._weak_form(q,self.mass,3)


    def pack(self,q):
        c=self.case
        return np.concatenate([
            (self.basis[j].T@q[j]).ravel()
            for j in range(self.n)])

    def unpack(self,z):
        c=self.case
        vec=np.asarray(z,float)
        if vec.size!=sum(self.dofs):
            raise ValueError("MVI_MODEL_WRONG_STATE_DIMENSION")
        q=np.zeros((self.n,c.nx,c.ns))
        at=0
        for j,b in enumerate(self.basis):
            num=b.shape[1]*c.ns
            q[j]=b@vec[at:at+num].reshape(b.shape[1],c.ns)
            at+=num
        return q


    def _first_factors(self):
        c=self.case
        X=np.zeros((self.n,self.rank,c.nx))
        Y=np.zeros((self.n,self.rank,c.ns))
        for k,B in enumerate(self.basis):
            X[k]=B[:,:self.rank].T
        return X,Y

    @staticmethod
    def field(X,Y):
        return np.einsum("krx,krs->kxs",X,Y,optimize=True)

    def _normalize(self,X,Y,freeze):
        X,Y=X.copy(),Y.copy()
        for component,B in enumerate(self.basis):
            if freeze=="X":
                Q,R=np.linalg.qr(B.T@X[component].T,mode="reduced")
                X[component]=(B@Q).T
                Y[component]=R@Y[component]
            elif freeze=="Y":
                Q,R=np.linalg.qr(Y[component].T,mode="reduced")
                Y[component]=Q.T
                X[component]=R@X[component]
            else:
                raise ValueError("MVI_INVALID_FROZEN_PARTNER")
        return X,Y

    def _directions(self,X,Y,active):
        c=self.case
        rows=self.n*c.nx*c.ns
        widths=[self.rank*(c.ns if active=="Y" else B.shape[1])
                for B in self.basis]
        T=np.zeros((rows,sum(widths)))
        cursor=0
        stride=c.nx*c.ns
        for k,B in enumerate(self.basis):
            for r in range(self.rank):
                if active=="Y":
                    for s in range(c.ns):
                        temp=np.zeros((c.nx,c.ns))
                        temp[:,s]=X[k,r]
                        T[k*stride:(k+1)*stride,cursor]=temp.ravel()
                        cursor+=1
                else:
                    for j in range(B.shape[1]):
                        T[k*stride:(k+1)*stride,cursor]=(
                            B[:,j,None]*Y[k,r,None,:]).ravel()
                        cursor+=1
        return T,widths

    def _update_factors(self,X,Y,delta,widths,active):
        offset=0
        for k,B in enumerate(self.basis):
            row=delta[offset:offset+widths[k]]
            offset+=widths[k]
            if active=="Y":
                Y[k]+=row.reshape(self.rank,self.case.ns)
            else:
                X[k]+=row.reshape(self.rank,B.shape[1])@B.T
        if offset!=len(delta):
            raise ValueError("MVI_FACTOR_COORDINATE_SIZE")
        return X,Y

    def _effective_step(self,predict,time_s,kappa,previous=None,
                        *,max_sweeps=30,rtol=1.e-7):
        if previous is None:
            X,Y=self._first_factors()
        else:
            X,Y=(np.asarray(v,float).copy() for v in previous)
        rhs=self.area[None]*self.force*np.sin(
            2*pi*self.case.frequency_hz*time_s)+kappa*self.mass_action(predict)
        def effective(field):
            return self.stiffness_action(field)+kappa*self.mass_action(field)
        progress=[]
        for sweep in range(max_sweeps):
            before=self.field(X,Y)
            for freeze,active in (("X","Y"),("Y","X")):
                X,Y=self._normalize(X,Y,freeze)
                T,widths=self._directions(X,Y,active)
                source=T.T@(rhs-effective(self.field(X,Y))).ravel()
                if np.linalg.norm(source)<=1.e-20:
                    continue
                n=len(source)
                operator=LinearOperator((n,n),matvec=lambda z:T.T@
                    effective((T@z).reshape(
                        self.n,self.case.nx,self.case.ns)).ravel(),
                    dtype=float)
                restart=min(80,n)
                delta,status=gmres(
                    operator,source,rtol=rtol,atol=0.,
                    restart=restart,maxiter=28,callback_type="pr_norm")
                if status!=0 or not np.isfinite(delta).all():
                    raise RuntimeError("MVI_FROZEN_PARTNER_SOLVE_FAILED")
                X,Y=self._update_factors(X,Y,delta,widths,active)
                progress.append((sweep+1,active))
            after=self.field(X,Y)
            if np.linalg.norm(after-before)<=2.e-7*max(
                    np.linalg.norm(after),1.e-30):
                return after,(X,Y),tuple(progress)
        raise RuntimeError("MVI_XY_CYCLES_NOT_COMPLETED")

    def step_newmark(self,q,v,a,t,dt_s=.002,previous=None):
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MVI_TIME_STEP_INVALID")
        predictor=q+dt_s*v+.25*dt_s**2*a
        qend,factors,history=self._effective_step(
            predictor,t+dt_s,4/dt_s**2,previous)
        aend=4/dt_s**2*(qend-predictor)
        vend=v+.5*dt_s*(a+aend)
        return (qend,vend,aend),factors,history

    def step_bathe(self,q,v,a,t,dt_s=.002,previous=None):
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MVI_TIME_STEP_INVALID")
        gamma=2-sqrt(2)
        delta=gamma*dt_s
        predictor=q+delta*v+.25*delta**2*a
        q1,f1,h1=self._effective_step(predictor,t+delta,
                                     4/delta**2,previous)
        a1=4/delta**2*(q1-predictor)
        v1=v+.5*delta*(a+a1)
        c1=(1-gamma)/(gamma*dt_s)
        c2=-1/((1-gamma)*gamma*dt_s)
        c3=(2-gamma)/((1-gamma)*dt_s)
        p2=-(c3*(c1*q+c2*q1)+c1*v+c2*v1)/c3**2
        q2,f2,h2=self._effective_step(p2,t+dt_s,c3**2,f1)
        v2=c1*q+c2*q1+c3*q2
        a2=c1*v+c2*v1+c3*v2
        return (q1,v1,a1),(q2,v2,a2),f2,(h1,h2)

    def _endpoint_factors(self,field,tolerance=3.e-9):
        c=self.case
        X=np.zeros((self.n,self.rank,c.nx))
        Y=np.zeros((self.n,self.rank,c.ns))
        for k,B in enumerate(self.basis):
            projected=B.T@field[k]
            if np.linalg.norm(field[k]-B@projected)>tolerance*max(
                    np.linalg.norm(field[k]),1.e-30):
                raise RuntimeError("MVI_GAUSS_ENDPOINT_CONSTRAINT_FAILURE")
            U,s,V=np.linalg.svd(projected,full_matrices=False)
            if np.linalg.norm(s[self.rank:])>tolerance*max(
                    np.linalg.norm(s),1.e-30):
                raise RuntimeError("MVI_GAUSS_ENDPOINT_RANK_FAILURE")
            number=min(self.rank,len(s))
            X[k,:number]=(B@U[:,:number]).T*np.sqrt(s[:number,None])
            Y[k,:number]=V[:number]*np.sqrt(s[:number,None])
        return X,Y

    def step_gauss_irk4(self,q,v,a,t,dt_s=.002,previous=None,
                        *,max_sweeps=40):
        """Two coupled Gauss stages, each constrained by own frozen-partner MVI."""
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MVI_GAUSS_BAD_STEP")
        root=np.sqrt(3)
        nodes=np.array([.5-root/6,.5+root/6])
        A=np.array([[.25,.25-root/6],[.25+root/6,.25]])
        A2=A@A
        inv=np.linalg.inv(A2)
        base=[q+dt_s*nodes[i]*v for i in range(2)]
        force=[self.area[None]*self.force*np.sin(
            2*pi*self.case.frequency_hz*(t+dt_s*nodes[i]))
            for i in range(2)]
        factor=[self._first_factors() if previous is None else
                tuple(np.asarray(z,float).copy() for z in previous)
                for _ in range(2)]
        shape=(self.n,self.case.nx,self.case.ns)
        def fields():
            return [self.field(*pair) for pair in factor]
        def acc(Q):
            return [sum(inv[i,j]*(Q[j]-base[j]) for j in range(2))/dt_s**2
                    for i in range(2)]
        history=[]
        for sweep in range(max_sweeps):
            before=fields()
            for freeze,active in (("X","Y"),("Y","X")):
                matrices=[];widths=[]
                for i in range(2):
                    factor[i]=self._normalize(*factor[i],freeze)
                    T,wd=self._directions(*factor[i],active)
                    matrices.append(T);widths.append(wd)
                Q=fields();acceleration=acc(Q)
                g=np.concatenate([
                    matrices[i].T@(
                        self.mass_action(acceleration[i])+
                        self.stiffness_action(Q[i])-force[i]).ravel()
                    for i in range(2)])
                if np.linalg.norm(g)<1.e-20:continue
                n0=matrices[0].shape[1]
                def operator_action(z):
                    dz=(z[:n0],z[n0:])
                    dq=[(matrices[i]@dz[i]).reshape(shape)
                        for i in range(2)]
                    da=[sum(inv[i,j]*dq[j] for j in range(2))/dt_s**2
                        for i in range(2)]
                    return np.concatenate([
                        matrices[i].T@(
                            self.mass_action(da[i])+
                            self.stiffness_action(dq[i])).ravel()
                        for i in range(2)])
                n=len(g)
                operator=LinearOperator((n,n),matvec=operator_action,dtype=float)
                change,status=gmres(operator,-g,rtol=1.e-7,atol=0.,
                                    restart=min(n,80),maxiter=32,
                                    callback_type="pr_norm")
                if status!=0 or not np.isfinite(change).all():
                    raise RuntimeError("MVI_GAUSS_STAGE_SOLVE_FAILED")
                for i,part in enumerate((change[:n0],change[n0:])):
                    factor[i]=self._update_factors(
                        *factor[i],part,widths[i],active)
                history.append((sweep+1,active))
            current=fields()
            difference=sum(np.linalg.norm(
                current[i]-before[i])**2 for i in range(2))**.5
            if difference<=2.e-7*max(
                sum(np.linalg.norm(item)**2 for item in current)**.5,
                1.e-30):
                break
        else:
            raise RuntimeError("MVI_GAUSS_XY_NOT_COMPLETED")
        Q=fields()
        acceleration=acc(Q)
        V=[v+dt_s*sum(A[i,j]*acceleration[j] for j in range(2))
           for i in range(2)]
        endpoint=q+dt_s*v+dt_s**2*sum(
            sum(.5*A[i,j] for i in range(2))*acceleration[j]
            for j in range(2))
        factors=self._endpoint_factors(endpoint)
        vend=v+.5*dt_s*(acceleration[0]+acceleration[1])
        # Deliberate no-bypass: inverse consistent mass in the same
        # independent variational and constrained subspace.
        rhs=self.area[None]*self.force*np.sin(
            2*pi*self.case.frequency_hz*(t+dt_s))-self.stiffness_action(endpoint)
        n=sum(self.dofs)
        operator=LinearOperator((n,n),matvec=lambda z:self.pack(
            self.mass_action(self.unpack(z))),dtype=float)
        val,status=gmres(operator,self.pack(rhs),rtol=1.e-9,atol=0.,
                         restart=min(n,80),maxiter=35,
                         callback_type="pr_norm")
        if status!=0 or not np.isfinite(val).all():
            raise RuntimeError("MVI_GAUSS_FINAL_MASS_FAILED")
        aend=self.unpack(val)
        return tuple(zip(Q,V,acceleration)),(
            endpoint,vend,aend),factors,tuple(history)


__all__=("VariationalModelCase","IndependentVariationalModel","MODELS")
