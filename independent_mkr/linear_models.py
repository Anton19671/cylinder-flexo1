"""Independent strong finite-difference linear shell operators for seven theories.

Separate from every MVI implementation, the historical variational residual,
and the existing TE4 reference module. Each continuous kinematics is used to
build an independent strong equilibrium difference operator.

Scope: linear isotropic homogeneous non-electrical KL, FSDT, Reddy and CUF
TE1..TE4. Supports prescribed homogeneous axial displacement and, for KL,
zero axial radial slope. Full production natural edge consistency needs
independent physical validation. No calculation occurs during import.
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
class StrongModelCase:
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
    x_interval_m: tuple[float,float] = (.3,1.6)
    theta_interval_rad: tuple[float,float] = (.45,3.1)
    x_edge_m: float = .35
    theta_edge_rad: float = .6

    def __post_init__(self):
        if self.theory not in MODELS:
            raise ValueError("INDEPENDENT_MKR_UNSUPPORTED_THEORY")
        if (type(self.nx) is not int or self.nx<9 or
                type(self.ns) is not int or self.ns<5 or self.ns%2==0):
            raise ValueError("INDEPENDENT_MKR_GRID_REQUIRES_NX9_ODD_NS5")
        vals=(self.length_m,self.radius_m,self.thickness_m,self.young_pa,
              self.density_kg_m3,self.pressure_pa,self.frequency_hz)
        if not all(np.isfinite(v) and v>0 for v in vals):
            raise ValueError("INDEPENDENT_MKR_NONPOSITIVE_PHYSICS")
        if self.radius_m<=self.thickness_m/2 or not -1<self.poisson<.5:
            raise ValueError("INDEPENDENT_MKR_INVALID_PHYSICS")
        if self.pressure_kind not in ("UNIFORM","PATCH"):
            raise ValueError("INDEPENDENT_MKR_UNSUPPORTED_LOAD")
        if (not 0<=self.x_interval_m[0]<self.x_interval_m[1]<=self.length_m
                or not 0<=self.theta_interval_rad[0]<
                self.theta_interval_rad[1]<=2*pi):
            raise ValueError("INDEPENDENT_MKR_INVALID_PATCH")


def _finite_difference_x(n,length):
    """Own centered/one-sided derivative, no SBP transpose weak form."""
    dx=length/(n-1)
    d=np.zeros((n,n))
    d[0,:3]=(-1.5,2.,-.5)
    d[-1,-3:]=(.5,-2.,1.5)
    for i in range(1,n-1):
        d[i,i-1]=-.5
        d[i,i+1]=.5
    return d/dx


def _finite_difference_s(n,radius):
    step=2*pi*radius/n
    d=np.zeros((n,n))
    for j in range(n):
        d[j,(j+1)%n]=.5/step
        d[j,(j-1)%n]=-.5/step
    return d


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


class IndependentStrongModel:
    def __init__(self,case:StrongModelCase):
        if not isinstance(case,StrongModelCase):
            raise TypeError("MKR model requires standalone StrongModelCase")
        self.case=case
        self.n=_ncomponents(case.theory)
        self.dx=_finite_difference_x(case.nx,case.length_m)
        self.ds=_finite_difference_s(case.ns,case.radius_m)
        self.deriv_x=(np.eye(case.nx),self.dx,self.dx@self.dx)
        self.deriv_s=(np.eye(case.ns),self.ds,self.ds@self.ds)
        self.stiffness=np.zeros((6,6,self.n,self.n))
        self.mass=np.zeros((3,3,self.n,self.n))
        self._integrate_thickness()
        self.basis=self._boundary_basis()
        self.dofs=tuple(b.shape[1]*case.ns for b in self.basis)
        self.force=self._loading()

    def _integrate_thickness(self):
        c=self.case
        lmb=c.young_pa*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu=c.young_pa/(2*(1+c.poisson))
        hooke=np.diag((2*mu,2*mu,2*mu,mu,mu,mu))
        hooke[:3,:3]+=lmb
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
                    raise ValueError("MKR_KL_EDGE_SLOPE_SUBSPACE_INVALID")
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
        radial=2 if c.theory in ("KL","FSDT","REDDY") else (
            2*(int(c.theory[-1])+1))
        q[radial]=-c.pressure_pa*window
        return q

    def _differentiate(self,q,order):
        a,b=ORDERS[order]
        return np.einsum("ij,cjk,lk->cil",self.deriv_x[a],q,
                         self.deriv_s[b],optimize=True)

    def _strong_form(self,q,tensor,terms):
        q=np.asarray(q,float)
        field=[self._differentiate(q,k) for k in range(terms)]
        result=np.zeros_like(q)
        for i in range(terms):
            flux=sum(np.einsum("ab,bxs->axs",tensor[i,j],field[j],
                              optimize=True) for j in range(terms))
            sign=(-1)**sum(ORDERS[i])
            result+=sign*self._differentiate(flux,i)
        return result

    def stiffness_action(self,q):
        return self._strong_form(q,self.stiffness,6)

    def mass_action(self,q):
        return self._strong_form(q,self.mass,3)

    def pack(self,q):
        c=self.case
        return np.concatenate([
            (self.basis[j].T@q[j]).ravel()
            for j in range(self.n)])

    def unpack(self,z):
        c=self.case
        vec=np.asarray(z,float)
        if vec.size!=sum(self.dofs):
            raise ValueError("MKR_MODEL_WRONG_STATE_DIMENSION")
        q=np.zeros((self.n,c.nx,c.ns))
        at=0
        for j,b in enumerate(self.basis):
            num=b.shape[1]*c.ns
            q[j]=b@vec[at:at+num].reshape(b.shape[1],c.ns)
            at+=num
        return q

    def _solve_linear(self,action,rhs,*,rtol=1.e-8,limit=2400):
        b=self.pack(rhs)
        n=b.size
        op=LinearOperator((n,n),matvec=lambda z:self.pack(
            action(self.unpack(z))),dtype=float)
        restart=min(80,n)
        found,status=gmres(op,b,rtol=rtol,atol=0.,restart=restart,
                           maxiter=max(1,limit//restart),
                           callback_type="pr_norm")
        if status!=0 or not np.isfinite(found).all():
            raise RuntimeError("MKR_MODEL_INDEPENDENT_KRYLOV_FAILED")
        return self.unpack(found)

    def _endpoint(self,predict,time_s,kappa):
        force=self.force*np.sin(2*pi*self.case.frequency_hz*time_s)
        return self._solve_linear(
            lambda q:self.stiffness_action(q)+kappa*self.mass_action(q),
            force+kappa*self.mass_action(predict))

    def step_newmark(self,q,v,a,time_s,dt_s=.002):
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MKR_MODEL_INVALID_TIME_STEP")
        qp=q+dt_s*v+.25*dt_s**2*a
        qn=self._endpoint(qp,time_s+dt_s,4/dt_s**2)
        an=4/dt_s**2*(qn-qp)
        vn=v+.5*dt_s*(a+an)
        return qn,vn,an

    def step_bathe(self,q,v,a,time_s,dt_s=.002):
        g=2-sqrt(2)
        d=g*dt_s
        qp=q+d*v+.25*d*d*a
        q1=self._endpoint(qp,time_s+d,4/(d*d))
        a1=4/(d*d)*(q1-qp)
        v1=v+.5*d*(a+a1)
        c1=(1-g)/(g*dt_s)
        c2=-1/((1-g)*g*dt_s)
        c3=(2-g)/((1-g)*dt_s)
        p2=-(c3*(c1*q+c2*q1)+c1*v+c2*v1)/(c3*c3)
        q2=self._endpoint(p2,time_s+dt_s,c3*c3)
        v2=c1*q+c2*q1+c3*q2
        a2=c1*v+c2*v1+c3*v2
        return (q1,v1,a1),(q2,v2,a2)

    def step_gauss_irk4(self,q,v,a,time_s,dt_s=.002):
        """Own coupled two-stage Gauss4 via full nodal strong differences."""
        if not np.isfinite(dt_s) or dt_s<=0:
            raise ValueError("MKR_MODEL_INVALID_IRK4_STEP")
        root=np.sqrt(3.)
        t=np.array([.5-root/6,.5+root/6])
        A=np.array([[.25,.25-root/6],[.25+root/6,.25]])
        A2=A@A
        f=[self.force*np.sin(2*pi*self.case.frequency_hz*(
             time_s+dt_s*t[i])) for i in range(2)]
        base=[q+dt_s*t[i]*v for i in range(2)]
        n=sum(self.dofs)
        rhs=np.concatenate([self.pack(
            f[i]-self.stiffness_action(base[i])) for i in range(2)])
        def action(z):
            acceleration=[self.unpack(z[i*n:(i+1)*n])
                          for i in range(2)]
            fields=[self.mass_action(acceleration[i])+
                    dt_s*dt_s*sum(A2[i,j]*
                        self.stiffness_action(acceleration[j])
                        for j in range(2)) for i in range(2)]
            return np.concatenate([self.pack(item) for item in fields])
        op=LinearOperator((2*n,2*n),matvec=action,dtype=float)
        sol,code=gmres(op,rhs,rtol=1.e-8,atol=0.,restart=min(90,2*n),
                       maxiter=45,callback_type="pr_norm")
        if code!=0 or not np.isfinite(sol).all():
            raise RuntimeError("MKR_MODEL_GAUSS_COUPLED_SOLVE_FAILED")
        acceleration=[self.unpack(sol[i*n:(i+1)*n]) for i in range(2)]
        stage=[(base[i]+dt_s*dt_s*sum(A2[i,j]*acceleration[j]
                    for j in range(2)),
                v+dt_s*sum(A[i,j]*acceleration[j] for j in range(2)),
                acceleration[i]) for i in range(2)]
        qend=q+dt_s*v+dt_s*dt_s*sum(
            sum(.5*A[i,j] for i in range(2))*acceleration[j]
            for j in range(2))
        vend=v+.5*dt_s*(acceleration[0]+acceleration[1])
        force=self.force*np.sin(2*pi*self.case.frequency_hz*(time_s+dt_s))
        aend=self._solve_linear(self.mass_action,
                                force-self.stiffness_action(qend))
        return tuple(stage),(qend,vend,aend)


__all__=("StrongModelCase","IndependentStrongModel","MODELS")
