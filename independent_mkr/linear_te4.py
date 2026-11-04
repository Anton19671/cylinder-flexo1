"""Standalone finite-difference STRONG-FORM CUF TE4 shell dynamics.

This is an independent MKR/FDM discretization, not the full-node solution
of the MVI weak functional. It imports only numpy/scipy and the Python
standard library, and has its own physical operator, mass, loads, clamping
and Bathe scheme. This opt-in backend is restricted to linear isotropic,
homogeneous, non-electrical CUF TE4; unsupported features fail closed.

Coordinates: x in [0,L], s=R*theta periodic, z in [-h/2,h/2].
DOF order: U0..U4,V0..V4,W0..W4.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class PhysicalCase:
    length_m: float = 2.
    radius_m: float = 1.
    thickness_m: float = .05
    young_pa: float = 210e9
    poisson: float = .3
    density_kg_m3: float = 7800.
    nx: int = 9
    ns: int = 9
    pressure_pa: float = 100000.
    frequency_hz: float = 25.
    pressure_kind: str = "UNIFORM"  # UNIFORM or PATCH
    x_interval_m: tuple[float, float] = (.3, 1.6)
    theta_interval_rad: tuple[float, float] = (.45, 3.1)
    x_edge_m: float = .35
    theta_edge_rad: float = .6

    def __post_init__(self):
        if self.nx < 5 or self.ns < 5 or self.ns % 2 == 0:
            raise ValueError("MKR requires nx>=5 and odd ns>=5")
        if not (self.length_m > 0 and self.radius_m > self.thickness_m/2 > 0):
            raise ValueError("invalid cylindrical reference geometry")
        if (self.young_pa <= 0 or not -1. < self.poisson < .5 or
                self.density_kg_m3 <= 0 or self.frequency_hz <= 0 or
                self.pressure_pa <= 0):
            raise ValueError("invalid isotropic physical input")
        if self.pressure_kind not in ("UNIFORM", "PATCH"):
            raise ValueError("unsupported independent MKR pressure")
        if (not 0 <= self.x_interval_m[0] < self.x_interval_m[1] <= self.length_m or
                not 0 <= self.theta_interval_rad[0] <
                    self.theta_interval_rad[1] <= 2*pi):
            raise ValueError("patch intervals outside shell")


def _axial_fd(n: int, length: float) -> np.ndarray:
    """Independent three-point centered finite differences and one-sided edges."""
    d=length/(n-1)
    m=np.zeros((n,n),float)
    m[0,:3]=[-1.5,2.,-.5]
    m[-1,-3:]=[.5,-2.,1.5]
    for i in range(1,n-1):
        m[i,i-1]=-.5
        m[i,i+1]=.5
    return m/d


def _periodic_fd(n: int, radius: float) -> np.ndarray:
    d=2*pi*radius/n
    m=np.zeros((n,n),float)
    for j in range(n):
        m[j,(j+1)%n]+=.5/d
        m[j,(j-1)%n]-=.5/d
    return m


def _project_pressure(case: PhysicalCase) -> np.ndarray:
    x=np.linspace(0.,case.length_m,case.nx)
    theta=2*pi*np.arange(case.ns)/case.ns
    indicator=np.ones((case.nx,case.ns))
    if case.pressure_kind=="PATCH":
        def front(value,width):
            t=np.clip(value/width+.5,0.,1.)
            return np.sin(.5*pi*t)**2
        a,b=case.x_interval_m
        axial=front(x-a,case.x_edge_m)*front(b-x,case.x_edge_m)
        lo,hi=case.theta_interval_rad
        center=.5*(lo+hi)
        periodic=(theta-center+pi)%(2*pi)-pi
        signed=(hi-lo)/2-np.abs(periodic)
        angular=front(signed,case.theta_edge_rad)
        indicator=np.outer(axial,angular)
    out=np.zeros((15,case.nx,case.ns))
    out[10]=-case.pressure_pa*indicator
    return out


class StrongDifferenceTE4:
    """Independent strong discrete divergence of 3D Hooke CUF resultants."""

    def __init__(self, case: PhysicalCase):
        if not isinstance(case,PhysicalCase):
            raise TypeError("MKR requires independent PhysicalCase")
        self.case=case
        self.dx=_axial_fd(case.nx,case.length_m)
        self.ds=_periodic_fd(case.ns,case.radius_m)
        self.force=_project_pressure(case)
        self.xi,self.wi=np.polynomial.legendre.leggauss(10)
        self.lambda_lame=(case.young_pa*case.poisson /
                          ((1+case.poisson)*(1-2*case.poisson)))
        self.mu=case.young_pa/(2*(1+case.poisson))
        self.mass=self._mass_matrix()
        self.stiffness_blocks=self._constitutive_integrals()

    def _thickness(self):
        h=self.case.thickness_m
        for xi,weight in zip(self.xi,self.wi):
            z=h*xi/2
            yield xi,z,h*weight/2,1+z/self.case.radius_m

    def _mass_matrix(self):
        rho=self.case.density_kg_m3
        a=np.zeros((5,5))
        for xi,z,weight,H in self._thickness():
            b=xi**np.arange(5)
            a+=rho*weight*H*np.outer(b,b)
        return a

    def _constitutive_integrals(self):
        """Explicit six strain components in the independent STRONG operator."""
        c=self.case
        lam,mu=self.lambda_lame,self.mu
        d=np.diag([2*mu]*3+[mu]*3)
        d[:3,:3]+=lam
        matrices={(a,b):np.zeros((15,15)) for a in (0,1,2) for b in (0,1,2)}
        for xi,z,weight,H in self._thickness():
            f=xi**np.arange(5)
            fp=np.array([0. if k==0 else k*(2/c.thickness_m)*xi**(k-1)
                         for k in range(5)])
            bs=[np.zeros((6,15)) for _ in range(3)]
            b0,bx,bsurf=bs
            for k in range(5):
                bx[0,k]=f[k]
                bsurf[1,5+k]=f[k]/H
                b0[1,10+k]=f[k]/(c.radius_m*H)
                b0[2,10+k]=fp[k]
                bx[3,5+k]=f[k]
                bsurf[3,k]=f[k]/H
                bx[4,10+k]=f[k]
                b0[4,k]=fp[k]
                b0[5,5+k]=fp[k]-f[k]/(c.radius_m*H)
                bsurf[5,10+k]=f[k]/H
            for i in range(3):
                for j in range(3):
                    matrices[i,j]+=weight*H*(bs[i].T@d@bs[j])
        return matrices

    def _x(self,q):
        return np.einsum("ij,cjs->cis",self.dx,q,optimize=True)

    def _s(self,q):
        return np.einsum("ij,cxi->cxj",self.ds,q,optimize=True)

    def elastic_strong_action(self,q):
        """C0 - d(Ax)/dx - d(As)/ds, NOT a quadrature-adjoint gradient."""
        q=np.asarray(q,float)
        if q.shape!=(15,self.case.nx,self.case.ns):
            raise ValueError("MKR incompatible mechanical field")
        dqdx,dqds=self._x(q),self._s(q)
        b=self.stiffness_blocks
        def product(k):
            return (np.einsum("ab,bxs->axs",b[k,0],q,optimize=True)+
                    np.einsum("ab,bxs->axs",b[k,1],dqdx,optimize=True)+
                    np.einsum("ab,bxs->axs",b[k,2],dqds,optimize=True))
        return product(0)-self._x(product(1))-self._s(product(2))

    def inertia_action(self,q):
        a=np.asarray(q,float).reshape(3,5,self.case.nx,self.case.ns)
        return np.einsum("ij,cjxs->cixs",self.mass,a,optimize=True).reshape(q.shape)

    def _free_state(self,z):
        q=np.zeros((15,self.case.nx,self.case.ns))
        q[:,1:-1,:]=np.asarray(z,float).reshape(15,self.case.nx-2,self.case.ns)
        return q

    def _free_vector(self,q):
        return np.asarray(q,float)[:,1:-1,:].ravel()

    def _solve_endpoint(self,predictor,time_s,kappa,rtol=1.e-8,max_matvec=1600):
        from scipy.sparse.linalg import gmres
        rhs=(self.force*np.sin(2*pi*self.case.frequency_hz*time_s)+
             kappa*self.inertia_action(predictor))
        b=self._free_vector(rhs)
        size=b.size
        calls=[0]
        def action(z):
            calls[0]+=1
            return self._free_vector(
                self.elastic_strong_action(self._free_state(z))+
                kappa*self.inertia_action(self._free_state(z)))
        operator=LinearOperator((size,size),matvec=action,dtype=float)
        answer,info=gmres(operator,b,rtol=rtol,atol=0.,
                          restart=min(80,size),maxiter=max_matvec//min(80,size),
                          callback_type="pr_norm")
        if info!=0 or not np.isfinite(answer).all():
            raise RuntimeError("MKR_INDEPENDENT_GMRES_NOT_CONVERGED")
        if np.linalg.norm(action(answer)-b)>2*rtol*max(np.linalg.norm(b),1.e-30):
            raise RuntimeError("MKR_INDEPENDENT_LINEAR_SYSTEM_NOT_SOLVED")
        return self._free_state(answer)

    def step_bathe(self,q,v,a,time_s,dt_s=.002):
        """Independent direct Bathe three-point integration of strong FD ODEs."""
        q,v,a=(np.asarray(u,float) for u in (q,v,a))
        expect=(15,self.case.nx,self.case.ns)
        if any(u.shape!=expect for u in (q,v,a)) or dt_s<=0:
            raise ValueError("MKR invalid independent Bathe initial state")
        if any(np.any(u[:,[0,-1],:]) for u in (q,v,a)):
            raise ValueError("MKR homogeneous clamp violated")
        g=2-sqrt(2)
        delta=g*dt_s
        kappa=4/delta**2
        predictor=q+delta*v+delta**2*a/4
        q1=self._solve_endpoint(predictor,time_s+delta,kappa)
        a1=kappa*(q1-predictor)
        v1=v+delta*(a+a1)/2
        c1=(1-g)/(g*dt_s)
        c2=-1/((1-g)*g*dt_s)
        c3=(2-g)/((1-g)*dt_s)
        # The second stage uses the literal three-point Bathe formula,
        # rather than importing the MVI Newmark predictor routine.
        p2=-(c3*(c1*q+c2*q1)+c1*v+c2*v1)/(c3*c3)
        q2=self._solve_endpoint(p2,time_s+dt_s,c3*c3)
        v2=c1*q+c2*q1+c3*q2
        a2=c1*v+c2*v1+c3*v2
        for u in (q1,v1,a1,q2,v2,a2):
            if not np.isfinite(u).all():
                raise RuntimeError("MKR nonfinite physical Bathe stage")
        return (q1,v1,a1),(q2,v2,a2)


__all__=("PhysicalCase","StrongDifferenceTE4")
