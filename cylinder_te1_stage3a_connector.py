
from __future__ import annotations
from dataclasses import dataclass
import math, numpy as np
import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2

def _fd_weights(x0,xs,order):
    # Z1: solve the Vandermonde system in dimensionless local coordinates.
    # This leaves the FD formula unchanged but prevents cond(V) from growing
    # like h^(-m) purely because the physical unit/mesh scale changed.
    xs=np.asarray(xs,float); m=len(xs)
    if order>=m: raise ValueError("stencil too short")
    dx=xs-float(x0)
    hc=float(np.max(np.abs(dx)))
    if not np.isfinite(hc) or hc==0.0:
        raise ValueError("degenerate stencil")
    t=dx/hc
    A=np.vstack([t**k for k in range(m)])
    rhs=np.zeros(m); rhs[order]=math.factorial(order)
    return np.linalg.solve(A,rhs)/(hc**order)

def derivative_matrix_nonperiodic(x,order=1,stencil=7):
    x=np.asarray(x,float); N=len(x); s=min(int(stencil),N)
    if s<=order: raise ValueError("stencil too short")
    D=np.zeros((N,N)); half=s//2
    for i in range(N):
        left=max(0,min(i-half,N-s)); ids=np.arange(left,left+s)
        D[i,ids]=_fd_weights(x[i],x[ids],order)
    return D

def derivative_matrix_periodic_uniform(length,n,order=1,stencil=7):
    length=float(length); n=int(n); stencil=int(stencil)
    if length<=0 or n<stencil or stencil<3 or stencil%2!=1 or order>=stencil:
        raise ValueError("invalid periodic derivative parameters")
    h=length/n; half=stencil//2
    offsets=np.arange(-half,half+1,dtype=int)
    weights=_fd_weights(0.0,offsets.astype(float)*h,order)
    D=np.zeros((n,n))
    for i in range(n): D[i,(i+offsets)%n]=weights
    return D

@dataclass(frozen=True)
class CylinderSpatialOperators:
    grid:mvi.CylinderGrid
    D1x:np.ndarray
    D1s:np.ndarray
    D2x:np.ndarray
    D2s:np.ndarray
    @classmethod
    def build(cls,grid,stencil=7):
        if grid.nx<stencil or grid.ns<stencil: raise ValueError("nx/ns < stencil")
        return cls(grid,
                   derivative_matrix_nonperiodic(grid.x,1,stencil),
                   derivative_matrix_periodic_uniform(grid.circumference,grid.ns,1,stencil),
                   derivative_matrix_nonperiodic(grid.x,2,stencil),
                   derivative_matrix_periodic_uniform(grid.circumference,grid.ns,2,stencil))
    def Dx(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError("Dx shape")
        return A@self.D1x.T
    def Ds(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError("Ds shape")
        return self.D1s@A
    def Dxx(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError("Dxx shape")
        return A@self.D2x.T
    def Dss(self,A):
        A=np.asarray(A,float)
        if A.shape!=(self.grid.ns,self.grid.nx): raise ValueError("Dss shape")
        return self.D2s@A

def profile_fields_and_first_derivatives(state,ops):
    state.validate_grid(ops.grid)
    X1=state.X@ops.D1x.T; Y1=state.Y@ops.D1s.T
    q=np.empty((6,ops.grid.ns,ops.grid.nx)); qx=np.empty_like(q); qs=np.empty_like(q)
    for f in range(6):
        q[f]=state.Y[f].T@state.X[f]
        qx[f]=state.Y[f].T@X1[f]
        qs[f]=Y1[f].T@state.X[f]
    return q,qx,qs

def raw_strong_residual_from_profiles(state,ops,*,K0,GB,quadrature,generalized_load=None):
    q,qx,qs=profile_fields_and_first_derivatives(state,ops)
    cache=s2.thickness_cache_te1(q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,quadrature=quadrature)
    if generalized_load is None:
        generalized_load=s2.zero_generalized_load((ops.grid.ns,ops.grid.nx))
    residual=s2.te1_residuals(cache,Dx=ops.Dx,Ds=ops.Ds,R=ops.grid.radius,
                              generalized_load=generalized_load)
    return residual,cache,(q,qx,qs)

def flatten_residual(residual,grid):
    residual=np.asarray(residual,float)
    if residual.shape!=(6,grid.ns,grid.nx): raise ValueError("residual shape")
    return residual.reshape(-1)
