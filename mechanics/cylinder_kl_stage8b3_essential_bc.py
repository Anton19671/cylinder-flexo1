#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8B-3: exact homogeneous essential end-BC parameterization
for KL separated X profiles.

For homogeneous geometric conditions at x-ends:
  u=0      -> X_u,r(end)=0
  v=0      -> X_v,r(end)=0
  w=0      -> X_w,r(end)=0
  theta_n=0 -> n_x * dX_w,r/dx(end)=0

Each retained separated term satisfies the homogeneous essential condition.
Natural conditions do NOT constrain X profiles here.

The constraint C x = 0 is parameterized exactly:
    x = B c
where B is formed by direct elimination of a small pivot block using
np.linalg.solve only.

No KKT.
No QR/SVD/pinv/lstsq.
No regularization.
No MVI solve yet.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np



class KLNonzeroGeometricBCNotDecomposed(RuntimeError):
    pass

class KLEssentialConstraintError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentConstraintMap:
    component: int
    C: np.ndarray
    B: np.ndarray
    pivot_columns: tuple
    free_columns: tuple

    @property
    def n_full(self):
        return int(self.B.shape[0])

    @property
    def n_free(self):
        return int(self.B.shape[1])

    def expand(self,c):
        c=np.asarray(c,float)
        if c.shape!=(self.n_free,):
            raise ValueError("reduced vector shape mismatch")
        return self.B@c

    def reduce_from_full_if_exact(self,x,tol=1e-10):
        x=np.asarray(x,float)
        if x.shape!=(self.n_full,):
            raise ValueError("full vector shape mismatch")
        if self.C.size and np.max(np.abs(self.C@x))>tol:
            raise KLEssentialConstraintError("full X profile violates essential BC")
        return x[np.asarray(self.free_columns,int)].copy()


@dataclass(frozen=True)
class KLXConstraintMaps:
    maps: tuple  # 3 ComponentConstraintMap

    def __post_init__(self):
        if len(self.maps)!=3:
            raise ValueError("need three component maps")

    @property
    def reduced_per_rank(self):
        return tuple(m.n_free for m in self.maps)


def _zero_geometric_data_or_reject(cond,ns):
    if cond.kind!="GEOMETRIC":
        return
    a=cond.data.array(ns)
    if np.max(np.abs(a))!=0.0:
        raise KLNonzeroGeometricBCNotDecomposed(
            "NONZERO_GEOMETRIC_BC_NOT_DECOMPOSED"
        )


def _constraint_rows_for_component(component,ops,bc):
    grid=ops.grid
    ns=grid.ns
    rows=[]

    for end,ix,n in ((bc.left,0,-1.0),(bc.right,-1,+1.0)):
        if component==0:
            cond=end.u
            _zero_geometric_data_or_reject(cond,ns)
            if cond.kind=="GEOMETRIC":
                r=np.zeros(grid.nx); r[ix]=1.0
                rows.append(r)

        elif component==1:
            cond=end.v
            _zero_geometric_data_or_reject(cond,ns)
            if cond.kind=="GEOMETRIC":
                r=np.zeros(grid.nx); r[ix]=1.0
                rows.append(r)

        elif component==2:
            _zero_geometric_data_or_reject(end.w,ns)
            _zero_geometric_data_or_reject(end.slope,ns)
            if end.w.kind=="GEOMETRIC":
                r=np.zeros(grid.nx); r[ix]=1.0
                rows.append(r)
            if end.slope.kind=="GEOMETRIC":
                # theta_n = n*dX/dx; zero constraint, keep n for exact definition.
                rows.append(n*np.asarray(ops.D1x[ix],float))
        else:
            raise ValueError("component must be 0,1,2")

    if not rows:
        return np.zeros((0,grid.nx),float)
    return np.vstack(rows)


def _parameterize_constraint(C):
    C=np.asarray(C,float)
    if C.ndim!=2:
        raise ValueError("C must be 2D")
    k,n=C.shape
    if k==0:
        return np.eye(n),tuple(),tuple(range(n))
    if k>=n:
        raise KLEssentialConstraintError("too many essential constraints")

    # deterministic candidate order emphasizes boundary/near-boundary columns,
    # but acceptance is based only on exact direct solve of C[:,piv].
    order=[]
    for j in range(n):
        if j not in order:
            order.append(j)
        jj=n-1-j
        if jj not in order:
            order.append(jj)

    chosen=None
    for comb in itertools.combinations(order,k):
        Cp=C[:,comb]
        try:
            inv_action=np.linalg.solve(Cp,np.eye(k))
        except np.linalg.LinAlgError:
            continue
        err=np.max(np.abs(Cp@inv_action-np.eye(k)))
        if np.all(np.isfinite(inv_action)) and err<1e-10:
            chosen=tuple(comb)
            break
    if chosen is None:
        raise KLEssentialConstraintError("cannot parameterize essential constraints")

    piv=np.asarray(chosen,int)
    free=np.asarray([j for j in range(n) if j not in chosen],int)
    Cp=C[:,piv]
    Cf=C[:,free]

    B=np.zeros((n,len(free)),float)
    B[free,np.arange(len(free))]=1.0
    if len(free):
        B[piv,:]=-np.linalg.solve(Cp,Cf)

    if np.max(np.abs(C@B))>1e-9:
        raise KLEssentialConstraintError("constraint parameterization residual too large")
    return B,tuple(int(x) for x in piv),tuple(int(x) for x in free)


def build_x_constraint_maps(ops,bc):
    bc.check_rigid_mode_requirement()
    maps=[]
    for a in range(3):
        C=_constraint_rows_for_component(a,ops,bc)
        B,piv,free=_parameterize_constraint(C)
        maps.append(ComponentConstraintMap(a,C,B,piv,free))
    return KLXConstraintMaps(tuple(maps))


def constrain_state_X_from_reduced(reduced,template_state,maps):
    """
    reduced: list/tuple of arrays:
        reduced[a] shape (rank,n_free[a])
    Returns a copy with full constrained X and unchanged Y.
    """
    st=template_state.copy()
    if len(reduced)!=3:
        raise ValueError("reduced must have three component arrays")
    for a,m in enumerate(maps.maps):
        c=np.asarray(reduced[a],float)
        if c.shape!=(st.rank,m.n_free):
            raise ValueError(f"reduced[{a}] shape mismatch")
        st.X[a]=c@m.B.T
    return st


def reduce_constrained_state_X(state,maps,tol=1e-9):
    out=[]
    for a,m in enumerate(maps.maps):
        c=np.empty((state.rank,m.n_free),float)
        for rr in range(state.rank):
            c[rr]=m.reduce_from_full_if_exact(state.X[a,rr],tol=tol)
        out.append(c)
    return tuple(out)


def constrained_X_tangent(state,grid,maps):
    """
    Exact dU/dc for all reduced X parameters, ordered:
      component -> rank -> reduced-node.
    """
    state.validate_grid(grid)
    cols=[]
    labels=[]
    n2=grid.n_surface_nodes
    for a,m in enumerate(maps.maps):
        for rr in range(state.rank):
            y=state.Y[a,rr]
            for j in range(m.n_free):
                dx=m.B[:,j]
                col=np.zeros(3*n2,float)
                base=a*n2
                # outer(y, dx), row-major ns x nx
                col[base:base+n2]=np.outer(y,dx).ravel()
                cols.append(col)
                labels.append((a,rr,j))
    if not cols:
        raise KLEssentialConstraintError("no reduced X parameters")
    return np.column_stack(cols),tuple(labels)


def enforce_homogeneous_essential_X_in_place(state,maps):
    """
    Projection by exact free-coordinate extraction is ONLY valid if the state
    already satisfies C X=0. This routine therefore does not silently modify
    an incompatible state.
    """
    reduced=reduce_constrained_state_X(state,maps)
    rebuilt=constrain_state_X_from_reduced(reduced,state,maps)
    state.X[:]=rebuilt.X
    return state
