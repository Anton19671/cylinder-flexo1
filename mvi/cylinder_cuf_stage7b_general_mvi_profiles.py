#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7B: GENERAL CUF separated-profile layer for MVI.

This stage generalizes only the accepted separated representation and its
exact algebraic profile tangent from 6 TE1 fields to 3m general CUF fields.

State:
    X[component,tau,rank,x_node]
    Y[component,tau,rank,s_node]
component order: u,v,w.

For m=2, the flattened field order is exactly:
    [u0,u1,v0,v1,w0,w1],
which is the legacy verified TE1 order.

No new MVI algorithm is introduced:
- partner remains frozen during a halfstep;
- this module only provides exact reconstruction, active parameters and T=dU/dc;
- no solver, normalization, balancing, gauge or fallback is added here.

Stage 7B also connects general CUF profiles to Stage 7A mechanics:
    X/Y -> q,q_x,q_s -> general 3m raw strong residual.

End boundary conditions and the full general-CUF MVI solve are NOT yet
implemented in this stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import cylinder_te1_mvi_core as te1mvi
import cylinder_cuf_stage7a_general_mechanics as g7


class CUFProfileError(RuntimeError):
    pass


@dataclass
class ProfileStateCUF:
    # X: (3,m,rank,nx), Y: (3,m,rank,ns)
    X: np.ndarray
    Y: np.ndarray

    def __post_init__(self):
        self.X=np.asarray(self.X,float)
        self.Y=np.asarray(self.Y,float)
        if self.X.ndim != 4 or self.Y.ndim != 4:
            raise ValueError("X/Y must be 4D: (3,m,rank,n_dir)")
        if self.X.shape[0] != 3 or self.Y.shape[0] != 3:
            raise ValueError("first dimension must be 3: [u,v,w]")
        if self.X.shape[1] != self.Y.shape[1] or self.X.shape[1] < 1:
            raise ValueError("CUF basis count m mismatch/invalid")
        if self.X.shape[2] != self.Y.shape[2] or self.X.shape[2] < 1:
            raise ValueError("MVI rank mismatch/invalid")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.Y)):
            raise ValueError("non-finite profiles")

    @property
    def m(self):
        return int(self.X.shape[1])

    @property
    def rank(self):
        return int(self.X.shape[2])

    @property
    def n_fields(self):
        return 3*self.m

    def copy(self):
        return ProfileStateCUF(self.X.copy(),self.Y.copy())

    def validate_grid(self,grid):
        if self.X.shape[3] != grid.nx or self.Y.shape[3] != grid.ns:
            raise ValueError("profile/grid mismatch")


def reconstruct_fields(state):
    """
    Output shape (3,m,ns,nx).
    """
    ns=state.Y.shape[3]
    nx=state.X.shape[3]
    U=np.empty((3,state.m,ns,nx),dtype=float)
    for a in range(3):
        for tau in range(state.m):
            U[a,tau]=state.Y[a,tau].T@state.X[a,tau]
    return U


def reconstruct_vector(state,grid):
    state.validate_grid(grid)
    return reconstruct_fields(state).reshape(-1)


def n_full_dofs(state,grid):
    state.validate_grid(grid)
    return state.n_fields*grid.n_surface_nodes


def residual_projection_weights(state,grid):
    """
    Same midpoint/trapezoidal surface weights as verified TE1 Stage 1,
    tiled over 3m fields in component,tau order.
    """
    state.validate_grid(grid)
    wx=np.full(grid.nx,grid.dx,dtype=float)
    wx[0]*=.5; wx[-1]*=.5
    ws=np.full(grid.ns,grid.ds,dtype=float)
    w2=np.outer(ws,wx)
    return np.tile(w2.ravel(),state.n_fields)


def _array_for_direction(state,direction):
    d=str(direction).upper()
    if d=="X":
        return state.X
    if d=="Y":
        return state.Y
    raise ValueError("direction must be X or Y")


def _active_ids(state,direction,active_mask=None):
    arr=_array_for_direction(state,direction)
    if active_mask is None:
        mask=np.ones_like(arr,dtype=bool)
    else:
        mask=np.asarray(active_mask,dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("active_mask shape mismatch")
    ids=np.flatnonzero(mask.ravel())
    if ids.size == 0:
        raise CUFProfileError(
            f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )
    return ids


def get_active_parameters(state,direction,active_mask=None):
    arr=_array_for_direction(state,direction)
    ids=_active_ids(state,direction,active_mask)
    return arr.ravel()[ids].copy(),ids


def set_active_parameters(state,direction,ids,values):
    arr=_array_for_direction(state,direction)
    ids=np.asarray(ids,int)
    vals=np.asarray(values,float)
    if vals.shape != ids.shape:
        raise ValueError("values/ids shape mismatch")
    # Write through to the original profile array.  ravel() can allocate a
    # copy for a non-C-contiguous view, which would silently discard the
    # half-step update.
    arr.flat[ids]=vals


def _decode(flat_idx,m,rank,n_dir):
    """
    Ravel order of (3,m,rank,n_dir):
       component -> tau -> rank -> node.
    """
    block,node=divmod(int(flat_idx),int(n_dir))
    block,r=divmod(block,int(rank))
    component,tau=divmod(block,int(m))
    return component,tau,r,node


def build_direction_tangent(state,grid,direction,active_mask=None):
    """
    Exact algebraic T=dU/dc for the general 3m-field separated representation.
    """
    state.validate_grid(grid)
    d=str(direction).upper()
    ids=_active_ids(state,d,active_mask)

    n2=grid.n_surface_nodes
    T=np.zeros((state.n_fields*n2,len(ids)),dtype=float)
    n_dir=grid.nx if d=="X" else grid.ns

    for col,idx in enumerate(ids):
        a,tau,r,node=_decode(idx,state.m,state.rank,n_dir)
        field=a*state.m+tau
        base=field*n2

        if d=="Y":
            row0=base+node*grid.nx
            T[row0:row0+grid.nx,col]=state.X[a,tau,r]
        else:
            rows=base+np.arange(grid.ns)*grid.nx+node
            T[rows,col]=state.Y[a,tau,r]
    return T,ids


def profile_fields_and_first_derivatives(state,ops):
    """
    Accepted spatial derivative rule, generalized to 3m fields:
        X_x = X @ D1x.T
        Y_s = Y @ D1s.T
        q   = sum_r X_r Y_r
        q_x = sum_r X'_r Y_r
        q_s = sum_r X_r Y'_r

    Outputs shape (3,m,ns,nx).
    """
    state.validate_grid(ops.grid)
    X1=state.X@ops.D1x.T
    Y1=state.Y@ops.D1s.T

    q=np.empty((3,state.m,ops.grid.ns,ops.grid.nx),dtype=float)
    qx=np.empty_like(q)
    qs=np.empty_like(q)

    for a in range(3):
        for tau in range(state.m):
            q[a,tau]=state.Y[a,tau].T@state.X[a,tau]
            qx[a,tau]=state.Y[a,tau].T@X1[a,tau]
            qs[a,tau]=Y1[a,tau].T@state.X[a,tau]
    return q,qx,qs


def raw_strong_residual_from_profiles(
    state,ops,*,K0,GB,quadrature,basis,generalized_load=None
):
    """
    General profile -> Stage 7A mechanics connector.
    Output residual shape (3,m,ns,nx).
    """
    if basis.m != state.m:
        raise ValueError("basis.m must equal state.m")

    q,qx,qs=profile_fields_and_first_derivatives(state,ops)
    resultants=g7.general_thickness_resultants(
        q,qx,qs,
        R=ops.grid.radius,
        K0=K0,
        GB=GB,
        quadrature=quadrature,
        basis=basis,
    )
    if generalized_load is None:
        generalized_load=g7.zero_generalized_load(
            state.m,(ops.grid.ns,ops.grid.nx)
        )
    residual=g7.general_residuals(
        resultants,
        Dx=ops.Dx,
        Ds=ops.Ds,
        generalized_load=generalized_load,
    )
    return residual,resultants,(q,qx,qs)


def te1_state_to_general(state_te1):
    """
    Exact storage reinterpretation:
      legacy (6,rank,n_dir) -> general (3,2,rank,n_dir).
    """
    return ProfileStateCUF(
        np.asarray(state_te1.X,float).reshape(
            3,2,state_te1.rank,state_te1.X.shape[2]
        ).copy(),
        np.asarray(state_te1.Y,float).reshape(
            3,2,state_te1.rank,state_te1.Y.shape[2]
        ).copy(),
    )


def general_te1_state_to_legacy(state):
    if state.m != 2:
        raise ValueError("general state must have m=2")
    return te1mvi.ProfileStateTE1(
        state.X.reshape(6,state.rank,state.X.shape[3]).copy(),
        state.Y.reshape(6,state.rank,state.Y.shape[3]).copy(),
    )
