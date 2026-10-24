#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8B-2: separated X/Y profiles for the displacement-only KL cylinder.

For each displacement component a in [u,v,w]:
    q_a(x,s) = sum_{r=1..N} X[a,r](x) Y[a,r](s)

Storage:
    X shape (3,rank,nx)
    Y shape (3,rank,ns)

This stage supplies:
- exact reconstruction;
- exact algebraic T_X=dU/dX and T_Y=dU/dY;
- projection weights;
- first/second/mixed derivatives from the verified 1D FD operators;
- connector to the verified Stage8A KL mechanics.

No end BC are imposed here.
No MVI solve is performed here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_stage8a_elastic_mechanics as kl


class KLProfileError(RuntimeError):
    pass


@dataclass
class ProfileStateKL:
    X: np.ndarray
    Y: np.ndarray

    def __post_init__(self):
        self.X=np.asarray(self.X,float)
        self.Y=np.asarray(self.Y,float)
        if self.X.ndim!=3 or self.Y.ndim!=3:
            raise ValueError("X/Y must be 3D: (3,rank,n_dir)")
        if self.X.shape[0]!=3 or self.Y.shape[0]!=3:
            raise ValueError("first dimension must be 3: [u,v,w]")
        if self.X.shape[1]!=self.Y.shape[1] or self.X.shape[1]<1:
            raise ValueError("rank mismatch/invalid")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.Y)):
            raise ValueError("non-finite profiles")

    @property
    def rank(self):
        return int(self.X.shape[1])

    def copy(self):
        return ProfileStateKL(self.X.copy(),self.Y.copy())

    def validate_grid(self,grid):
        if self.X.shape[2]!=grid.nx or self.Y.shape[2]!=grid.ns:
            raise ValueError("profile/grid mismatch")


def reconstruct_fields(state):
    ns=state.Y.shape[2]
    nx=state.X.shape[2]
    U=np.empty((3,ns,nx),float)
    for a in range(3):
        U[a]=state.Y[a].T@state.X[a]
    return U


def reconstruct_vector(state,grid):
    state.validate_grid(grid)
    return reconstruct_fields(state).reshape(-1)


def residual_projection_weights(state,grid):
    state.validate_grid(grid)
    wx=np.full(grid.nx,grid.dx,float)
    wx[0]*=.5; wx[-1]*=.5
    ws=np.full(grid.ns,grid.ds,float)
    return np.tile(np.outer(ws,wx).ravel(),3)


def _arr(state,direction):
    d=str(direction).upper()
    if d=="X": return state.X
    if d=="Y": return state.Y
    raise ValueError("direction must be X or Y")


def _active_ids(state,direction,active_mask=None):
    a=_arr(state,direction)
    if active_mask is None:
        mask=np.ones_like(a,dtype=bool)
    else:
        mask=np.asarray(active_mask,bool)
        if mask.shape!=a.shape:
            raise ValueError("active_mask shape mismatch")
    ids=np.flatnonzero(mask.ravel())
    if ids.size==0:
        raise KLProfileError(f"{str(direction).upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")
    return ids


def get_active_parameters(state,direction,active_mask=None):
    a=_arr(state,direction)
    ids=_active_ids(state,direction,active_mask)
    return a.ravel()[ids].copy(),ids


def set_active_parameters(state,direction,ids,values):
    a=_arr(state,direction)
    ids=np.asarray(ids,int)
    values=np.asarray(values,float)
    if values.shape!=ids.shape:
        raise ValueError("values/ids shape mismatch")
    # ndarray.flat preserves write-through semantics for non-contiguous
    # profile views; ravel() does not guarantee that.
    a.flat[ids]=values


def _decode(flat_idx,rank,n_dir):
    block,node=divmod(int(flat_idx),int(n_dir))
    component,r=divmod(block,int(rank))
    return component,r,node


def build_direction_tangent(state,grid,direction,active_mask=None):
    state.validate_grid(grid)
    d=str(direction).upper()
    ids=_active_ids(state,d,active_mask)
    n2=grid.n_surface_nodes
    T=np.zeros((3*n2,len(ids)),float)
    n_dir=grid.nx if d=="X" else grid.ns

    for col,idx in enumerate(ids):
        a,r,node=_decode(idx,state.rank,n_dir)
        base=a*n2
        if d=="Y":
            row0=base+node*grid.nx
            T[row0:row0+grid.nx,col]=state.X[a,r]
        else:
            rows=base+np.arange(grid.ns)*grid.nx+node
            T[rows,col]=state.Y[a,r]
    return T,ids


def profile_fields_and_derivatives(state,ops):
    """
    Returns U,Ux,Us,Uxx,Uss,Uxs, all shape (3,ns,nx).
    Derivatives are constructed directly from separated 1D profiles using
    the same verified D1/D2 matrices as full-grid KL kinematics:
        X1=X D1x^T, X2=X D2x^T
        Y1=Y D1s^T, Y2=Y D2s^T
    """
    state.validate_grid(ops.grid)
    X1=state.X@ops.D1x.T
    X2=state.X@ops.D2x.T
    Y1=state.Y@ops.D1s.T
    Y2=state.Y@ops.D2s.T

    shp=(3,ops.grid.ns,ops.grid.nx)
    U=np.empty(shp,float); Ux=np.empty(shp,float); Us=np.empty(shp,float)
    Uxx=np.empty(shp,float); Uss=np.empty(shp,float); Uxs=np.empty(shp,float)

    for a in range(3):
        U[a]=state.Y[a].T@state.X[a]
        Ux[a]=state.Y[a].T@X1[a]
        Us[a]=Y1[a].T@state.X[a]
        Uxx[a]=state.Y[a].T@X2[a]
        Uss[a]=Y2[a].T@state.X[a]
        Uxs[a]=Y1[a].T@X1[a]
    return U,Ux,Us,Uxx,Uss,Uxs


def elastic_residual_from_profiles(
    state,ops,*,E,nu,h,geometry,generalized_load=None,
    resultants_method="ANALYTIC",quadrature=None
):
    U=reconstruct_fields(state)
    return kl.elastic_strong_residual(
        U,ops,E=E,nu=nu,h=h,geometry=geometry,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )
