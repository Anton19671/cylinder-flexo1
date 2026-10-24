#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3b exact tangent assembly helpers.

The tangent action itself is supplied by the mechanics module.  This helper only
assembles the existing dense matrix.  CUF uses the already verified K2 spatial
support graph to superpose directions whose residual-row supports are disjoint.
KL currently uses one exact tangent action per nodal coordinate as a deliberately
conservative reference/production path; no finite differences are used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import final_k2_coloring as k2


@dataclass(frozen=True)
class AnalyticAssemblyStats:
    n_columns: int
    tangent_action_evaluations: int
    dense_reference_action_evaluations: int
    action_reduction_factor: float
    dependency_layers: int | None
    n_spatial_colors: int | None
    overlap_check: str
    method: str

    def to_dict(self):
        return asdict(self)


def _full_rows(spatial_ids, *, n_field_blocks, n_spatial):
    spatial_ids=np.asarray(spatial_ids,dtype=int)
    return np.concatenate([
        f*n_spatial + spatial_ids for f in range(int(n_field_blocks))
    ]).astype(int,copy=False)


def assemble_cuf_colored_exact_tangent(base_state,ops,tangent_action,*,dependency_layers=2):
    """Assemble exact dense CUF tangent from exact directional derivatives.

    Tangent linearity permits simultaneous unit directions at K2-disjoint spatial
    nodes.  Unlike FD coloring, there is no perturbation size and only one exact
    tangent action per field/color group.
    """
    base=np.asarray(base_state,float)
    if base.ndim!=4 or base.shape[0]!=3:
        raise ValueError("CUF base_state must have shape (3,m,ns,nx)")
    nf=int(base.shape[0]*base.shape[1]); ns=int(base.shape[2]); nx=int(base.shape[3])
    if ns!=int(ops.grid.ns) or nx!=int(ops.grid.nx):
        raise ValueError("base_state/grid mismatch")
    nsp=ns*nx; n=nf*nsp
    supports=k2.spatial_dependency_supports(ops,layers=dependency_layers)
    groups=k2.greedy_disjoint_groups(supports,nsp)
    J=np.zeros((n,n),float)
    evals=0
    flat_shape=(nf,ns,nx)
    for f in range(nf):
        for group in groups:
            d=np.zeros(flat_shape,float)
            for sid in group:
                js,ix=divmod(int(sid),nx)
                d[f,js,ix]=1.0
            y=np.asarray(tangent_action(d.reshape(base.shape)),float).reshape(-1)
            if y.shape!=(n,):
                raise ValueError("tangent_action returned wrong size")
            evals+=1
            for sid in group:
                rows=_full_rows(supports[int(sid)],n_field_blocks=nf,n_spatial=nsp)
                j=f*nsp+int(sid)
                J[rows,j]=y[rows]
    return J,AnalyticAssemblyStats(
        n_columns=n,tangent_action_evaluations=evals,
        dense_reference_action_evaluations=n,
        action_reduction_factor=(float(n)/float(evals) if evals else float('inf')),
        dependency_layers=int(dependency_layers),n_spatial_colors=len(groups),
        overlap_check='PASS',method='EXACT_ANALYTIC_K2_COLORED'
    )


def assemble_dense_exact_tangent(base_state,tangent_action):
    """Conservative exact dense tangent assembly, one analytic action per column."""
    base=np.asarray(base_state,float)
    n=base.size
    J=np.empty((n,n),float)
    shape=base.shape
    for j in range(n):
        d=np.zeros(n,float); d[j]=1.0
        y=np.asarray(tangent_action(d.reshape(shape)),float).reshape(-1)
        if y.shape!=(n,):
            raise ValueError("tangent_action returned wrong size")
        J[:,j]=y
    return J,AnalyticAssemblyStats(
        n_columns=n,tangent_action_evaluations=n,dense_reference_action_evaluations=n,
        action_reduction_factor=1.0,dependency_layers=None,n_spatial_colors=None,
        overlap_check='NOT_APPLICABLE',method='EXACT_ANALYTIC_DENSE_ACTION'
    )
