#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2: safe colored assembly for full-grid CUF finite differences.

The CUF residual does not have the one-stencil dependency assumed by the
original audit sketch.  The nodal fields are differentiated once to build
resultants and the resultants are differentiated again in the strong residual.
Therefore the spatial dependency graph is the *two-step* closure of the actual
D1x/D1s matrices.

This module constructs that graph directly from the verified derivative
matrices, including the periodic wrap in s and one-sided x stencils.  Columns
are grouped only when their conservative residual-row supports are disjoint.
No SciPy/sparse solver is used; the returned Jacobian/operator is the same
dense ndarray consumed by the existing np.linalg.solve path.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class K2AssemblyStats:
    n_columns: int
    n_spatial_nodes: int
    n_field_blocks: int
    dependency_layers: int
    n_spatial_colors: int
    perturbed_residual_evaluations: int
    base_residual_evaluations_inside: int
    total_residual_evaluations_inside: int
    dense_reference_perturbed_evaluations: int
    evaluation_reduction_factor: float
    min_group_size: int
    max_group_size: int
    periodic_s: bool
    overlap_check: str

    def to_dict(self):
        return asdict(self)


def _axis_output_rows_by_input(D):
    """For each input index j, return derivative-output rows i with D[i,j]!=0."""
    D=np.asarray(D,float)
    if D.ndim!=2 or D.shape[0]!=D.shape[1]:
        raise ValueError("derivative matrix must be square")
    # The verified matrices are explicitly sparse-in-dense storage.  Keep even
    # tiny stencil coefficients: support detection must be conservative.
    return tuple(np.flatnonzero(D[:,j] != 0.0).astype(int) for j in range(D.shape[1]))


def spatial_dependency_supports(ops, *, layers=2):
    """Return conservative output-spatial supports for every input node.

    One graph step represents either identity, one Dx dependency, or one Ds
    dependency.  CUF uses two layers: q -> (q,qx,qs)->resultants -> residual.
    Periodicity is inherited exactly from ops.D1s rather than reconstructed by
    modular arithmetic.
    """
    layers=int(layers)
    if layers < 0:
        raise ValueError("layers must be >=0")
    ns=int(ops.grid.ns); nx=int(ops.grid.nx)
    x_rows=_axis_output_rows_by_input(ops.D1x)
    s_rows=_axis_output_rows_by_input(ops.D1s)

    supports=[]
    for is0 in range(ns):
        for ix0 in range(nx):
            reached={(is0,ix0)}
            frontier={(is0,ix0)}
            for _ in range(layers):
                nxt=set()
                for js,jx in frontier:
                    nxt.add((js,jx))
                    for io in x_rows[jx]:
                        nxt.add((js,int(io)))
                    for so in s_rows[js]:
                        nxt.add((int(so),jx))
                nxt-=reached
                reached|=nxt
                frontier=nxt
                if not frontier:
                    break
            supports.append(np.asarray(sorted(s*nx+x for s,x in reached),dtype=int))
    return tuple(supports)


def greedy_disjoint_groups(spatial_supports, n_spatial):
    """Greedy coloring where supports inside each group are pairwise disjoint."""
    n_spatial=int(n_spatial)
    groups=[]
    occupied=[]
    # Largest supports first gives a stable conservative coloring near x ends.
    order=sorted(range(n_spatial), key=lambda j:(-len(spatial_supports[j]),j))
    for node in order:
        supp=spatial_supports[node]
        placed=False
        for g,occ in zip(groups,occupied):
            if not np.any(occ[supp]):
                g.append(node)
                occ[supp]=True
                placed=True
                break
        if not placed:
            occ=np.zeros(n_spatial,dtype=bool)
            occ[supp]=True
            groups.append([node])
            occupied.append(occ)

    # Hard safety check: this is the central K2 invariant.
    for g in groups:
        seen=np.zeros(n_spatial,dtype=bool)
        for node in g:
            supp=spatial_supports[node]
            if np.any(seen[supp]):
                raise RuntimeError("K2_INTERNAL_OVERLAP: colored supports intersect")
            seen[supp]=True
    return tuple(tuple(g) for g in groups)


def _full_rows(spatial_ids, *, n_field_blocks, n_spatial):
    spatial_ids=np.asarray(spatial_ids,dtype=int)
    return np.concatenate([
        f*n_spatial + spatial_ids for f in range(int(n_field_blocks))
    ]).astype(int,copy=False)


def assemble_cuf_colored_columns(
    base_state, ops, residual_evaluator, *, base_residual=None, step=1.0,
    dependency_layers=2,
):
    """Assemble a dense CUF Jacobian/operator with safe simultaneous columns.

    Parameters
    ----------
    base_state : ndarray, shape (3,m,ns,nx)
    residual_evaluator : callable(state)->1D residual of identical flattened size
    base_residual : optional already evaluated R(base_state)
    step : finite-difference step; use 1.0 for affine unit-column extraction
    dependency_layers : 2 for the current CUF strong residual
    """
    base=np.asarray(base_state,float)
    if base.ndim!=4 or base.shape[0]!=3:
        raise ValueError("CUF base_state must have shape (3,m,ns,nx)")
    nf=int(base.shape[0]*base.shape[1])
    ns=int(base.shape[2]); nx=int(base.shape[3])
    if ns!=int(ops.grid.ns) or nx!=int(ops.grid.nx):
        raise ValueError("base_state/grid mismatch")
    nsp=ns*nx
    n=nf*nsp
    step=float(step)
    if not np.isfinite(step) or step<=0.0:
        raise ValueError("step must be finite and >0")

    base_eval_count=0
    if base_residual is None:
        r0=np.asarray(residual_evaluator(base),float).reshape(-1)
        base_eval_count=1
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
    if r0.shape!=(n,):
        raise ValueError("base_residual size mismatch")

    supports=spatial_dependency_supports(ops,layers=dependency_layers)
    groups=greedy_disjoint_groups(supports,nsp)
    J=np.zeros((n,n),float)
    flat_shape=(nf,ns,nx)

    evals=0
    for f in range(nf):
        for group in groups:
            trial=base.copy().reshape(flat_shape)
            for sid in group:
                js,ix=divmod(int(sid),nx)
                trial[f,js,ix]+=step
            rp=np.asarray(residual_evaluator(trial.reshape(base.shape)),float).reshape(-1)
            if rp.shape!=(n,):
                raise ValueError("residual_evaluator returned wrong size")
            delta=(rp-r0)/step
            evals+=1
            for sid in group:
                rows=_full_rows(
                    supports[int(sid)],n_field_blocks=nf,n_spatial=nsp
                )
                j=f*nsp+int(sid)
                J[rows,j]=delta[rows]

    sizes=[len(g) for g in groups]
    stats=K2AssemblyStats(
        n_columns=n,
        n_spatial_nodes=nsp,
        n_field_blocks=nf,
        dependency_layers=int(dependency_layers),
        n_spatial_colors=len(groups),
        perturbed_residual_evaluations=evals,
        base_residual_evaluations_inside=base_eval_count,
        total_residual_evaluations_inside=evals+base_eval_count,
        dense_reference_perturbed_evaluations=n,
        evaluation_reduction_factor=(float(n)/float(evals) if evals else float("inf")),
        min_group_size=min(sizes) if sizes else 0,
        max_group_size=max(sizes) if sizes else 0,
        periodic_s=True,
        overlap_check="PASS",
    )
    return J,r0,stats


def assemble_cuf_colored_central_columns(
    base_state, ops, residual_evaluator, *, characteristic_scale, relative_step,
    base_residual=None, dependency_layers=2,
):
    """V3+K2 dense Jacobian: safe coloring with per-column central steps.

    ``relative_step`` is dimensionless.  Each column uses
    h_j=relative_step*max(abs(x_j), characteristic_scale_j).
    Disjoint K2 supports allow different h_j values inside one color group:
    each extracted column is divided by its own 2*h_j.
    """
    import final_v3_fd as v3fd

    base=np.asarray(base_state,float)
    if base.ndim!=4 or base.shape[0]!=3:
        raise ValueError("CUF base_state must have shape (3,m,ns,nx)")
    nf=int(base.shape[0]*base.shape[1])
    ns=int(base.shape[2]); nx=int(base.shape[3])
    if ns!=int(ops.grid.ns) or nx!=int(ops.grid.nx):
        raise ValueError("base_state/grid mismatch")
    nsp=ns*nx; n=nf*nsp

    c=np.asarray(characteristic_scale,float)
    if c.shape!=base.shape:
        raise ValueError("characteristic_scale shape mismatch")
    steps=v3fd.relative_steps(base,c,relative_step).reshape(nf,ns,nx)

    base_eval_count=0
    if base_residual is None:
        r0=np.asarray(residual_evaluator(base),float).reshape(-1)
        base_eval_count=1
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
    if r0.shape!=(n,):
        raise ValueError("base_residual size mismatch")

    supports=spatial_dependency_supports(ops,layers=dependency_layers)
    groups=greedy_disjoint_groups(supports,nsp)
    J=np.zeros((n,n),float)
    flat_shape=(nf,ns,nx)
    evals=0

    for f in range(nf):
        for group in groups:
            plus=base.copy().reshape(flat_shape)
            minus=base.copy().reshape(flat_shape)
            for sid in group:
                js,ix=divmod(int(sid),nx)
                hj=steps[f,js,ix]
                plus[f,js,ix]+=hj
                minus[f,js,ix]-=hj
            rp=np.asarray(residual_evaluator(plus.reshape(base.shape)),float).reshape(-1)
            rm=np.asarray(residual_evaluator(minus.reshape(base.shape)),float).reshape(-1)
            if rp.shape!=(n,) or rm.shape!=(n,):
                raise ValueError("residual_evaluator returned wrong size")
            delta=rp-rm
            evals+=2
            for sid in group:
                js,ix=divmod(int(sid),nx)
                rows=_full_rows(supports[int(sid)],n_field_blocks=nf,n_spatial=nsp)
                j=f*nsp+int(sid)
                J[rows,j]=delta[rows]/(2.0*steps[f,js,ix])

    sizes=[len(g) for g in groups]
    stats=K2AssemblyStats(
        n_columns=n,
        n_spatial_nodes=nsp,
        n_field_blocks=nf,
        dependency_layers=int(dependency_layers),
        n_spatial_colors=len(groups),
        perturbed_residual_evaluations=evals,
        base_residual_evaluations_inside=base_eval_count,
        total_residual_evaluations_inside=evals+base_eval_count,
        dense_reference_perturbed_evaluations=2*n,
        evaluation_reduction_factor=(float(2*n)/float(evals) if evals else float("inf")),
        min_group_size=min(sizes) if sizes else 0,
        max_group_size=max(sizes) if sizes else 0,
        periodic_s=True,
        overlap_check="PASS",
    )
    return J,r0,stats
