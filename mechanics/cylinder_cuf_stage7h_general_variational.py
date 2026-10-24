#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental weak/variational CUF residual for MVI verification.

This module does not replace the production strong-form CUF equations.  It
uses the same CUF kinematics, constitutive law, thickness basis, resultants
and exact material/geometric tangent, but assembles the discrete first
variation before integration by parts:

  deltaPi = int [A_x delta(q_,x) + A_s delta(q_,s) + C delta(q) - p delta(q)] dx ds.

The initial verification scope is homogeneous geometric conditions at both
x-ends (the clamped-clamped Chinese comparison case).  This avoids introducing
an unverified discrete natural-boundary treatment while still giving an
independent strong-vs-weak discretisation check.
"""
from __future__ import annotations

import numpy as np

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as g8


class GeneralCUFVariationalError(RuntimeError):
    pass


def _require_geometric_ends(bc):
    if not (bc.left.all_geometric() and bc.right.all_geometric()):
        raise GeneralCUFVariationalError(
            "VARIATIONAL_MVI_INITIAL_SCOPE_REQUIRES_ALL_GEOMETRIC_ENDS"
        )


def surface_weights(grid):
    """Trapezoidal x and periodic midpoint s weights, identical to MVI surface weights."""
    wx=np.full(int(grid.nx),float(grid.dx),dtype=float)
    wx[0]*=0.5
    wx[-1]*=0.5
    ws=np.full(int(grid.ns),float(grid.ds),dtype=float)
    return np.outer(ws,wx)


def _surface_first_derivatives_full_direction(dq,ops):
    """Apply the accepted CUF first-derivative matrices to a full field direction."""
    a=np.asarray(dq,float)
    if a.ndim!=4 or a.shape[0]!=3:
        raise ValueError("dq must have shape (3,m,ns,nx)")
    if a.shape[2:]!=(ops.grid.ns,ops.grid.nx):
        raise ValueError("dq/grid shape mismatch")
    dqx=a@np.asarray(ops.D1x,float).T
    dqs=np.einsum(
        "ij,abjx->abix",np.asarray(ops.D1s,float),a,optimize=False
    )
    return dqx,dqs


def weak_residual_from_resultants(resultants,*,ops,generalized_load,bc):
    """Assemble the nodal gradient of the discrete CUF total potential."""
    _require_geometric_ends(bc)
    p=np.asarray(generalized_load,float)
    expected=(3,resultants.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    w2=surface_weights(ops.grid)
    out=np.empty_like(p)
    triples=(
        (resultants.Au_x,resultants.Au_s,resultants.Cu),
        (resultants.Av_x,resultants.Av_s,resultants.Cv),
        (resultants.Aw_x,resultants.Aw_s,resultants.Cw),
    )
    D1x=np.asarray(ops.D1x,float)
    D1s=np.asarray(ops.D1s,float)

    for comp,(Ax,As,C) in enumerate(triples):
        for tau in range(resultants.m):
            wax=w2*np.asarray(Ax[tau],float)
            was=w2*np.asarray(As[tau],float)
            local=w2*(np.asarray(C[tau],float)-p[comp,tau])
            # q_x = D1x q and q_s = D1s q, therefore the discrete first
            # variation contains D1x.T W A_x and D1s.T W A_s.
            out[comp,tau]=wax@D1x + D1s.T@was + local

    # Homogeneous geometric values are not variational unknowns.  Zero their
    # rows so the reported full weak residual is the admissible-space residual.
    out[:,:,:,0]=0.0
    out[:,:,:,-1]=0.0
    return out


def weak_tangent_from_resultants(dresultants,dq,*,ops,bc):
    """Exact action of the weak-form tangent on one full-field direction."""
    _require_geometric_ends(bc)
    dq=np.asarray(dq,float)
    zload=np.zeros_like(dq)
    out=weak_residual_from_resultants(
        dresultants,ops=ops,generalized_load=zload,bc=bc
    )
    # Geometric rows were already zeroed by weak_residual_from_resultants.
    return out


def generalized_load_vector(*,ops,generalized_load,bc):
    """Discrete external-force vector consistent with the weak residual."""
    _require_geometric_ends(bc)
    p=np.asarray(generalized_load,float)
    w2=surface_weights(ops.grid)
    out=w2[None,None,:,:]*p
    out=out.copy()
    out[:,:,:,0]=0.0
    out[:,:,:,-1]=0.0
    return out


def make_variational_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    """Return residual(state) plus the exact tangent-action hook used by MVI."""
    _require_geometric_ends(bc)
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if basis.m!=bc.m:
        raise ValueError("basis/BC m mismatch")

    def residual(state):
        q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
        resultants=g7.general_thickness_resultants(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        return weak_residual_from_resultants(
            resultants,ops=ops,generalized_load=p,bc=bc
        ).reshape(-1)

    def prepare_exact_tangent_action(state):
        q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
        ctx=g7.prepare_general_elastic_tangent_context(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        shape=q.shape

        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(shape)
            dqx,dqs=_surface_first_derivatives_full_direction(dq,ops)
            dres=g7.general_thickness_resultants_tangent_from_context(
                ctx,dq,dqx,dqs
            )
            return weak_tangent_from_resultants(
                dres,dq,ops=ops,bc=bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    # Weak assembly contains adjoint derivative operators (D^T).  The current
    # profile-color support proof is for the strong residual and is not valid
    # at one-sided axial stencils for the adjoint graph.  Evaluate each exact
    # variational tangent direction separately until adjoint-aware coloring is
    # implemented; correctness takes precedence over the speed shortcut.
    residual.mvi_exact_tangent_profile_coloring=False
    residual.mvi_tangent_mode="EXACT_VARIATIONAL_GREEN_LAGRANGE_FROZEN_GB"
    residual.mvi_projected_reference_residual=generalized_load_vector(
        ops=ops,generalized_load=p,bc=bc
    ).reshape(-1)
    residual.mvi_formulation="VARIATIONAL"
    residual.mvi_residual_already_integrated=True
    return residual
