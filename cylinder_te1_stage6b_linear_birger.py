#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 Stage 6B: LINEAR GEOMETRY + BIRGER material branch.

This closes the fourth mandatory TE1 combination:
  linear geometry + physically nonlinear Birger material.

Kinematics:
  E^L from the verified Stage 4A linear TE1 branch.

Material:
  S = K0 tr(E^L) I + 2 GB dev(E^L)
  with frozen GB during each inner MVI solve.

First-order stress measure:
  P^L = S.

Outer material cycle:
  exactly the verified Stage 5A Birger cycle, including the mandatory final
  inner solve after updating GB.

No new nonlinear solver, no new material rule, no benchmark data.
"""
from __future__ import annotations

import numpy as np

import cylinder_te1_local_kernel as nk
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage3b_boundary as b3
import cylinder_te1_stage3c_internal_mvi as c4
import cylinder_te1_stage4a_linear_elastic as l4
import cylinder_te1_stage5a_birger as b5


def linear_birger_local_te1(q, qx, qs, z, R, K0, GB):
    H, abc = nk.te1_gradient_quantities(q, qx, qs, z, R)
    E = l4.linear_strain_from_abc(abc)
    S = nk.birger_second_piola(E, float(K0), float(GB))
    P = S.copy()
    return {
        "H": H,
        "abc": abc,
        "E": E,
        "epsilon_i": nk.strain_intensity_from_E(E),
        "S": S,
        "sigma_i": nk.stress_intensity_from_S(S),
        "P": P,
    }


def _gb_at(GB, g, idx, nz, surf_shape):
    return b5._GB_value(GB, g, idx, nz, surf_shape)


def linear_birger_thickness_cache_te1(
    q, qx, qs, *, R, K0, GB, quadrature
):
    q=np.asarray(q,float); qx=np.asarray(qx,float); qs=np.asarray(qs,float)
    if q.shape != qx.shape or q.shape != qs.shape or q.ndim < 1 or q.shape[0] != 6:
        raise ValueError("q/qx/qs must have identical shape (6,...)")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qs))):
        raise ValueError("non-finite q/qx/qs")

    surf_shape=q.shape[1:]
    nz=quadrature.z.size
    acc={k:np.zeros(surf_shape,float) for k in s2.CACHE_KEYS}

    for idx in np.ndindex(surf_shape):
        q6=q[(slice(None),)+idx]
        qx6=qx[(slice(None),)+idx]
        qs6=qs[(slice(None),)+idx]
        for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
            GBg=_gb_at(GB,g,idx,nz,surf_shape)
            rr=linear_birger_local_te1(
                q6,qx6,qs6,z,R,K0,GBg
            )
            H=rr["H"]; P=rr["P"]
            Pxx,Pxs,Pxr=P[0,0],P[0,1],P[0,2]
            Psx,Pss,Psr=P[1,0],P[1,1],P[1,2]
            Prx,Prs,Prr=P[2,0],P[2,1],P[2,2]

            acc["Ixx0"][idx]+=wz*(H*Pxx)
            acc["Ixx1"][idx]+=wz*(H*z*Pxx)
            acc["Ixs0"][idx]+=wz*Pxs
            acc["Ixs1"][idx]+=wz*(z*Pxs)
            acc["Ixr1"][idx]+=wz*(H*Pxr)

            acc["Isx0"][idx]+=wz*(H*Psx)
            acc["Isx1"][idx]+=wz*(H*z*Psx)
            acc["Iss0"][idx]+=wz*Pss
            acc["Iss1"][idx]+=wz*(z*Pss)
            acc["Isr1"][idx]+=wz*(H*Psr)

            acc["Irs0"][idx]+=wz*Prs
            acc["Irs1"][idx]+=wz*(z*Prs)
            acc["Irx0"][idx]+=wz*(H*Prx)
            acc["Irx1"][idx]+=wz*(H*z*Prx)
            acc["Irr1"][idx]+=wz*(H*Prr)

    return s2.ThicknessCache(**acc)


def linear_birger_raw_residual_from_profiles(
    state, ops, *, K0, GB, quadrature, generalized_load
):
    q,qx,qs=c3.profile_fields_and_first_derivatives(state,ops)
    cache=linear_birger_thickness_cache_te1(
        q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,quadrature=quadrature
    )
    residual=s2.te1_residuals(
        cache,Dx=ops.Dx,Ds=ops.Ds,R=ops.grid.radius,
        generalized_load=generalized_load
    )
    return residual,cache,(q,qx,qs)


def make_linear_birger_residual_function(
    ops, *, K0, GB, quadrature, generalized_load, bc
):
    p=np.asarray(generalized_load,float)
    expected=(6,ops.grid.ns,ops.grid.nx)
    if p.shape != expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def residual(state):
        raw,cache,(q,_qx,_qs)=linear_birger_raw_residual_from_profiles(
            state,ops,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=p
        )
        return b3.assemble_strong_residual_with_end_bcs(
            raw,q,cache,bc
        ).reshape(-1)
    return residual


def evaluate_linear_birger_material_points(
    state, ops, *, K0, GB, quadrature
):
    q,qx,qs=c3.profile_fields_and_first_derivatives(state,ops)
    surf_shape=(ops.grid.ns,ops.grid.nx)
    nz=quadrature.z.size
    eps=np.empty((nz,)+surf_shape,float)
    sig=np.empty_like(eps)

    # Linear-geometry branch has no nonlinear deformation-gradient J diagnostic.
    # For compatibility with the common Birger outer-cycle result structure,
    # the first-order reference mapping is J=1 identically.
    det=np.ones_like(eps)

    for idx in np.ndindex(surf_shape):
        q6=q[(slice(None),)+idx]
        qx6=qx[(slice(None),)+idx]
        qs6=qs[(slice(None),)+idx]
        for g,z in enumerate(quadrature.z):
            GBg=_gb_at(GB,g,idx,nz,surf_shape)
            rr=linear_birger_local_te1(
                q6,qx6,qs6,z,ops.grid.radius,K0,GBg
            )
            eps[(g,)+idx]=rr["epsilon_i"]
            sig[(g,)+idx]=rr["sigma_i"]

    return b5.MaterialPointState(eps,sig,det)


def solve_linear_birger_load_step(
    initial_state,
    ops,
    *,
    E0,
    nu0,
    quadrature,
    generalized_load,
    bc,
    inner_settings,
    birger_settings,
    material_diagram,
    initial_GB=None,
    previous_accepted_epsilon_i=None,
):
    ec=nk.ElasticConstants(float(E0),float(nu0))
    gb_shape=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    if initial_GB is None:
        GB0=np.full(gb_shape,ec.G0,float)
    else:
        GB0=np.broadcast_to(
            np.asarray(initial_GB,float),gb_shape
        ).astype(float,copy=True)

    p=np.asarray(generalized_load,float)
    expected=(6,ops.grid.ns,ops.grid.nx)
    if p.shape != expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def inner(st,GBfield):
        fn=make_linear_birger_residual_function(
            ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,
            generalized_load=p,bc=bc
        )
        return c4.solve_internal_mvi(
            st,ops.grid,bc,fn,inner_settings,
            quadrature=quadrature
        )

    def evaluator(st,GBfield):
        return evaluate_linear_birger_material_points(
            st,ops,K0=ec.K0,GB=GBfield,quadrature=quadrature
        )

    return b5.solve_birger_outer_cycle(
        initial_state,
        GB0,
        G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: c3.mvi.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
