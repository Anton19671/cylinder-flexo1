#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8D-1: unified one-load-step driver for the six KL modes.

Modes:
    KL_LINEAR_ELASTIC
    KL_LINEAR_BIRGER
    KL_VK_ELASTIC
    KL_VK_BIRGER
    KL_GL_ELASTIC
    KL_GL_BIRGER

No new mechanics:
- elastic modes use Stage8B-5 strict MVI projected from the matching Stage9
  strong-FDM residual;
- Birger modes use the same strong projection inside the Stage8C-3 generic
  Stage5A outer-cycle adapter.

The driver only dispatches and packages results.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

import cylinder_kl_stage8b5_strict_mvi as km
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_kl_stage8c3_birger_outer as bo


MODES=(
    "KL_LINEAR_ELASTIC",
    "KL_LINEAR_BIRGER",
    "KL_VK_ELASTIC",
    "KL_VK_BIRGER",
    "KL_GL_ELASTIC",
    "KL_GL_BIRGER",
)


@dataclass
class KLOneStepResult:
    mode: str
    state: object
    converged: bool
    status: str
    solver_iterations: int
    geometry: str
    GB: Optional[np.ndarray]=None
    epsilon_i: Optional[np.ndarray]=None
    sigma_i: Optional[np.ndarray]=None
    detF: Optional[np.ndarray]=None
    unloading_mask: Optional[np.ndarray]=None
    max_epsilon_i: Optional[float]=None
    max_abs_J_minus_1: Optional[float]=None
    epsilon_limit_exceeded: Optional[bool]=None
    delta_J_limit_exceeded: Optional[bool]=None
    critical_epsilon_i_index: Optional[tuple]=None
    critical_J_index: Optional[tuple]=None
    unloading_point_fraction: Optional[float]=None
    inner_solve_calls: Optional[int]=None
    solver_history: Optional[list]=None


def validate_mode(mode):
    m=str(mode).upper()
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return m


def geometry_for_mode(mode):
    m=validate_mode(mode)
    if m in ("KL_LINEAR_ELASTIC","KL_LINEAR_BIRGER"):
        return "LINEAR"
    if m in ("KL_GL_ELASTIC","KL_GL_BIRGER"):
        return "GREEN_LAGRANGE"
    return "VON_KARMAN"


def is_birger_mode(mode):
    return validate_mode(mode) in ("KL_LINEAR_BIRGER","KL_VK_BIRGER","KL_GL_BIRGER")


def solve_kl_one_step(
    mode,
    initial_state,
    *,
    ops,maps,bc,
    E0,nu0,h,
    quadrature,
    generalized_load,
    mvi_settings,
    material_diagram=None,
    birger_settings=None,
    initial_GB=None,
    previous_accepted_epsilon_i=None,
):
    m=validate_mode(mode)
    geometry=geometry_for_mode(m)
    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    E0=float(E0); nu0=float(nu0); h=float(h)
    if E0<=0 or h<=0 or not (-1.0<nu0<0.5):
        raise ValueError("invalid E0/nu0/h")

    if not is_birger_mode(m):
        pf=km.make_elastic_projected_function(
            ops,maps,bc,
            E=E0,nu=nu0,h=h,geometry=geometry,
            generalized_load=p,
            resultants_method="QUADRATURE",
            quadrature=quadrature,
        )
        sol=km.solve_internal_mvi(
            initial_state,ops,maps,bc,pf,mvi_settings,quadrature=quadrature
        )
        return KLOneStepResult(
            mode=m,
            state=sol.state.copy(),
            converged=bool(sol.converged),
            status=sol.status,
            solver_iterations=int(sol.cycles),
            geometry=geometry,
            solver_history=list(sol.history),
        )

    if material_diagram is None:
        raise ValueError("material_diagram is required for KL Birger modes")
    if birger_settings is None:
        raise ValueError("birger_settings is required for KL Birger modes")

    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)
    expected_G=(len(quadrature.z),ops.grid.ns,ops.grid.nx)

    if initial_GB is None:
        GB0=np.full(expected_G,G0,float)
    else:
        GB0=np.asarray(initial_GB,float).copy()
        if GB0.shape!=expected_G:
            raise ValueError(f"initial_GB must have shape {expected_G}")
        if np.any(~np.isfinite(GB0)) or np.any(GB0<=0):
            raise ValueError("initial_GB must be finite and positive")

    counter=bo.KLInnerSolveCounter()
    sol,counter=bo.solve_kl_birger_load_step(
        initial_state,GB0,
        G0=G0,
        material_diagram=material_diagram,
        birger_settings=birger_settings,
        ops=ops,maps=maps,bc=bc,
        K0=K0,quadrature=quadrature,geometry=geometry,
        generalized_load=p,mvi_settings=mvi_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
        inner_counter=counter,
    )

    return KLOneStepResult(
        mode=m,
        state=sol.state.copy(),
        converged=bool(sol.converged),
        status=sol.status,
        solver_iterations=int(sol.outer_iterations),
        geometry=geometry,
        GB=np.asarray(sol.GB,float).copy(),
        epsilon_i=np.asarray(sol.material_state.epsilon_i,float).copy(),
        sigma_i=np.asarray(sol.material_state.sigma_i,float).copy(),
        detF=np.asarray(sol.material_state.detF,float).copy(),
        unloading_mask=(
            None if sol.unloading_mask is None
            else np.asarray(sol.unloading_mask,bool).copy()
        ),
        max_epsilon_i=float(sol.max_epsilon_i),
        max_abs_J_minus_1=float(sol.max_abs_J_minus_1),
        epsilon_limit_exceeded=bool(sol.epsilon_limit_exceeded),
        delta_J_limit_exceeded=bool(sol.delta_J_limit_exceeded),
        critical_epsilon_i_index=sol.critical_epsilon_i_index,
        critical_J_index=sol.critical_J_index,
        unloading_point_fraction=float(sol.unloading_point_fraction),
        inner_solve_calls=int(counter.calls),
        solver_history=list(sol.history),
    )
