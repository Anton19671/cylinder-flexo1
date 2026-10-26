#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kirchhoff--Love MVI physical projection for incremental J2 plasticity.

The strict KL MVI algorithm is unchanged.  This module supplies only its weak
projected physical residual at a fixed accepted J2 history.  LINEAR/VK use the
small-strain plane-stress J2 bridge; GL uses finite J2 second-Piola resultants.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import final_progress as fprog
import final_step_predictor as fpred
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b2_profiles as kp
import cylinder_kl_stage8b3_essential_bc as ke
import cylinder_kl_stage8b4_weak_projected as kw
import cylinder_kl_stage8b5_strict_mvi as km
import cylinder_kl_stage8j_j2_material as small_j2
import cylinder_kl_stage8l_j2_finite_material as finite_j2
from final_j2_incremental import J2Parameters, J2State
from final_j2_finite_strain import FiniteJ2State


@dataclass
class KLJ2MVIStepResult:
    state: object
    converged: bool
    status: str
    iterations: int
    history: list
    projected_inf: float
    increment_norm: float
    projected_relative: float
    projected_scale: float
    trial_state: J2State | FiniteJ2State
    material_state: object
    resultants: object


@dataclass
class KLJ2MVIAcceptedStep:
    step_index: int
    requested_step_index: int
    load_factor: float
    state: object
    solver_status: str
    solver_iterations: int
    solver_history: list
    material_state: object
    resultants: object
    history_state: J2State | FiniteJ2State
    subdivision_depth: int = 0
    is_substep: bool = False


@dataclass
class KLJ2MVILoadSeriesResult:
    converged: bool
    status: str
    requested_steps: int
    accepted_steps: list
    failed_step_index: int | None
    failed_step_status: str | None
    subdivisions_used: int = 0


def _resultants(state, ops, *, geometry, committed, parameters, quadrature, with_tangent=False):
    U = kp.reconstruct_fields(state)
    kin = kl.field_kinematics(U, ops, geometry=geometry)
    if str(geometry).upper() == "GREEN_LAGRANGE":
        return kin, finite_j2.resultants_from_gl_kinematics(
            kin, committed=committed, parameters=parameters, quadrature=quadrature
        )
    return kin, small_j2.resultants_from_kinematics(
        kin, committed=committed, parameters=parameters, quadrature=quadrature,
        with_tangent=bool(with_tangent),
    )


def make_j2_projected_function(
    ops, maps, bc, *, geometry, committed, parameters: J2Parameters,
    quadrature, generalized_load,
):
    """Return weak J2 projected equations for the unmodified strict KL MVI."""
    p = np.asarray(generalized_load, float)
    expected = (3, ops.grid.ns, ops.grid.nx)
    if p.shape != expected or not np.isfinite(p).all():
        raise ValueError(f"generalized_load must be finite with shape {expected}")

    def projected(state, direction):
        kin, jout = _resultants(
            state, ops, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
        )
        variations = kw.direction_variations(state, ops, maps, direction)
        return kw.projected_residual_from_variations(
            kin, jout.resultants, variations, ops, bc, geometry=geometry,
            generalized_load=p,
        )

    def prepare(state, direction):
        variations = kw.direction_variations(state, ops, maps, direction)
        labels = tuple(item.label for item in variations)
        norms = np.asarray(
            [np.linalg.norm(item.dU.reshape(-1)) for item in variations], float
        )
        use_tangent = str(geometry).upper() in ("LINEAR", "VON_KARMAN")
        kin, jout = _resultants(
            state, ops, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
            with_tangent=use_tangent,
        )
        return {
            "variations": variations, "labels": labels, "norms": norms,
            "kin": kin, "jout": jout,
        }

    def evaluate_prepared(state, direction, prepared):
        # The profile variations are frozen during a halfstep, but the
        # kinematics, stresses and J2 trial response must be evaluated at the
        # supplied trial state.  Reusing prepared["kin"]/["jout"] here made a
        # finite-difference retry see an unchanged residual and a zero tangent.
        kin, jout = _resultants(
            state, ops, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
            with_tangent=False,
        )
        return kw.projected_residual_from_variations(
            kin, jout.resultants, prepared["variations"], ops, bc,
            geometry=geometry, generalized_load=p,
        )

    def assemble_prepared_tangent(state, direction, prepared, norms, active):
        calg = prepared["jout"].consistent_tangent
        if calg is None or str(geometry).upper() not in ("LINEAR", "VON_KARMAN"):
            return None
        raw = kw.j2_algorithmic_projected_tangent(
            prepared["kin"], prepared["jout"].resultants, calg,
            prepared["variations"], ops, bc, geometry=geometry,
            quadrature=quadrature,
        )
        pos = np.flatnonzero(active)
        if raw.shape != (len(prepared["labels"]), len(prepared["labels"])):
            raise ValueError("invalid KL J2 projected tangent shape")
        return raw[np.ix_(pos, pos)] / (
            norms[pos][:, None] * norms[pos][None, :]
        )

    projected.mvi_prepare_direction = prepare
    projected.mvi_evaluate_prepared = evaluate_prepared
    # The fast algorithmic tangent remains the primary path.  If its Newton
    # direction is not decreasing at a plastic active-set switch, the strict
    # MVI kernel rebuilds the reduced Jacobian from actual trial residuals.
    # Central differences make that exceptional retry independent of the
    # one-sided local constitutive perturbation.
    projected.mvi_tangent_fd_scheme = "CENTRAL"
    projected.mvi_assemble_tangent = assemble_prepared_tangent
    projected.mvi_projected_reference_scale = km._surface_load_resultant_scale(
        ops.grid, p
    )
    projected.j2_evaluate = lambda state: _resultants(
        state, ops, geometry=geometry, committed=committed,
        parameters=parameters, quadrature=quadrature,
    )[1]
    return projected


def solve_j2_mvi_load_step(
    initial_state, ops, *, geometry, committed, parameters: J2Parameters,
    quadrature, generalized_load, bc, settings,
):
    maps = ke.build_x_constraint_maps(ops, bc)
    projected = make_j2_projected_function(
        ops, maps, bc, geometry=geometry, committed=committed,
        parameters=parameters, quadrature=quadrature,
        generalized_load=generalized_load,
    )
    inner = km.solve_internal_mvi(
        initial_state, ops, maps, bc, projected, settings, quadrature=quadrature
    )
    jout = projected.j2_evaluate(inner.state)
    return KLJ2MVIStepResult(
        state=inner.state, converged=bool(inner.converged), status=str(inner.status),
        iterations=int(inner.cycles), history=list(inner.history),
        projected_inf=float(inner.final_projected_inf),
        increment_norm=float(inner.final_increment_rel),
        projected_relative=float(inner.final_projected_relative),
        projected_scale=float(inner.projected_scale),
        trial_state=jout.trial_state, material_state=jout.material_state,
        resultants=jout.resultants,
    )



def _retryable_j2_step_failure(failure):
    """Only a documented numerical failure can trigger bounded bisection.

    Input, grid, seed and shape errors cannot be cured by load increments.
    This is the same whitelist/blacklist contract as CUF J2-MVI.
    """
    message=str(failure).upper()
    nonretry=(
        "MISSING_REQUIRED_INPUT", "UNSPECIFIED_REQUIRED_INPUT",
        "SHAPE MISMATCH", "MUST HAVE SHAPE", "NON-FINITE",
        "NONFINITE", "VALUEERROR:", "GENERALCASEINPUTERROR",
        "INITIAL_PROFILE_SEED_REQUIRED",
    )
    retryable=(
        "DET(F)>0", "DET_F", "INVALID_DEFORMATION_MAPPING",
        "NOT_CONVERGED", "NOT CONVERGE", "MAX_HALFSTEP",
        "HALFSTEPDIDNOTCONVERGE", "MAX_XY",
        "GUARDED_CORRECTION_FAILED", "LOCAL RETURN MAPPING",
        "LOCAL J2", "DIVERG",
    )
    return (not any(tag in message for tag in nonretry)
            and any(tag in message for tag in retryable))


def solve_j2_explicit_load_series(
    initial_state, ops, *, geometry, initial_history, parameters: J2Parameters,
    quadrature, load_steps, load_factors, bc, settings,
    step_predictor_mode="ERROR", automatic_step_subdivision=True,
    max_subdivision_depth=8, on_step_accepted=None,
):
    """Sequential KL J2-MVI loading with safe bisection of a failed increment."""
    loads = [np.asarray(load, float).copy() for load in load_steps]
    expected = (3, ops.grid.ns, ops.grid.nx)
    if not loads or any(load.shape != expected or not np.isfinite(load).all() for load in loads):
        raise ValueError(f"each KL J2 MVI load must be finite with shape {expected}")
    factors = (
        [float(i + 1) for i in range(len(loads))]
        if load_factors is None else [float(value) for value in load_factors]
    )
    if len(factors) != len(loads) or not np.isfinite(factors).all():
        raise ValueError("KL J2 MVI load_factors must be finite and match loads")
    if int(max_subdivision_depth) < 0:
        raise ValueError("max_subdivision_depth must be nonnegative")

    predictor = fpred.normalize_mode(step_predictor_mode)
    queue = [
        {"load": load, "factor": factor, "requested_index": k,
         "depth": 0, "is_substep": False}
        for k, (load, factor) in enumerate(zip(loads, factors), start=1)
    ]
    accepted = []
    state = initial_state.copy()
    history = initial_history.copy()
    previous_state = None
    previous_factor = None
    current_factor = 0.0
    current_load = np.zeros(expected, float)
    subdivisions = 0
    pos = 0

    while pos < len(queue):
        job = queue[pos]
        fprog.set_context(
            step_index=int(len(accepted) + 1),
            requested_step_index=int(job["requested_index"]),
            total_steps=int(len(loads)),
            load_factor=float(job["factor"]),
            subdivision_depth=int(job["depth"]),
            is_substep=bool(job["is_substep"]),
        )
        fprog.emit("step_start", stage="LOAD_STEP")
        guess = fpred.predict_state(
            previous_state, state, float(job["factor"]), current_factor,
            previous_factor, mode=predictor,
        )
        try:
            result = solve_j2_mvi_load_step(
                guess, ops, geometry=geometry, committed=history,
                parameters=parameters, quadrature=quadrature,
                generalized_load=job["load"], bc=bc, settings=settings,
            )
        except Exception as exc:
            result = None
            failure = f"{type(exc).__name__}: {exc}"
        else:
            failure = None if result.converged else str(result.status)

        if failure is not None:
            can_split=(
                bool(automatic_step_subdivision)
                and _retryable_j2_step_failure(failure)
                and int(job["depth"]) < int(max_subdivision_depth)
            )
            if can_split:
                next_depth=int(job["depth"])+1
                mid_factor=0.5*(current_factor+float(job["factor"]))
                mid_load=0.5*(current_load+job["load"])
                # Require actual progress in BOTH the user factor and load.
                # This also stops floating-point midpoint stagnation.
                if (mid_factor!=current_factor
                    and mid_factor!=float(job["factor"])
                    and not np.array_equal(mid_load,current_load)
                    and not np.array_equal(mid_load,job["load"])):
                    job["depth"]=next_depth
                    midpoint={
                        "load": mid_load,
                        "factor": mid_factor,
                        "requested_index": int(job["requested_index"]),
                        "depth": next_depth,
                        "is_substep": True,
                    }
                    queue[pos:pos+1]=[midpoint,job]
                    subdivisions+=1
                    fprog.emit(
                        "load_step_subdivided",
                        stage="LOAD_STEP_SUBDIVISION",
                        requested_step_index=int(job["requested_index"]),
                        subdivision_depth=next_depth,
                        midpoint_factor=float(mid_factor),
                        failed_target_factor=float(job["factor"]),
                        failure_status=str(failure),
                    )
                    continue
            return KLJ2MVILoadSeriesResult(
                False, "LOAD_SERIES_STOPPED", len(loads), accepted,
                int(job["requested_index"]), failure, subdivisions,
            )

        accepted.append(KLJ2MVIAcceptedStep(
            step_index=len(accepted) + 1,
            requested_step_index=int(job["requested_index"]),
            load_factor=float(job["factor"]),
            state=result.state.copy(), solver_status=str(result.status),
            solver_iterations=int(result.iterations),
            solver_history=list(result.history), material_state=result.material_state,
            resultants=result.resultants,
            history_state=result.trial_state.copy(),
            subdivision_depth=int(job["depth"]), is_substep=bool(job["is_substep"]),
        ))
        previous_state = state.copy()
        previous_factor = current_factor
        state = result.state.copy()
        history = result.trial_state.copy()
        current_factor = float(job["factor"])
        current_load = np.asarray(job["load"], float).copy()
        pos += 1
        if on_step_accepted is not None:
            on_step_accepted(accepted[-1])

    return KLJ2MVILoadSeriesResult(
        True, "CONVERGED_LOAD_SERIES", len(loads), accepted, None, None, subdivisions,
    )


__all__ = [
    "KLJ2MVIStepResult", "KLJ2MVIAcceptedStep", "KLJ2MVILoadSeriesResult",
    "make_j2_projected_function", "solve_j2_mvi_load_step",
    "solve_j2_explicit_load_series",
]
