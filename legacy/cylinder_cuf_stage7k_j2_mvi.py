#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUF MVI physical residuals for incremental J2 (TE1--TE4).

This module does not alter the MVI Y/X half-step algorithm.  It provides the
physical residual for its projection.  The residual is evaluated against a
fixed *accepted* J2 history, so tangent probes and rejected MVI corrections
cannot advance plasticity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import final_progress as fprog
import final_step_predictor as fpred

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as gp
import cylinder_cuf_stage7c_general_boundary as gbc
import cylinder_cuf_stage7d_general_internal_mvi as gmvi
import cylinder_cuf_stage7e_general_driver as driver
import cylinder_cuf_stage7j_j2_material as jm
from final_j2_incremental import J2Parameters, J2State
from final_j2_finite_strain import FiniteJ2State


@dataclass
class CUFJ2MVIStepResult:
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
    material_state: jm.CUFJ2MaterialState
    resultants: object


def make_j2_residual_function(
    ops, *, geometry, committed, parameters: J2Parameters, quadrature, basis,
    generalized_load, bc,
):
    """Return the exact physical CUF J2 residual at fixed accepted history."""
    p = np.asarray(generalized_load, float)
    expected = (3, basis.m, ops.grid.ns, ops.grid.nx)
    if p.shape != expected or not np.isfinite(p).all():
        raise ValueError(f"generalized_load must be finite with shape {expected}")
    if bc.m != basis.m:
        raise ValueError("basis/BC m mismatch")
    mode = str(geometry).upper()

    def evaluate(state):
        q, qx, qs = gp.profile_fields_and_first_derivatives(state, ops)
        if mode in ("LINEAR", "VON_KARMAN"):
            jout = jm.resultants_small_strain(
                q, qx, qs, geometry=mode, R=ops.grid.radius,
                committed=committed, parameters=parameters,
                quadrature=quadrature, basis=basis,
            )
        elif mode == "GREEN_LAGRANGE":
            jout = jm.resultants_finite_strain(
                q, qx, qs, R=ops.grid.radius, committed=committed,
                parameters=parameters, quadrature=quadrature, basis=basis,
            )
        else:
            raise ValueError("CUF J2 MVI geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")
        raw = g7.general_residuals(
            jout.resultants, Dx=ops.Dx, Ds=ops.Ds, generalized_load=p
        )
        assembled = gbc.assemble_strong_residual_with_end_bcs(
            raw, q, jout.resultants, bc, reject_fully_free=True
        )
        return assembled.reshape(-1), jout

    def residual(state):
        return evaluate(state)[0]

    # The projected reference scale is diagnostic only; it does not replace
    # the original projected equilibrium equations.
    residual.mvi_projected_reference_residual = p.reshape(-1).copy()
    residual.j2_evaluate = evaluate
    return residual


def solve_j2_mvi_load_step(
    initial_state, ops, *, geometry, committed, parameters: J2Parameters,
    quadrature, basis, generalized_load, bc, inner_settings, E0,
    validate_initial_seed=True,
):
    """Solve one projected J2 equilibrium step without committing history."""
    residual = make_j2_residual_function(
        ops, geometry=geometry, committed=committed, parameters=parameters,
        quadrature=quadrature, basis=basis, generalized_load=generalized_load,
        bc=bc,
    )
    if validate_initial_seed:
        driver.require_seed_if_nonzero_residual(initial_state, ops.grid, residual)
    inner = gmvi.solve_internal_mvi(
        initial_state, ops.grid, bc, residual, inner_settings,
        measure_multipliers=driver._v4_measure_multipliers(
            ops, basis, bc, E0=E0, quadrature=quadrature
        ),
        quadrature=quadrature,
    )
    _residual, jout = residual.j2_evaluate(inner.state)
    return CUFJ2MVIStepResult(
        state=inner.state, converged=bool(inner.converged), status=str(inner.status),
        iterations=int(inner.cycles), history=list(inner.history),
        projected_inf=float(inner.final_projected_inf),
        increment_norm=float(inner.final_increment_rel),
        projected_relative=float(inner.final_projected_relative),
        projected_scale=float(inner.projected_scale),
        trial_state=jout.trial_state, material_state=jout.material_state,
        resultants=jout.resultants,
    )


@dataclass
class CUFJ2MVIAcceptedStep:
    step_index: int
    requested_step_index: int
    load_factor: float
    state: object
    solver_status: str
    solver_iterations: int
    solver_history: list
    material_state: jm.CUFJ2MaterialState
    resultants: object
    history_state: J2State | FiniteJ2State
    subdivision_depth: int = 0
    is_substep: bool = False


@dataclass
class CUFJ2MVILoadSeriesResult:
    converged: bool
    status: str
    requested_steps: int
    accepted_steps: list
    failed_step_index: int | None
    failed_step_status: str | None
    subdivisions_used: int = 0
    failed_step_code: str | None = None
    failed_step_diagnostic_ru: str | None = None
    failed_step_recommendation_ru: str | None = None


def _failure_diagnostics(status):
    """Classify a J2-MVI failure without hiding the original status."""
    text = "" if status is None else str(status)
    upper = text.upper()
    if "DET(F)>0" in upper or "DET_F" in upper or "INVALID_DEFORMATION_MAPPING" in upper:
        return (
            "J2_INVALID_DEFORMATION",
            "В пробной деформации J2 получено det(F) <= 0.",
            "Шаг нагрузки отклонён. Уменьшите приращение нагрузки; проверку det(F) не отключать.",
        )
    if "LOCAL RETURN MAPPING" in upper or "LOCAL J2" in upper:
        return (
            "J2_LOCAL_RETURN_FAILED",
            "Локальный J2 return map не сошёлся.",
            "Шаг нагрузки отклонён. Проверьте параметры J2 и уменьшите приращение нагрузки.",
        )
    if "MAX_HALFSTEP" in upper or "HALFSTEPDIDNOTCONVERGE" in upper or "NOT_CONVERGED" in upper or "NOT CONVERGE" in upper:
        return (
            "MVI_HALFSTEP_NOT_CONVERGED",
            "MVI halfstep не достиг заданной сходимости.",
            "Шаг нагрузки отклонён. Уменьшите приращение нагрузки или проверьте projected/full residual.",
        )
    if text:
        return (
            "J2_MVI_STEP_FAILED",
            "Шаг J2-MVI остановлен с численной ошибкой.",
            "Проверьте исходный статус; обычно требуется уменьшить приращение нагрузки.",
        )
    return (None, None, None)


def solve_j2_explicit_load_series(
    initial_state, ops, *, geometry, initial_history, parameters: J2Parameters,
    quadrature, basis, load_steps, load_factors, bc, inner_settings, E0,
    step_predictor_mode="ERROR", automatic_step_subdivision=True,
    max_subdivision_depth=8, on_step_accepted=None,
):
    """Sequential J2-MVI loading with safe bisection of a failed increment.

    A rejected MVI trial never changes the profile state or J2 internal state.
    The inserted midpoint is solved from the last accepted state; unloading and
    reloading are therefore evaluated by the same return-mapping history.
    """
    loads = [np.asarray(load, float).copy() for load in load_steps]
    expected = (3, basis.m, ops.grid.ns, ops.grid.nx)
    if not loads or any(load.shape != expected or not np.isfinite(load).all() for load in loads):
        raise ValueError(f"each CUF J2 MVI load must be finite with shape {expected}")
    factors = (
        [float(i + 1) for i in range(len(loads))]
        if load_factors is None else [float(value) for value in load_factors]
    )
    if len(factors) != len(loads) or not np.isfinite(factors).all():
        raise ValueError("CUF J2 MVI load_factors must be finite and match loads")
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
                parameters=parameters, quadrature=quadrature, basis=basis,
                generalized_load=job["load"], bc=bc, inner_settings=inner_settings,
                E0=E0, validate_initial_seed=(len(accepted) == 0),
            )
        except Exception as exc:
            result = None
            failure = f"{type(exc).__name__}: {exc}"
        else:
            failure = None if result.converged else str(result.status)

        if failure is not None:
            # Only numerical failures are repairable by a smaller load
            # increment.  Input, shape, seed and configuration errors must
            # stop immediately instead of being retried as microsteps.
            upper_failure = str(failure).upper()
            nonretry = (
                "MISSING_REQUIRED_INPUT", "UNSPECIFIED_REQUIRED_INPUT",
                "SHAPE MISMATCH", "MUST HAVE SHAPE", "NON-FINITE",
                "NONFINITE", "VALUEERROR:", "GENERALCASEINPUTERROR",
                "INITIAL_PROFILE_SEED_REQUIRED",
            )
            retryable = (
                "DET(F)>0", "DET_F", "INVALID_DEFORMATION_MAPPING",
                "NOT_CONVERGED", "NOT CONVERGE",
                "MAX_HALFSTEP", "HALFSTEPDIDNOTCONVERGE",
                "MAX_XY", "GUARDED_CORRECTION_FAILED",
                "LOCAL RETURN MAPPING", "LOCAL J2",
                "DIVERG",
            )
            can_split = (
                bool(automatic_step_subdivision)
                and not any(token in upper_failure for token in nonretry)
                and any(token in upper_failure for token in retryable)
                and int(job["depth"]) < int(max_subdivision_depth)
            )
            if can_split:
                next_depth = int(job["depth"]) + 1
                midpoint_factor = 0.5 * (current_factor + float(job["factor"]))
                midpoint_load = 0.5 * (current_load + job["load"])
                # Guard against a numerically unchanged midpoint.
                if (
                    midpoint_factor != float(job["factor"])
                    and not np.array_equal(midpoint_load, job["load"])
                ):
                    # Increase the depth of the original target as well.
                    # Otherwise the same failed target remains at depth=0
                    # and bypasses max_subdivision_depth indefinitely.
                    job["depth"] = next_depth
                    midpoint = {
                        "load": midpoint_load,
                        "factor": midpoint_factor,
                        "requested_index": int(job["requested_index"]),
                        "depth": next_depth,
                        "is_substep": True,
                    }
                    queue[pos:pos + 1] = [midpoint, job]
                    subdivisions += 1
                    code, diagnostic_ru, recommendation_ru = _failure_diagnostics(failure)
                    fprog.emit(
                        "load_step_subdivided",
                        stage="LOAD_STEP_SUBDIVISION",
                        requested_step_index=int(job["requested_index"]),
                        failed_target_factor=float(job["factor"]),
                        midpoint_factor=float(midpoint_factor),
                        subdivision_depth=int(next_depth),
                        failure_code=code,
                        failure_status=str(failure),
                    )
                    continue
            code, diagnostic_ru, recommendation_ru = _failure_diagnostics(failure)
            return CUFJ2MVILoadSeriesResult(
                False, "LOAD_SERIES_STOPPED", len(loads), accepted,
                int(job["requested_index"]), failure, subdivisions,
                code, diagnostic_ru, recommendation_ru,
            )

        accepted.append(CUFJ2MVIAcceptedStep(
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

    return CUFJ2MVILoadSeriesResult(
        True, "CONVERGED_LOAD_SERIES", len(loads), accepted, None, None, subdivisions,
    )



__all__ = [
    "CUFJ2MVIStepResult", "CUFJ2MVIAcceptedStep", "CUFJ2MVILoadSeriesResult",
    "make_j2_residual_function", "solve_j2_mvi_load_step",
    "solve_j2_explicit_load_series",
]