#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 Stage 5B: explicit load-step series driver.

STRICT RULES
------------
- Load steps are supplied explicitly by the caller.
- No automatic subdivision, retry, step halving, continuation extrapolation,
  or hidden adaptation is performed.
- A step is accepted only if its corresponding solver returns converged=True.
- A failed step stops the series and does NOT overwrite the last accepted state.
- The next accepted step starts from the previous accepted state.
- For BIRGER:
    * step 1 starts from G0 unless caller explicitly provides initial_GB;
    * every next step starts from the accepted GB of the previous step;
    * previous accepted epsilon_i is carried into the next step for unloading
      diagnostics.
- No literature benchmark values are embedded.

Supported modes:
    LINEAR_ELASTIC
    NL_ELASTIC
    BIRGER
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional
import json
import numpy as np
import final_step_predictor as fpred

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage4a_linear_elastic as l4
import cylinder_te1_stage4b_nonlinear_elastic as n4
import cylinder_te1_stage5a_birger as b5
import cylinder_te1_stage6b_linear_birger as lb6

MODES = ("LINEAR_ELASTIC", "LINEAR_BIRGER", "NL_ELASTIC", "BIRGER")


class LoadSeriesError(RuntimeError):
    pass


@dataclass
class AcceptedStep:
    step_index: int
    load_factor: float
    state: mvi.ProfileStateTE1
    solver_status: str
    converged: bool
    solver_iterations: int
    GB: Optional[np.ndarray] = None
    epsilon_i: Optional[np.ndarray] = None
    sigma_i: Optional[np.ndarray] = None
    detF: Optional[np.ndarray] = None
    max_epsilon_i: Optional[float] = None
    max_abs_J_minus_1: Optional[float] = None
    epsilon_limit_exceeded: Optional[bool] = None
    delta_J_limit_exceeded: Optional[bool] = None
    unloading_mask: Optional[np.ndarray] = None
    critical_epsilon_i_index: Optional[tuple] = None
    critical_J_index: Optional[tuple] = None
    unloading_point_fraction: Optional[float] = None


@dataclass
class LoadSeriesResult:
    mode: str
    converged: bool
    status: str
    requested_steps: int
    accepted_steps: list
    failed_step_index: Optional[int]
    failed_step_status: Optional[str]
    output_dir: Optional[str]
    failed_step_code: Optional[str] = None
    failed_step_diagnostic_ru: Optional[str] = None
    failed_step_recommendation_ru: Optional[str] = None

    @property
    def last_accepted(self):
        return self.accepted_steps[-1] if self.accepted_steps else None


def _failure_diagnostics(status):
    text = "" if status is None else str(status)
    if "BIRGER_EPSILON_I_LIMIT_EXCEEDED" in text:
        return {
            "code": "BIRGER_EPSILON_I_LIMIT_EXCEEDED",
            "diagnostic_ru": "Превышен заданный предел интенсивности деформаций Биргера.",
            "recommendation_ru": "Шаг не принят. Уменьшите приращение нагрузки или конечную нагрузку.",
        }
    if "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED" in text:
        return {
            "code": "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
            "diagnostic_ru": "Превышен заданный предел |J-1| для модели Биргера.",
            "recommendation_ru": "Шаг не принят. Уменьшите приращение нагрузки; не ослабляйте предел без физического обоснования.",
        }
    if "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED" in text:
        return {
            "code": "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
            "diagnostic_ru": "Обнаружена локальная разгрузка, недопустимая для применяемой деформационной теории Биргера.",
            "recommendation_ru": "Шаг не принят. Проверьте монотонность и уменьшите приращение нагрузки.",
        }
    if "INVALID_DEFORMATION_MAPPING" in text or "InvalidDeformationMapping" in text:
        return {
            "code": "INVALID_DEFORMATION_MAPPING",
            "diagnostic_ru": "Следующий шаг нагрузки дал detF <= 0: деформационное отображение локально обращается.",
            "recommendation_ru": "Не отключайте проверку detF. Уменьшите шаг нагрузки или конечную нагрузку; принятые шаги до остановки являются последней допустимой веткой.",
        }
    if "MAX_HALFSTEP_ITERATIONS" in text or "HalfstepDidNotConverge" in text:
        return {
            "code": "MVI_HALFSTEP_NOT_CONVERGED",
            "diagnostic_ru": "Внутренний MVI halfstep не достиг projected_tolerance за заданное число итераций.",
            "recommendation_ru": "Уменьшите шаг нагрузки. В Expert можно увеличить max_halfstep_iterations, но расчет станет дольше.",
        }
    if text:
        return {
            "code": "STEP_FAILED",
            "diagnostic_ru": "Шаг нагрузки остановлен с ошибкой решателя.",
            "recommendation_ru": "Проверьте failed_step_status и сохраненные принятые шаги; для продолжения обычно нужен меньший шаг нагрузки.",
        }
    return {"code": None, "diagnostic_ru": None, "recommendation_ru": None}


def _validate_mode(mode):
    m = str(mode).upper()
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return m


def _copy_state(state):
    return state.copy()


def _validate_loads(loads, expected_shape):
    out = []
    for k, item in enumerate(loads):
        a = np.asarray(item, dtype=float)
        if a.shape != expected_shape:
            raise ValueError(
                f"load step {k} has shape {a.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"load step {k} contains non-finite values")
        out.append(a.copy())
    if len(out) < 1:
        raise ValueError("at least one explicit load step is required")
    return out


def _validate_load_factors(load_factors, n):
    if load_factors is None:
        return [float(k+1) for k in range(n)]
    vals = [float(x) for x in load_factors]
    if len(vals) != n:
        raise ValueError("load_factors length mismatch")
    if not np.all(np.isfinite(vals)):
        raise ValueError("load_factors contain non-finite values")
    return vals


def _state_arrays(step: AcceptedStep):
    data = {
        "X": np.asarray(step.state.X, dtype=float),
        "Y": np.asarray(step.state.Y, dtype=float),
        "step_index": np.asarray(step.step_index, dtype=int),
        "load_factor": np.asarray(step.load_factor, dtype=float),
    }
    if step.GB is not None:
        data["GB"] = np.asarray(step.GB, dtype=float)
    if step.epsilon_i is not None:
        data["epsilon_i"] = np.asarray(step.epsilon_i, dtype=float)
    if step.sigma_i is not None:
        data["sigma_i"] = np.asarray(step.sigma_i, dtype=float)
    if step.detF is not None:
        data["detF"] = np.asarray(step.detF, dtype=float)
    if step.unloading_mask is not None:
        data["unloading_mask"] = np.asarray(step.unloading_mask, dtype=bool)
    return data


def _save_step(step: AcceptedStep, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz = output_dir / f"STEP_{step.step_index:04d}.npz"
    tmp_npz=npz.with_name(npz.name+".tmp")
    with tmp_npz.open("wb") as f:
        np.savez_compressed(f, **_state_arrays(step)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_npz,npz)
    meta = {
        "step_index": step.step_index,
        "load_factor": step.load_factor,
        "solver_status": step.solver_status,
        "converged": step.converged,
        "solver_iterations": step.solver_iterations,
        "max_epsilon_i": step.max_epsilon_i,
        "max_abs_J_minus_1": step.max_abs_J_minus_1,
        "epsilon_limit_exceeded": step.epsilon_limit_exceeded,
        "delta_J_limit_exceeded": step.delta_J_limit_exceeded,
        "critical_epsilon_i_index": step.critical_epsilon_i_index,
        "critical_J_index": step.critical_J_index,
        "unloading_point_fraction": step.unloading_point_fraction,
        "has_GB": step.GB is not None,
        "has_epsilon_i": step.epsilon_i is not None,
        "has_sigma_i": step.sigma_i is not None,
        "has_detF": step.detF is not None,
        "has_unloading_mask": step.unloading_mask is not None,
    }
    js=output_dir / f"STEP_{step.step_index:04d}.json"
    tmp_js=js.with_name(js.name+".tmp")
    with tmp_js.open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_js,js)


def _save_summary(result: LoadSeriesResult, output_dir: Path):
    summary = {
        "mode": result.mode,
        "converged": result.converged,
        "status": result.status,
        "requested_steps": result.requested_steps,
        "accepted_step_count": len(result.accepted_steps),
        "failed_step_index": result.failed_step_index,
        "failed_step_status": result.failed_step_status,
        "failed_step_code": result.failed_step_code,
        "failed_step_diagnostic_ru": result.failed_step_diagnostic_ru,
        "failed_step_recommendation_ru": result.failed_step_recommendation_ru,
        "automatic_step_subdivision": False,
        "automatic_retry": False,
        "accepted_steps": [
            {
                "step_index": s.step_index,
                "load_factor": s.load_factor,
                "solver_status": s.solver_status,
                "solver_iterations": s.solver_iterations,
                "max_epsilon_i": s.max_epsilon_i,
                "max_abs_J_minus_1": s.max_abs_J_minus_1,
                "epsilon_limit_exceeded": s.epsilon_limit_exceeded,
                "delta_J_limit_exceeded": s.delta_J_limit_exceeded,
                "critical_epsilon_i_index": s.critical_epsilon_i_index,
                "critical_J_index": s.critical_J_index,
                "unloading_point_fraction": s.unloading_point_fraction,
                "has_unloading_mask": s.unloading_mask is not None,
            }
            for s in result.accepted_steps
        ],
    }
    (output_dir / "SERIES_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def solve_explicit_load_series(
    mode,
    initial_state,
    ops,
    *,
    E0,
    nu0,
    quadrature,
    bc,
    load_steps,
    inner_settings,
    load_factors=None,
    birger_settings=None,
    material_diagram=None,
    initial_GB=None,
    step_predictor_mode="ERROR",
    output_dir=None,
):
    """
    Solve exactly the supplied load sequence.

    Each load_steps[k] is the COMPLETE generalized load array for that step,
    shape (6,ns,nx). The driver does not scale, interpolate, insert, delete,
    retry or subdivide these arrays.
    """
    mode = _validate_mode(mode)
    expected = (6, ops.grid.ns, ops.grid.nx)
    loads = _validate_loads(load_steps, expected)
    factors = _validate_load_factors(load_factors, len(loads))

    if mode in ("LINEAR_BIRGER", "BIRGER"):
        if birger_settings is None:
            raise ValueError("birger_settings required for BIRGER mode")
        if material_diagram is None:
            raise ValueError("material_diagram required for BIRGER mode")

    outdir = Path(output_dir) if output_dir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    accepted = []
    predictor_mode=fpred.normalize_mode(step_predictor_mode)
    current_state = _copy_state(initial_state)
    previous_state = None
    previous_factor = None
    current_factor = 0.0
    current_GB = None if initial_GB is None else np.asarray(initial_GB, float).copy()
    previous_epsilon = None
    failed_index = None
    failed_status = None

    for k, (load, factor) in enumerate(zip(loads, factors), start=1):
        try:
            step_initial = fpred.predict_state(
                previous_state,current_state,factor,current_factor,previous_factor,
                mode=predictor_mode
            )
            if mode == "LINEAR_ELASTIC":
                sol = l4.solve_linear_elastic_mvi(
                    step_initial, ops,
                    E0=E0, nu0=nu0,
                    quadrature=quadrature,
                    generalized_load=load,
                    bc=bc,
                    settings=inner_settings,
                )
                if not sol.converged:
                    failed_index = k
                    failed_status = sol.status
                    break

                step = AcceptedStep(
                    step_index=k,
                    load_factor=factor,
                    state=sol.state.copy(),
                    solver_status=sol.status,
                    converged=True,
                    solver_iterations=sol.cycles,
                )

            elif mode == "LINEAR_BIRGER":
                sol = lb6.solve_linear_birger_load_step(
                    step_initial, ops,
                    E0=E0, nu0=nu0,
                    quadrature=quadrature,
                    generalized_load=load,
                    bc=bc,
                    inner_settings=inner_settings,
                    birger_settings=birger_settings,
                    material_diagram=material_diagram,
                    initial_GB=current_GB,
                    previous_accepted_epsilon_i=previous_epsilon,
                )
                if not sol.converged:
                    failed_index = k
                    failed_status = sol.status
                    break

                step = AcceptedStep(
                    step_index=k,
                    load_factor=factor,
                    state=sol.state.copy(),
                    solver_status=sol.status,
                    converged=True,
                    solver_iterations=sol.outer_iterations,
                    GB=np.asarray(sol.GB, float).copy(),
                    epsilon_i=np.asarray(sol.material_state.epsilon_i, float).copy(),
                    sigma_i=np.asarray(sol.material_state.sigma_i, float).copy(),
                    detF=np.asarray(sol.material_state.detF, float).copy(),
                    max_epsilon_i=float(sol.max_epsilon_i),
                    max_abs_J_minus_1=float(sol.max_abs_J_minus_1),
                    epsilon_limit_exceeded=bool(sol.epsilon_limit_exceeded),
                    delta_J_limit_exceeded=bool(sol.delta_J_limit_exceeded),
                    unloading_mask=(
                        None if sol.unloading_mask is None
                        else np.asarray(sol.unloading_mask, bool).copy()
                    ),
                    critical_epsilon_i_index=sol.critical_epsilon_i_index,
                    critical_J_index=sol.critical_J_index,
                    unloading_point_fraction=float(sol.unloading_point_fraction),
                )

            elif mode == "NL_ELASTIC":
                sol = n4.solve_nonlinear_elastic_mvi(
                    step_initial, ops,
                    E0=E0, nu0=nu0,
                    quadrature=quadrature,
                    generalized_load=load,
                    bc=bc,
                    settings=inner_settings,
                )
                if not sol.converged:
                    failed_index = k
                    failed_status = sol.status
                    break

                step = AcceptedStep(
                    step_index=k,
                    load_factor=factor,
                    state=sol.state.copy(),
                    solver_status=sol.status,
                    converged=True,
                    solver_iterations=sol.cycles,
                )

            else:  # BIRGER
                sol = b5.solve_physical_birger_load_step(
                    step_initial, ops,
                    E0=E0, nu0=nu0,
                    quadrature=quadrature,
                    generalized_load=load,
                    bc=bc,
                    inner_settings=inner_settings,
                    birger_settings=birger_settings,
                    material_diagram=material_diagram,
                    initial_GB=current_GB,
                    previous_accepted_epsilon_i=previous_epsilon,
                )
                if not sol.converged:
                    failed_index = k
                    failed_status = sol.status
                    break

                step = AcceptedStep(
                    step_index=k,
                    load_factor=factor,
                    state=sol.state.copy(),
                    solver_status=sol.status,
                    converged=True,
                    solver_iterations=sol.outer_iterations,
                    GB=np.asarray(sol.GB, float).copy(),
                    epsilon_i=np.asarray(sol.material_state.epsilon_i, float).copy(),
                    sigma_i=np.asarray(sol.material_state.sigma_i, float).copy(),
                    detF=np.asarray(sol.material_state.detF, float).copy(),
                    max_epsilon_i=float(sol.max_epsilon_i),
                    max_abs_J_minus_1=float(sol.max_abs_J_minus_1),
                    epsilon_limit_exceeded=bool(sol.epsilon_limit_exceeded),
                    delta_J_limit_exceeded=bool(sol.delta_J_limit_exceeded),
                    unloading_mask=(
                        None if sol.unloading_mask is None
                        else np.asarray(sol.unloading_mask, bool).copy()
                    ),
                    critical_epsilon_i_index=sol.critical_epsilon_i_index,
                    critical_J_index=sol.critical_J_index,
                    unloading_point_fraction=float(sol.unloading_point_fraction),
                )

            # Accept atomically only after the step solver has converged.
            accepted.append(step)
            previous_state = current_state.copy()
            previous_factor = current_factor
            current_state = step.state.copy()
            current_factor = float(factor)
            if mode in ("LINEAR_BIRGER", "BIRGER"):
                current_GB = step.GB.copy()
                previous_epsilon = step.epsilon_i.copy()

            if outdir is not None:
                _save_step(step, outdir)

        except Exception as exc:
            # Fail explicitly; do not mutate the last accepted state.
            failed_index = k
            failed_status = f"{type(exc).__name__}: {exc}"
            break

    all_ok = (len(accepted) == len(loads))
    failure_diag = _failure_diagnostics(failed_status)
    result = LoadSeriesResult(
        mode=mode,
        converged=all_ok,
        status=("CONVERGED_LOAD_SERIES" if all_ok else "LOAD_SERIES_STOPPED"),
        requested_steps=len(loads),
        accepted_steps=accepted,
        failed_step_index=failed_index,
        failed_step_status=failed_status,
        output_dir=(str(outdir) if outdir is not None else None),
        failed_step_code=failure_diag["code"],
        failed_step_diagnostic_ru=failure_diag["diagnostic_ru"],
        failed_step_recommendation_ru=failure_diag["recommendation_ru"],
    )

    if outdir is not None:
        _save_summary(result, outdir)

    return result
