#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8D-2: explicit load-series driver for the four verified KL modes.

Rules:
- loads are explicit user-provided arrays; no hidden load construction;
- no automatic subdivision;
- no retry at reduced load;
- an accepted state is transferred to the next step;
- a failed step is NOT accepted and the series stops immediately.

Transferred state:
Elastic:
    X,Y
Birger:
    X,Y,GB,previous accepted epsilon_i

Each accepted step can be saved independently as:
    STEP_0001.npz
    STEP_0001.json
    ...
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional
import json
import numpy as np
import final_progress as fprog
import final_step_predictor as fpred

import cylinder_kl_stage8d1_four_mode_driver as d1


class KLLoadSeriesError(RuntimeError):
    pass


@dataclass
class KLAcceptedStep:
    step_index: int
    load_factor: float
    state: object
    solver_status: str
    converged: bool
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


@dataclass
class KLLoadSeriesResult:
    mode: str
    converged: bool
    status: str
    requested_steps: int
    accepted_steps: list
    failed_step_index: Optional[int]
    failed_step_status: Optional[str]
    output_dir: Optional[str]
    failed_step_code: Optional[str]=None
    failed_step_diagnostic_ru: Optional[str]=None
    failed_step_recommendation_ru: Optional[str]=None

    @property
    def last_accepted(self):
        return self.accepted_steps[-1] if self.accepted_steps else None


def _failure_diagnostics(status):
    text="" if status is None else str(status)
    if "BIRGER_EPSILON_I_LIMIT_EXCEEDED" in text:
        return ("BIRGER_EPSILON_I_LIMIT_EXCEEDED",
                "Превышен заданный предел интенсивности деформаций Биргера.",
                "Шаг не принят. Уменьшите приращение нагрузки или конечную нагрузку.")
    if "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED" in text:
        return ("BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
                "Превышен заданный предел |J-1| для модели Биргера.",
                "Шаг не принят. Уменьшите приращение нагрузки; не ослабляйте предел без физического обоснования.")
    if "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED" in text:
        return ("BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
                "Обнаружена локальная разгрузка, недопустимая для применяемой деформационной теории Биргера.",
                "Шаг не принят. Проверьте монотонность и уменьшите приращение нагрузки.")
    return (None,None,None)


def _validate_loads(loads,expected_shape):
    out=[]
    for k,item in enumerate(loads,start=1):
        a=np.asarray(item,float)
        if a.shape!=expected_shape:
            raise ValueError(
                f"load step {k} has shape {a.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"load step {k} contains non-finite values")
        out.append(a.copy())
    if not out:
        raise ValueError("at least one explicit load step is required")
    return out


def _validate_load_factors(load_factors,n):
    if load_factors is None:
        return [float(k) for k in range(1,n+1)]
    vals=[float(x) for x in load_factors]
    if len(vals)!=n:
        raise ValueError("load_factors length mismatch")
    if not np.all(np.isfinite(vals)):
        raise ValueError("load_factors contain non-finite values")
    return vals


def _accepted_from_one_step(step_index,load_factor,sol):
    return KLAcceptedStep(
        step_index=int(step_index),
        load_factor=float(load_factor),
        state=sol.state.copy(),
        solver_status=str(sol.status),
        converged=bool(sol.converged),
        solver_iterations=int(sol.solver_iterations),
        geometry=str(sol.geometry),
        GB=None if sol.GB is None else np.asarray(sol.GB,float).copy(),
        epsilon_i=None if sol.epsilon_i is None else np.asarray(sol.epsilon_i,float).copy(),
        sigma_i=None if sol.sigma_i is None else np.asarray(sol.sigma_i,float).copy(),
        detF=None if sol.detF is None else np.asarray(sol.detF,float).copy(),
        unloading_mask=(
            None if sol.unloading_mask is None
            else np.asarray(sol.unloading_mask,bool).copy()
        ),
        max_epsilon_i=(
            None if sol.max_epsilon_i is None else float(sol.max_epsilon_i)
        ),
        max_abs_J_minus_1=(
            None if sol.max_abs_J_minus_1 is None else float(sol.max_abs_J_minus_1)
        ),
        epsilon_limit_exceeded=(
            None if sol.epsilon_limit_exceeded is None
            else bool(sol.epsilon_limit_exceeded)
        ),
        delta_J_limit_exceeded=(
            None if sol.delta_J_limit_exceeded is None
            else bool(sol.delta_J_limit_exceeded)
        ),
        critical_epsilon_i_index=sol.critical_epsilon_i_index,
        critical_J_index=sol.critical_J_index,
        unloading_point_fraction=sol.unloading_point_fraction,
        inner_solve_calls=(
            None if sol.inner_solve_calls is None else int(sol.inner_solve_calls)
        ),
        solver_history=None if sol.solver_history is None else list(sol.solver_history),
    )


def _step_npz_data(step):
    data={
        "X":np.asarray(step.state.X,float),
        "Y":np.asarray(step.state.Y,float),
        "step_index":np.asarray(step.step_index,int),
        "load_factor":np.asarray(step.load_factor,float),
    }
    for name in ("GB","epsilon_i","sigma_i","detF","unloading_mask"):
        val=getattr(step,name)
        if val is not None:
            data[name]=np.asarray(val)
    return data


def save_accepted_step(step,output_dir,*,K0=None):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)

    npz=out/f"STEP_{step.step_index:04d}.npz"
    js=out/f"STEP_{step.step_index:04d}.json"

    tmp_npz=npz.with_name(npz.name+".tmp")
    with tmp_npz.open("wb") as f:
        np.savez_compressed(f,**_step_npz_data(step)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_npz,npz)
    meta={
        "step_index":step.step_index,
        "load_factor":step.load_factor,
        "solver_status":step.solver_status,
        "converged":step.converged,
        "solver_iterations":step.solver_iterations,
        "geometry":step.geometry,
        "has_GB":step.GB is not None,
        "has_epsilon_i":step.epsilon_i is not None,
        "has_sigma_i":step.sigma_i is not None,
        "has_detF":step.detF is not None,
        "has_unloading_mask":step.unloading_mask is not None,
        "max_epsilon_i":step.max_epsilon_i,
        "max_abs_J_minus_1":step.max_abs_J_minus_1,
        "epsilon_limit_exceeded":step.epsilon_limit_exceeded,
        "delta_J_limit_exceeded":step.delta_J_limit_exceeded,
        "critical_epsilon_i_index":step.critical_epsilon_i_index,
        "critical_J_index":step.critical_J_index,
        "unloading_point_fraction":step.unloading_point_fraction,
        "inner_solve_calls":step.inner_solve_calls,
        "solver_history":step.solver_history,
    }
    if step.GB is not None and K0 is not None:
        gb=np.asarray(step.GB,float)
        k0=float(K0)
        # Reviewer J-15: output-only effective Poisson-ratio diagnostic for
        # constant K0 and the already accepted variable GB field.
        nu_eff=(3.0*k0-2.0*gb)/(2.0*(3.0*k0+gb))
        meta["birger_nu_effective_min"]=float(np.min(nu_eff))
        meta["birger_nu_effective_max"]=float(np.max(nu_eff))
    tmp_js=js.with_name(js.name+".tmp")
    with tmp_js.open("w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(meta,indent=2,ensure_ascii=False)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_js,js)
    return npz,js


def run_kl_load_series(
    mode,
    initial_state,
    loads,
    *,
    ops,maps,bc,
    E0,nu0,h,
    quadrature,
    mvi_settings,
    load_factors=None,
    material_diagram=None,
    birger_settings=None,
    initial_GB=None,
    initial_previous_accepted_epsilon_i=None,
    step_predictor_mode="ERROR",
    output_dir=None,
):
    m=d1.validate_mode(mode)
    expected=(3,ops.grid.ns,ops.grid.nx)
    load_list=_validate_loads(loads,expected)
    factors=_validate_load_factors(load_factors,len(load_list))

    predictor_mode=fpred.normalize_mode(step_predictor_mode)
    current_state=initial_state.copy()
    previous_state=None
    previous_factor=None
    current_factor=0.0
    current_GB=None if initial_GB is None else np.asarray(initial_GB,float).copy()
    previous_epsilon=(
        None if initial_previous_accepted_epsilon_i is None
        else np.asarray(initial_previous_accepted_epsilon_i,float).copy()
    )

    accepted=[]
    failed_index=None
    failed_status=None

    for k,(load,factor) in enumerate(zip(load_list,factors),start=1):
        fprog.set_context(step_index=int(k),load_factor=float(factor))
        fprog.emit("step_start",stage="LOAD_STEP")
        step_initial=fpred.predict_state(
            previous_state,current_state,factor,current_factor,previous_factor,
            mode=predictor_mode
        )
        try:
            sol=d1.solve_kl_one_step(
                m,
                step_initial,
                ops=ops,maps=maps,bc=bc,
                E0=E0,nu0=nu0,h=h,
                quadrature=quadrature,
                generalized_load=load,
                mvi_settings=mvi_settings,
                material_diagram=material_diagram,
                birger_settings=birger_settings,
                initial_GB=current_GB,
                previous_accepted_epsilon_i=previous_epsilon,
            )
    
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failed_index=k
            failed_status=f"{type(exc).__name__}: {exc}"
            fprog.emit("step_failed",stage="LOAD_STEP",
                       solver_status=failed_status,converged=False)
            break
        if not sol.converged:
            failed_index=k
            failed_status=sol.status
            break

        step=_accepted_from_one_step(k,factor,sol)
        accepted.append(step)

        if output_dir is not None:
            K0_out=(float(E0)/(3.0*(1.0-2.0*float(nu0)))) if step.GB is not None else None
            save_accepted_step(step,output_dir,K0=K0_out)
        try:
            U=np.einsum("crx,crs->csx",np.asarray(step.state.X,float),np.asarray(step.state.Y,float))
            w=U[2]; js,ix=np.unravel_index(int(np.argmax(np.abs(w))),w.shape)
            fprog.emit("step_complete",stage="STEP_COMPLETE",solver_status=step.solver_status,converged=True,
                       solver_iterations=int(step.solver_iterations),w_abs_max=float(np.max(np.abs(w))),
                       w_at_abs_max=float(w[js,ix]),wmax_is=int(js),wmax_ix=int(ix),max_epsilon_i=step.max_epsilon_i,
                       max_abs_J_minus_1=step.max_abs_J_minus_1)
        except Exception:
            fprog.emit("step_complete",stage="STEP_COMPLETE",solver_status=step.solver_status,converged=True,solver_iterations=int(step.solver_iterations))

        # Transfer ONLY accepted state.
        previous_state=current_state.copy()
        previous_factor=current_factor
        current_state=step.state.copy()
        current_factor=float(factor)

        if d1.is_birger_mode(m):
            current_GB=np.asarray(step.GB,float).copy()
            previous_epsilon=np.asarray(step.epsilon_i,float).copy()

    if failed_index is None:
        return KLLoadSeriesResult(
            mode=m,
            converged=True,
            status="CONVERGED_LOAD_SERIES",
            requested_steps=len(load_list),
            accepted_steps=accepted,
            failed_step_index=None,
            failed_step_status=None,
            output_dir=None if output_dir is None else str(Path(output_dir)),
        )

    failure_code,failure_text,failure_recommendation=_failure_diagnostics(failed_status)
    return KLLoadSeriesResult(
        mode=m,
        converged=False,
        status="LOAD_SERIES_STOPPED_ON_FAILED_STEP",
        requested_steps=len(load_list),
        accepted_steps=accepted,
        failed_step_index=failed_index,
        failed_step_status=failed_status,
        output_dir=None if output_dir is None else str(Path(output_dir)),
        failed_step_code=failure_code,
        failed_step_diagnostic_ru=failure_text,
        failed_step_recommendation_ru=failure_recommendation,
    )
