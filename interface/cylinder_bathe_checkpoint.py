"""Fail-closed opt-in restart of *accepted* physical Bathe time states.

No residual, load, physical equation or numerical method is changed here.
The original physical CASE must match exactly. Never infer or synthesize an
unaccepted step; either stage can have been written before a job stopped.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _checked_state(path, qshape, pshape, *, t, factor, trace, method, rank,
                   initial=False):
    if not path.is_file():
        raise ValueError("BATHE_RESTART_ACCEPTED_STATE_MISSING:" + path.name)
    with np.load(path, allow_pickle=False) as raw:
        keys = set(raw.files)
        required = {"time_s", "mechanical", "velocity", "acceleration"}
        if not initial:
            required |= {"load_factor"}
        if method == "MVI" and not initial:
            required |= {"x_profiles", "y_profiles", "state_scales"}
        if not required.issubset(keys):
            raise ValueError("BATHE_RESTART_STATE_FIELDS_MISSING:" + path.name)
        q = np.array(raw["mechanical"], dtype=float)
        v = np.array(raw["velocity"], dtype=float)
        a = np.array(raw["acceleration"], dtype=float)
        p = np.array(raw["electrical"], dtype=float) if "electrical" in keys else None
        if (q.shape != qshape or v.shape != qshape or a.shape != qshape or
                (p is None) != (pshape is None) or
                (p is not None and p.shape != pshape) or
                any(not np.isfinite(f).all() for f in (q, v, a) if f is not None) or
                (p is not None and not np.isfinite(p).all())):
            raise ValueError("BATHE_RESTART_STATE_SHAPE_OR_FINITE:" + path.name)
        time = np.asarray(raw["time_s"], dtype=float)
        if time.shape != () or not math.isclose(float(time), t, abs_tol=1e-12):
            raise ValueError("BATHE_RESTART_STATE_TIME_MISMATCH:" + path.name)
        if not initial:
            saved_factor = np.asarray(raw["load_factor"], dtype=float)
            if (saved_factor.shape != () or
                    not math.isclose(float(saved_factor), factor,
                                     rel_tol=1e-11, abs_tol=1e-11)):
                raise ValueError("BATHE_RESTART_LOAD_FACTOR_MISMATCH:" + path.name)
        c, d = trace
        if (float(np.max(np.abs(c @ q.ravel() - d), initial=0.)) > 1e-9 or
                float(np.max(np.abs(c @ v.ravel()), initial=0.)) > 1e-8 or
                float(np.max(np.abs(c @ a.ravel()), initial=0.)) > 1e-7):
            raise ValueError("BATHE_RESTART_BOUNDARY_STATE_MISMATCH:" + path.name)
        prev = None
        if method == "MVI" and not initial:
            nx, ns = qshape[1:]
            nc = qshape[0] + (0 if pshape is None else pshape[0])
            x = np.array(raw["x_profiles"], dtype=float)
            y = np.array(raw["y_profiles"], dtype=float)
            scales = np.array(raw["state_scales"], dtype=float)
            physical = q if p is None else np.concatenate((q,p), axis=0)
            if (x.shape != (nc, rank, nx) or y.shape != (nc, rank, ns) or
                    scales.shape != physical.shape or
                    not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not np.isfinite(scales).all() or np.any(scales <= 0)):
                raise ValueError("BATHE_RESTART_MVI_FACTOR_SHAPE:" + path.name)
            represented = np.einsum("crx,crs->cxs",x,y)*scales
            err = np.abs(represented - physical)
            allowed = 1e-10 * np.maximum(
                scales,np.maximum(np.abs(represented),np.abs(physical)))
            if np.any(err > allowed):
                raise ValueError("BATHE_RESTART_MVI_RECONSTRUCTION:" + path.name)
            # Previous accepted stage's certificate was checked by _accept()
            # *before* the journal entry was persisted. This minimal passport
            # enables the unchanged, strictly checked rank-fixed continuation.
            prev = SimpleNamespace(
                converged=True,
                mvi_certificate={"accepted":True,"fixed_rank":rank},
                x_profiles=x,y_profiles=y,scales=scales,
                mechanical=q,electrical=p)
    return q,p,v,a,prev


def load_checked_bathe_checkpoint(out, case, prepared, dyn, *, q0, p0, v0,
                                  a0, initial_trace, debug_restart):
    """Read only full accepted stages, verifying CASE, time and constraints."""
    out = Path(out)
    stored_case = _read_json(out/"REFERENCE_CASE.json")
    # The restart toggle is an execution-control flag, not physical data.
    # It may be enabled on retry of an older archive without changing
    # geometry, loads, method, time grid, material or any other CASE key.
    original = dict(stored_case)
    current = dict(case)
    original_dynamics = dict(original.get("dynamics") or {})
    current_dynamics = dict(current.get("dynamics") or {})
    original_dynamics.pop("bathe_resume_accepted", None)
    current_dynamics.pop("bathe_resume_accepted", None)
    original["dynamics"] = original_dynamics
    current["dynamics"] = current_dynamics
    if original != current:
        raise ValueError("BATHE_RESTART_PHYSICAL_CASE_CHANGED")
    summary = _read_json(out/"FINAL_RUN_SUMMARY.json")
    cfg = dyn.settings
    expected_rank=(prepared.mvi_settings.rank if prepared.method=="MVI" else None)
    expected = {
        "solver_backend":"VARIATIONAL_REFERENCE",
        "calculation_kind":"DYNAMIC",
        "integrator":"BATHE",
        "method":prepared.method,
        "theory":prepared.problem.functional.theory,
        "geometry_model":case["geometry_model"],
        "fdm_backend":prepared.fdm_backend,
        "rank":expected_rank,
        "requested_steps":cfg.n_steps,
        "debug_cold_start_each_full_step":debug_restart,
    }
    for key,value in expected.items():
        if summary.get(key) != value:
            raise ValueError("BATHE_RESTART_SUMMARY_CHANGED:" + key)
    for key,value in (
        ("time_step_s",cfg.dt_s),("start_time_s",cfg.t_start_s),
        ("density_kg_m3",dyn.density_kg_m3)):
        if not math.isclose(float(summary[key]),float(value),rel_tol=1e-13,abs_tol=1e-14):
            raise ValueError("BATHE_RESTART_NUMERICAL_PASSPORT_CHANGED:" + key)
    accepted=summary.get("accepted_steps")
    steps=summary.get("steps")
    if (type(accepted) is not int or not 0<=accepted<=cfg.n_steps or
            not isinstance(steps,list) or len(steps)!=accepted):
        raise ValueError("BATHE_RESTART_ACCEPTED_COUNT_INVALID")
    if summary.get("status") not in ("NOT_STARTED","IN_PROGRESS","CONVERGED"):
        raise ValueError("BATHE_RESTART_PREVIOUSLY_REJECTED_OR_UNSUPPORTED")
    if summary.get("converged") is not (accepted==cfg.n_steps):
        raise ValueError("BATHE_RESTART_SUMMARY_STATUS_MISMATCH")
    if accepted==cfg.n_steps and summary.get("status")!="CONVERGED":
        raise ValueError("BATHE_RESTART_COMPLETED_STATUS_MISMATCH")
    if accepted==0 and summary.get("status")!="NOT_STARTED":
        raise ValueError("BATHE_RESTART_INITIAL_STATUS_MISMATCH")
    if 0<accepted<cfg.n_steps and summary.get("status")!="IN_PROGRESS":
        raise ValueError("BATHE_RESTART_PARTIAL_STATUS_MISMATCH")

    qshape=q0.shape
    pshape=None if p0 is None else p0.shape
    initial=_checked_state(
        out/"REFERENCE_DYNAMIC_INITIAL.npz",qshape,pshape,
        t=cfg.t_start_s,factor=0.,trace=initial_trace,
        method=prepared.method,rank=expected_rank,initial=True)
    for expected_array,actual_array,label in (
        (q0,initial[0],"q0"),(v0,initial[2],"v0"),(a0,initial[3],"a0")):
        if not np.allclose(expected_array,actual_array,rtol=1e-13,atol=1e-13):
            raise ValueError("BATHE_RESTART_INITIAL_STATE_CHANGED:" + label)
    if p0 is not None and not np.allclose(p0,initial[1],rtol=1e-13,atol=1e-13):
        raise ValueError("BATHE_RESTART_INITIAL_ELECTRIC_STATE_CHANGED")
    latest=initial
    previous=None
    for i,row in enumerate(steps,1):
        if (not isinstance(row,dict) or row.get("index")!=i or
                row.get("converged") is not True):
            raise ValueError("BATHE_RESTART_INVALID_ACCEPTED_ROW")
        t1=cfg.t_start_s+(i-1+2-math.sqrt(2))*cfg.dt_s
        t2=cfg.t_start_s+i*cfg.dt_s
        gamma_file=f"BATHE_SUBSTEP_{i:04d}_GAMMA.npz"
        end_file=f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz"
        if (row.get("first_substep_file") != gamma_file or
                row.get("state_file") != end_file or
                not math.isclose(float(row["time_gamma_s"]),t1,abs_tol=1e-12) or
                not math.isclose(float(row["time_s"]),t2,abs_tol=1e-12) or
                not math.isclose(float(row["load_factor"]),
                                 dyn.time_law.value(t2),abs_tol=1e-11)):
            raise ValueError("BATHE_RESTART_ACCEPTED_ROW_TIME_OR_LOAD_CHANGED")
        if prepared.method=="MVI" and row.get("status") not in (
                "CONVERGED_FULL","CONVERGED_PROJECTED"):
            raise ValueError("BATHE_RESTART_PREVIOUS_MVI_UNACCEPTED")
        _checked_state(out/gamma_file,qshape,pshape,t=t1,
                       factor=dyn.time_law.value(t1),trace=initial_trace,
                       method=prepared.method,rank=expected_rank)
        latest=_checked_state(out/end_file,qshape,pshape,t=t2,
                              factor=dyn.time_law.value(t2),trace=initial_trace,
                              method=prepared.method,rank=expected_rank)
        previous=latest[4]

    # Atomic step summary is the acceptance boundary. Persist, but never
    # treat transient stage files from an interrupted unaccepted step as
    # physical history.
    for path in out.iterdir():
        match=re.fullmatch(
            r"(?:BATHE_SUBSTEP_(\d{4})_GAMMA|REFERENCE_DYNAMIC_STEP_(\d{4}))\.npz",
            path.name)
        if match and int(match.group(1) or match.group(2))>accepted:
            quarantine=out/"UNACCEPTED_INTERRUPTED_STAGE_FILES"
            quarantine.mkdir(exist_ok=True)
            target=quarantine/path.name
            if target.exists():
                raise ValueError("BATHE_RESTART_UNACCEPTED_FILE_COLLISION")
            shutil.move(str(path),str(target))
    print("BATHE_RESTART_CHECKED_ACCEPTED_STATES",
          accepted,"of",cfg.n_steps,flush=True)
    return latest[:4],summary,list(steps),previous,(accepted+1)


__all__=["load_checked_bathe_checkpoint"]
