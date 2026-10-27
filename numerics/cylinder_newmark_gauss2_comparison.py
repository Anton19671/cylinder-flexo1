"""Compare independently calculated Newmark/MVI and Gauss-IRK4/MVI.

Only completed saved histories with identical declared physical-case digests
and exact time grids are comparable. No fabricated data or solver fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _summary(folder,name):
    with (folder/name).open("r",encoding="utf-8") as f:
        value=json.load(f)
    if not isinstance(value,dict):
        raise ValueError("DYNAMIC_COMPARISON_INVALID_SUMMARY")
    return value


def _load(folder,name,keys):
    if not isinstance(name,str) or name!=Path(name).name:
        raise ValueError("DYNAMIC_COMPARISON_UNSAFE_FILENAME")
    with np.load(folder/name,allow_pickle=False) as f:
        if not set(keys).issubset(f.files):
            raise ValueError("DYNAMIC_COMPARISON_MISSING_FIELDS")
        result={k:np.array(f[k],float).copy() for k in keys}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("DYNAMIC_COMPARISON_NONFINITE_FIELDS")
    return result


def _errors(candidate,baseline):
    if candidate.shape!=baseline.shape:
        raise ValueError("DYNAMIC_COMPARISON_DIFFERENT_GRID_OR_FIELDS")
    d=candidate-baseline
    peak=float(np.max(np.abs(baseline),initial=0))
    maximum=float(np.max(np.abs(d),initial=0))
    return dict(max_abs_difference=maximum,
                max_rel_to_newmark_peak=None if peak==0 else maximum/peak,
                l2_difference=float(np.linalg.norm(d.ravel())))


def compare_newmark_gauss2_mvi(newmark_folder,gauss_folder):
    """Compare stored physical q/v/p fields; NEVER claim convergence study."""
    nfolder=Path(newmark_folder).expanduser().resolve()
    gfolder=Path(gauss_folder).expanduser().resolve()
    n=_summary(nfolder,"FINAL_RUN_SUMMARY.json")
    g=_summary(gfolder,"GAUSS2_MVI_HISTORY.json")
    if (n.get("calculation_kind")!="DYNAMIC" or n.get("method")!="MVI" or
            n.get("converged") is not True or n.get("status")!="CONVERGED" or
            g.get("status")!="CONVERGED" or
            g.get("method")!="EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI"):
        raise ValueError("DYNAMIC_COMPARISON_REQUIRES_TWO_COMPLETED_MVI_RUNS")
    digest=n.get("physical_case_sha256")
    if (not isinstance(digest,str) or len(digest)!=64 or
            digest!=g.get("physical_case_sha256")):
        raise ValueError("DYNAMIC_COMPARISON_PHYSICAL_CASE_MISMATCH_OR_MISSING")
    rank=n.get("rank")
    if type(rank) is not int or rank!=g.get("rank") or not 1<=rank<=8:
        raise ValueError("DYNAMIC_COMPARISON_MVI_RANK_MISMATCH")
    count=n.get("requested_steps")
    if (type(count) is not int or count<1 or
            n.get("accepted_steps")!=count or
            g.get("n_steps_requested")!=count or g.get("n_steps_accepted")!=count or
            not isinstance(n.get("steps"),list) or len(n["steps"])!=count or
            not isinstance(g.get("steps"),list) or len(g["steps"])!=count):
        raise ValueError("DYNAMIC_COMPARISON_INCOMPLETE_TRAJECTORY")
    temporal=n.get("newmark")
    if not isinstance(temporal,dict):
        raise ValueError("DYNAMIC_COMPARISON_NO_NEWMARK_TIME_GRID")
    dt=float(temporal["dt_s"]); t0=float(temporal["start_time_s"])
    if (not np.isfinite([dt,t0,float(g.get("dt_s",np.nan)),
            float(g.get("t_start_s",np.nan))]).all() or dt<=0 or
            abs(dt-float(g["dt_s"]))>1e-12*dt or
            abs(t0-float(g["t_start_s"]))>1e-12*max(1,abs(t0))):
        raise ValueError("DYNAMIC_COMPARISON_TIME_GRID_MISMATCH")
    fields=("mechanical","velocity","electrical") if n.get("electrical_enabled") else (
        "mechanical","velocity")
    ni=_load(nfolder,"REFERENCE_DYNAMIC_INITIAL.npz",("time_s",)+fields)
    gi=_load(gfolder,"GAUSS2_MVI_INITIAL.npz",("time_s",)+fields)
    for key in ("time_s",)+fields:
        if ni[key].shape!=gi[key].shape or not np.array_equal(ni[key],gi[key]):
            raise ValueError("DYNAMIC_COMPARISON_DIFFERENT_INITIAL_STATES")
    rows=[]
    for i,(ns,gs) in enumerate(zip(n["steps"],g["steps"]),1):
        if ns.get("index")!=i or gs.get("index")!=i:
            raise ValueError("DYNAMIC_COMPARISON_STEP_INDEX_MISMATCH")
        nf=f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz"
        gf=f"GAUSS2_MVI_STEP_{i:04d}.npz"
        if ns.get("state_file")!=nf or gs.get("state_file")!=gf:
            raise ValueError("DYNAMIC_COMPARISON_STEP_FILENAME_MISMATCH")
        a=_load(nfolder,nf,("time_s",)+fields)
        b=_load(gfolder,gf,("time_s",)+fields)
        expected=t0+i*dt
        for value in (a["time_s"],b["time_s"],ns.get("time_s"),gs.get("time_s")):
            vv=float(value)
            if not np.isfinite(vv) or abs(vv-expected)>32*np.finfo(float).eps*max(1,abs(expected)):
                raise ValueError("DYNAMIC_COMPARISON_TIME_GRID_MISMATCH")
        rows.append(dict(index=i,time_s=expected,
                         field_errors={key:_errors(b[key],a[key]) for key in fields}))
    return dict(comparison="Gauss IRK4 fixed-rank MVI versus Newmark fixed-rank MVI",
                physical_case_sha256=digest,rank=rank,time_step_s=dt,
                steps=rows,convergence_study_verified=False)


__all__=("compare_newmark_gauss2_mvi",)
