"""Compare independently calculated Newmark/MVI and Gauss-IRK4/MVI.

Only completed saved histories with identical declared physical-case digests
and exact time grids are comparable. No fabricated data or solver fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cylinder_dynamic_case_passport import dynamic_physics_sha256
from cylinder_newmark_audit import audit_newmark_step
from cylinder_gauss2_mvi_history import _bounded_field_error
from cylinder_gauss2_rk import GAUSS2_A, GAUSS2_B, GAUSS2_C



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
    physical_n=_summary(nfolder,"REFERENCE_CASE.json")
    physical_g=_summary(gfolder,"GAUSS2_PHYSICAL_CASE.json")
    if (dynamic_physics_sha256(physical_n)!=digest or
            dynamic_physics_sha256(physical_g)!=digest):
        raise ValueError("DYNAMIC_COMPARISON_SAVED_PHYSICS_DISAGREE_WITH_PASSPORT")
    fields=("mechanical","velocity","electrical") if n.get("electrical_enabled") else (
        "mechanical","velocity")
    ni=_load(nfolder,"REFERENCE_DYNAMIC_INITIAL.npz",("time_s",)+fields)
    gi=_load(gfolder,"GAUSS2_MVI_INITIAL.npz",("time_s",)+fields)
    for key in ("time_s",)+fields:
        if ni[key].shape!=gi[key].shape or not np.array_equal(ni[key],gi[key]):
            raise ValueError("DYNAMIC_COMPARISON_DIFFERENT_INITIAL_STATES")
    ni_all=_load(nfolder,"REFERENCE_DYNAMIC_INITIAL.npz",("acceleration",))
    ni_prev={**ni,**ni_all}
    gi_prev=gi
    beta=float(temporal["beta"]);gamma=float(temporal["gamma"])
    if not np.isfinite([beta,gamma]).all() or beta<=0 or gamma<0:
        raise ValueError("DYNAMIC_COMPARISON_INVALID_NEWMARK_PARAMETERS")
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
        kin=_load(nfolder,nf,("acceleration",))
        audit_newmark_step(ni_prev["mechanical"],ni_prev["velocity"],
                           ni_prev["acceleration"],a["mechanical"],
                           a["velocity"],kin["acceleration"],
                           dt_s=dt,beta=beta,gamma=gamma)
        ni_prev={**a,**kin}
        stage=_load(gfolder,gf,("stage_time_s","stage_mechanical",
                               "stage_velocity","stage_acceleration",
                               "x_profiles","y_profiles","state_scales"))
        for index in range(2):
            if abs(float(stage["stage_time_s"][index])-
                   (t0+(i-1+GAUSS2_C[index])*dt)) > 1e-12*max(1,abs(expected)):
                raise ValueError("DYNAMIC_COMPARISON_GAUSS_STAGE_TIME_MISMATCH")
        aa=stage["stage_acceleration"]
        expected_v=gi_prev["velocity"][None]+dt*np.einsum("ij,j...->i...",GAUSS2_A,aa)
        expected_q=(gi_prev["mechanical"][None]+
                    dt*GAUSS2_C[:,None,None,None]*gi_prev["velocity"][None]+
                    dt*dt*np.einsum("ij,j...->i...",GAUSS2_A@GAUSS2_A,aa))
        _bounded_field_error(stage["stage_velocity"],expected_v,"STAGE_VELOCITY")
        _bounded_field_error(stage["stage_mechanical"],expected_q,"STAGE_DISPLACEMENT")
        final_q=gi_prev["mechanical"]+dt*np.einsum("i,i...->...",GAUSS2_B,stage["stage_velocity"])
        final_v=gi_prev["velocity"]+dt*np.einsum("i,i...->...",GAUSS2_B,aa)
        _bounded_field_error(b["mechanical"],final_q,"END_DISPLACEMENT")
        _bounded_field_error(b["velocity"],final_v,"END_VELOCITY")
        combined=b["mechanical"] if "electrical" not in b else np.concatenate(
            (b["mechanical"],b["electrical"]))
        x=stage["x_profiles"];y=stage["y_profiles"];scale=stage["state_scales"]
        if (x.shape!=(combined.shape[0],rank,combined.shape[1]) or
                y.shape!=(combined.shape[0],rank,combined.shape[2]) or
                scale.shape!=combined.shape or np.any(scale<=0)):
            raise ValueError("DYNAMIC_COMPARISON_GAUSS_PROFILE_SHAPE_MISMATCH")
        _bounded_field_error(scale*np.einsum("crx,crs->cxs",x,y),
                             combined,"FIXED_RANK_ENDPOINT")
        gi_prev=b
        rows.append(dict(index=i,time_s=expected,
                         field_errors={key:_errors(b[key],a[key]) for key in fields}))
    return dict(comparison="Gauss IRK4 fixed-rank MVI versus Newmark fixed-rank MVI",
                physical_case_sha256=digest,rank=rank,time_step_s=dt,
                steps=rows,convergence_study_verified=False)


__all__=("compare_newmark_gauss2_mvi",)
