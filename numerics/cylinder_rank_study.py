#!/usr/bin/env python3
"""Auditable MVI rank qualification against a computed FDM Newmark baseline.

This procedure selects a numerical *candidate on one mesh/time grid*, never a
mathematically exact reference. It requires independent completed runs and
checks the full admissible residual, boundary defect and physical field errors
at EVERY time step. An XY projected residual alone never qualifies a rank.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from cylinder_dynamic_comparison import compare_dynamic_runs

REQUIRED_FIELDS=("radial_displacement","mechanical","velocity","acceleration")
OPTIONAL_FIELDS=("electrical","damping_force","constraint_reaction")


def _threshold(value, name):
    if isinstance(value,bool):
        raise ValueError(f"{name}: finite nonnegative number required")
    try: x=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"{name}: finite nonnegative number required") from exc
    if not math.isfinite(x) or x<0:
        raise ValueError(f"{name}: finite nonnegative number required")
    return x


def assess_rank_series(fdm_folder, mvi_runs, *, field_tolerances,
                       max_full_residual_scaled, max_boundary_defect_scaled,
                       require_adjacent_rank=True):
    """Assess ranks 1..8, never inferring results for missing ranks."""
    if not isinstance(field_tolerances,dict):
        raise ValueError("explicit physical-field tolerance dictionary required")
    allowed=set(REQUIRED_FIELDS+OPTIONAL_FIELDS)
    if set(REQUIRED_FIELDS)-set(field_tolerances) or set(field_tolerances)-allowed:
        raise ValueError("explicit tolerances for radial, mechanical, velocity, acceleration required")
    tolerances={}
    for field,bounds in field_tolerances.items():
        if not isinstance(bounds,dict) or set(bounds)!={"abs","rel"}:
            raise ValueError(field+": require abs/rel tolerance pair")
        aa=_threshold(bounds["abs"],field+".abs")
        rr=_threshold(bounds["rel"],field+".rel")
        if aa==0 and rr==0:raise ValueError(field+": zero tolerance is not defined")
        tolerances[field]={"abs":aa,"rel":rr}
    residual_limit=_threshold(max_full_residual_scaled,"max_full_residual_scaled")
    bc_limit=_threshold(max_boundary_defect_scaled,"max_boundary_defect_scaled")
    if type(require_adjacent_rank) is not bool:
        raise ValueError("require_adjacent_rank must be boolean")
    if not isinstance(mvi_runs,dict) or any(type(k) is not int or k<1 or k>8 for k in mvi_runs):
        raise ValueError("MVI ranks must be mapped by distinct integers 1..8")
    results=[]
    passed=set()
    attempted=set(mvi_runs)
    for rank in range(1,9):
        row={"rank":rank,"status":"NOT_TESTED","qualified":False}
        if rank not in mvi_runs:
            results.append(row)
            continue
        row["folder"]=str(mvi_runs[rank])
        try:
            comparison=compare_dynamic_runs(mvi_runs[rank],fdm_folder)
            if comparison.get("mvi_rank")!=rank:
                raise ValueError("MVI rank/summary mismatch")
            steps=comparison["steps"]
            if not steps:raise ValueError("no calculated steps")
            failures=[]
            metrics={}
            for field,limit in tolerances.items():
                if any(field not in step["field_errors"] for step in steps):
                    failures.append(field+": missing from one or more computed steps")
                    continue
                errors=[step["field_errors"][field] for step in steps]
                max_error=max(float(e["max_abs_difference"]) for e in errors)
                max_excess=max(float(e["max_abs_difference"])-
                    limit["abs"]-limit["rel"]*float(e["fdm_peak_abs"]) for e in errors)
                if not math.isfinite(max_excess):raise ValueError("nonfinite computed field error")
                metrics[field]={"max_abs_difference":max_error,
                                "max_tolerance_excess":max_excess}
                if max_excess>0:failures.append(field+": exceeds abs+rel*FDM field")
            mf=max(float(x["mvi_admissible_residual_scaled"]) for x in steps)
            ff=max(float(x["fdm_admissible_residual_scaled"]) for x in steps)
            mb=max(float(x["mvi_constraint_defect_scaled"]) for x in steps)
            fb=max(float(x["fdm_constraint_defect_scaled"]) for x in steps)
            if not all(math.isfinite(v) and v>=0 for v in (mf,ff,mb,fb)):
                raise ValueError("invalid residual or boundary defect")
            if mf>residual_limit or ff>residual_limit:
                failures.append("full admissible residual above limit")
            if mb>bc_limit or fb>bc_limit:
                failures.append("boundary defect above limit")
            row.update(steps_compared=len(steps),fields=metrics,
                       max_mvi_full_residual_scaled=mf,
                       max_fdm_full_residual_scaled=ff,
                       max_mvi_boundary_defect_scaled=mb,
                       max_fdm_boundary_defect_scaled=fb,
                       failures=failures,qualified=not failures,
                       status="QUALIFIED_ON_THIS_GRID" if not failures else "NOT_QUALIFIED")
            if not failures:passed.add(rank)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            row.update(status="RUN_INCOMPLETE_OR_INVALID",reason=str(exc))
        results.append(row)
    candidate=None
    for rank in range(1,9):
        if rank not in passed:continue
        if any(smaller not in attempted for smaller in range(1,rank)):continue
        if require_adjacent_rank and rank+1 not in passed:continue
        candidate=rank
        break
    return {"comparison":"MVI ranks versus independently computed FDM",
            "fdm_folder":str(fdm_folder),"field_tolerances":tolerances,
            "max_full_residual_scaled":residual_limit,
            "max_boundary_defect_scaled":bc_limit,
            "ranks":results,"candidate_rank":candidate,
            "require_adjacent_rank":require_adjacent_rank,
            "certified_reference":False,
            "spatial_convergence_verified":False,
            "temporal_convergence_verified":False,
            "recommendation":("Conditional rank candidate, verify mesh/time refinement" if
                              candidate is not None else
                              "No supported MVI reference rank on this grid")}


def save_rank_report(report,path):
    target=Path(path).expanduser()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
                      encoding="utf-8")
    return target


__all__=("assess_rank_series","save_rank_report")
