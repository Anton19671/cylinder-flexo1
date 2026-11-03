#!/usr/bin/env python3
"""Independent postprocessing of the nine physically solved KL C-C Newmark runs.

This program ONLY audits saved physical states. No new shell calculation, no
FDM substitution, no forced acceptance, and no manual hand-entered results.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/"config/verification/kl_cc_mvi_rank_matrix"
KINDS=("mechanical","velocity","acceleration")
PHYSICS=("solver_backend","calculation_kind","scale_model","theory","mode",
         "geometry_model","material_model","units","geometry","material",
         "quadrature","boundary_conditions","load_definition","reference_newton",
         "postprocess","dynamics")


def read_case(rank):
    label="NEWMARK_FDM" if rank==0 else f"NEWMARK_MVI{rank}"
    folder=BASE/"output"/label
    case=json.loads((BASE/(label+".json")).read_text(encoding="utf-8"))
    summary=json.loads((folder/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    method="FDM" if rank==0 else "MVI"
    if case["method"]!=method or summary.get("method")!=method:
        raise ValueError(label+": SOLVER_METHOD_MISMATCH")
    if rank and (case.get("rank")!=rank or summary.get("rank")!=rank):
        raise ValueError(label+": FIXED_RANK_MISMATCH")
    if not summary.get("converged") or summary.get("status")!="CONVERGED":
        raise ValueError(label+": SOLVER_NOT_CONVERGED")
    if summary.get("accepted_steps")!=5 or summary.get("requested_steps")!=5:
        raise ValueError(label+": NOT_FIVE_ACCEPTED_STEPS")
    if len(summary.get("steps",[]))!=5:
        raise ValueError(label+": MISSING_STEP")
    grid=case.get("mvi_grid",case.get("fdm_grid"))
    if grid!={"nx":9,"ns":9,"operator_scheme":"SBP42_FOURIER_ODD"}:
        raise ValueError(label+": SPATIAL_GRID_MISMATCH")
    records=[]
    for i,step in enumerate(summary["steps"],1):
        if (step.get("index")!=i or step.get("converged") is not True or
            step.get("status") not in ("CONVERGED","CONVERGED_FULL",
                                        "CONVERGED_PROJECTED")):
            raise ValueError(label+": STEP_NOT_ACCEPTED")
        state=step.get("state_file","")
        if state!=f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz":
            raise ValueError(label+": INVALID_STATE_FILE")
        with np.load(folder/state,allow_pickle=False) as npz:
            data={key:np.array(npz[key],dtype=float) for key in KINDS}
            t=float(npz["time_s"])
            load=float(npz["load_factor"])
            if rank:
                x=np.array(npz["x_profiles"],dtype=float)
                y=np.array(npz["y_profiles"],dtype=float)
                scales=np.array(npz["state_scales"],dtype=float)
                if x.shape!=(3,rank,9) or y.shape!=(3,rank,9):
                    raise ValueError(label+": WRONG_MVI_PROFILE_SHAPE")
                reconstructed=np.einsum("crx,crs->cxs",x,y)*scales
                error=float(np.max(abs(reconstructed-data["mechanical"])))
                if error>1.e-12:
                    raise ValueError(label+": PROFILE_RECONSTRUCTION_FAILED")
            else:error=0.
        if abs(t-.002*i)>1.e-12 or abs(load-math.sin(50*math.pi*t))>1.e-12:
            raise ValueError(label+": TIME_OR_LOAD_MISMATCH")
        if any(d.shape!=(3,9,9) or not np.isfinite(d).all() for d in data.values()):
            raise ValueError(label+": WRONG_SHAPE_OR_NONFINITE")
        if (np.max(abs(data["mechanical"][:,[0,-1],:]))>1.e-9 or
            np.max(abs(data["velocity"][:,[0,-1],:]))>1.e-8 or
            np.max(abs(data["acceleration"][:,[0,-1],:]))>1.e-7):
            raise ValueError(label+": CLAMPED_TRACE_FAILURE")
        data["t"]=t
        data["reconstruction_error"]=error
        records.append(data)
    return case,records


def main():
    result={rank:read_case(rank) for rank in range(9)}
    fingerprint={k:result[0][0][k] for k in PHYSICS}
    for rank,(case,_) in result.items():
        if {k:case[k] for k in PHYSICS}!=fingerprint:
            raise ValueError(f"RANK{rank}_PHYSICAL_CASE_NOT_IDENTICAL")
    rows=[]
    for rank,(case,states) in result.items():
        differences={}
        for field in KINDS:
            rel=[]
            for s,ref in zip(states,result[0][1]):
                scale=float(np.linalg.norm(ref[field]))
                delta=float(np.linalg.norm(s[field]-ref[field]))
                rel.append(100*delta/scale if scale else (0. if delta==0. else math.inf))
            differences[field]=max(rel)
        final=states[-1]["mechanical"][2]
        rows.append({"method":"FDM" if rank==0 else f"MVI-{rank}",
                     "rank":rank,
                     "accepted_steps":5,
                     "final_time_s":states[-1]["t"],
                     "final_max_abs_w_m":float(np.max(abs(final))),
                     "final_mean_midspan_w_m":float(np.mean(final[4])),
                     "max_displacement_difference_percent":differences["mechanical"],
                     "max_velocity_difference_percent":differences["velocity"],
                     "max_acceleration_difference_percent":differences["acceleration"],
                     "max_x_y_reconstruction_error_m":max(s["reconstruction_error"] for s in states)})
    for row in rows[1:]:
        if (row["max_displacement_difference_percent"]>1.e-4 or
            row["max_velocity_difference_percent"]>1.e-3 or
            row["max_acceleration_difference_percent"]>1.e-3):
            raise ValueError(row["method"]+": MVI_AND_FDM_FIELDS_DISAGREE")
    with (BASE/"NEWMARK_MVI1_8_FIVE_STEP_COMPARISON.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest={"reference_workflow":"https://github.com/Anton19671/cylinder-flexo1/actions/runs/41624708013",
              "physics_identical":True,"accepted_methods":9,
              "accepted_time_steps":45,"fixed_rank_profiles_verified":True,
              "full_fields_uvw_velocity_acceleration_verified":True,
              "clamped_traces_verified":True,
              "time_step_convergence_verified":False,
              "nonseparable_rank_convergence_verified":False,
              "results":rows}
    (BASE/"NEWMARK_MVI1_8_FIVE_STEP_COMPARISON.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("KL_CC_REAL_DYNAMIC_RANK_MATRIX_ACCEPTED 9/9 methods, 45/45 physical steps")
    for row in rows:
        print(row)


if __name__=="__main__":
    main()
