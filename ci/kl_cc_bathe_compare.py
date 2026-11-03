#!/usr/bin/env python3
"""Reproducible independent verification of all nine *real* Bathe shell jobs.

Reads physical artifacts; does not run FDM for MVI, relax stage acceptance,
change the CASE, extrapolate time, or synthesize uncomputed field histories.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/"config/verification/kl_cc_bathe"
GAMMA=2.-math.sqrt(2.)
PHYSICS_FIELDS=("solver_backend","theory","calculation_kind","scale_model","mode",
 "geometry_model","material_model","units","geometry","material","quadrature",
 "boundary_conditions","load_definition","reference_newton","postprocess","dynamics")
FIELD_NAMES=("mechanical","velocity","acceleration")


def read_rank(rank):
    name="BATHE_FDM" if rank==0 else f"BATHE_MVI{rank}"
    folder=BASE/"output"/name
    case=json.loads((BASE/(name+".json")).read_text(encoding="utf-8"))
    summary=json.loads((folder/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    method="FDM" if rank==0 else "MVI"
    if case["method"]!=summary.get("method") or case["method"]!=method:
        raise ValueError(name+": WRONG_PHYSICAL_SOLVER")
    if summary.get("integrator")!="BATHE" or summary.get("integrator_order")!=2:
        raise ValueError(name+": NOT_BATHE_ORDER_TWO")
    if summary.get("rank")!=(rank if rank else None):
        raise ValueError(name+": FIXED_RANK_MISMATCH")
    if (summary.get("status")!="CONVERGED" or summary.get("converged") is not True
        or summary.get("accepted_steps")!=2 or summary.get("requested_steps")!=2
        or len(summary.get("steps",[]))!=2):
        raise ValueError(name+": INCOMPLETE_TWO_PHYSICAL_STEPS")
    grid=case.get("mvi_grid",case.get("fdm_grid"))
    if grid!={"nx":9,"ns":9,"operator_scheme":"SBP42_FOURIER_ODD"}:
        raise ValueError(name+": SPATIAL_GRID_MISMATCH")
    if case["dynamics"].get("integrator")!="BATHE":
        raise ValueError(name+": CASE_NOT_EXPLICIT_BATHE")
    records=[]
    for step_i,step in enumerate(summary["steps"],1):
        if (step["index"]!=step_i or not step["converged"]
            or step["status"] not in ("CONVERGED","CONVERGED_FULL","CONVERGED_PROJECTED")
            or step["first_substep_status"] not in
                     ("CONVERGED","CONVERGED_FULL","CONVERGED_PROJECTED")):
            raise ValueError(name+": REJECTED_SUBSTEP")
        for stage in ("FIRST","FINAL"):
            state_file=(f"BATHE_SUBSTEP_{step_i:04d}_GAMMA.npz" if stage=="FIRST"
                        else f"REFERENCE_DYNAMIC_STEP_{step_i:04d}.npz")
            if not (folder/state_file).is_file():
                raise ValueError(name+": MISSING_"+stage+"_FIELD")
            with np.load(folder/state_file,allow_pickle=False) as data:
                fields={key:np.array(data[key],float) for key in FIELD_NAMES}
                t=float(data["time_s"])
                load=float(data["load_factor"])
                if rank:
                    X=np.asarray(data["x_profiles"],float)
                    Y=np.asarray(data["y_profiles"],float)
                    scale=np.asarray(data["state_scales"],float)
                    if X.shape!=(3,rank,9) or Y.shape!=(3,rank,9):
                        raise ValueError(name+": INVALID_XY_RANK")
                    recon=np.einsum("crx,crs->cxs",X,Y)*scale
                    recon_err=float(np.max(abs(recon-fields["mechanical"])))
                    if recon_err>1.e-11:
                        raise ValueError(name+": FIXED_RANK_RECONSTRUCTION_DRIFT")
                else:
                    recon_err=0.
            required=(step_i-1+GAMMA)*.002 if stage=="FIRST" else step_i*.002
            if abs(t-required)>1.e-12:
                raise ValueError(name+": WRONG_ABSOLUTE_STAGE_TIME")
            if abs(load-math.sin(2*math.pi*25*t))>1.e-12:
                raise ValueError(name+": WRONG_PHYSICAL_STAGE_LOAD")
            if any(v.shape!=(3,9,9) or not np.isfinite(v).all()
                   for v in fields.values()):
                raise ValueError(name+": INVALID_FIELD_ARRAY")
            if (float(np.max(abs(fields["mechanical"][:,[0,-1],:])))>1.e-9
                or float(np.max(abs(fields["velocity"][:,[0,-1],:])))>1.e-8
                or float(np.max(abs(fields["acceleration"][:,[0,-1],:])))>1.e-7):
                raise ValueError(name+": CLAMPED_SUPPORT_DRIFT")
            fields.update(step=step_i,stage=stage,t=t,load=load,
                          reconstruction_error_m=recon_err)
            records.append(fields)
    return case,summary,records


def main():
    got={rank:read_rank(rank) for rank in range(9)}
    first={k:got[0][0][k] for k in PHYSICS_FIELDS}
    for rank,(case,_,_) in got.items():
        if {k:case[k] for k in PHYSICS_FIELDS}!=first:
            raise ValueError("BATHE_PHYSICAL_CASE_MISMATCH_"+str(rank))
    rows=[]
    reference=got[0][2]
    for rank,(case,summary,records) in got.items():
        errors={}
        for field in FIELD_NAMES:
            differences=[]
            for this,ref in zip(records,reference):
                assert this["step"]==ref["step"] and this["stage"]==ref["stage"]
                denominator=float(np.linalg.norm(ref[field]))
                absolute=float(np.linalg.norm(this[field]-ref[field]))
                differences.append(100.*absolute/denominator if denominator>0
                                   else (0. if absolute==0 else math.inf))
            errors[field]=max(differences)
        final=records[-1]["mechanical"][2]
        rows.append(dict(method=("FDM" if rank==0 else f"MVI-{rank}"),
            rank=rank,accepted_steps=2,accepted_physical_substeps=4,
            final_time_s=records[-1]["t"],final_mean_midspan_w_m=float(np.mean(final[4])),
            final_max_abs_w_m=float(np.max(abs(final))),
            max_displacement_difference_percent=errors["mechanical"],
            max_velocity_difference_percent=errors["velocity"],
            max_acceleration_difference_percent=errors["acceleration"],
            max_xy_reconstruction_error_m=max(r["reconstruction_error_m"]
                                              for r in records)))
        if rank and (errors["mechanical"]>1.e-3 or errors["velocity"]>1.e-2
                     or errors["acceleration"]>1.e-2):
            raise ValueError("BATHE_RANK"+str(rank)+"_PHYSICAL_FIELD_DIFFERS_FROM_FDM")
    output=BASE/"BATHE_MVI1_8_VS_FDM_REAL_PHYSICAL_COMPARISON.csv"
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    manifest=dict(source_physical_run=41637040470,integrator="BATHE",
        spatial_model="KL",method_count=9,accepted_full_steps=18,
        accepted_physical_substeps=36,common_physics=True,
        trace_and_load_verified=True,fixed_rank_xy_verified=True,
        time_step_convergence_verified=False,
        nonseparable_rank_convergence_verified=False,results=rows)
    (BASE/"BATHE_MVI1_8_VS_FDM_REAL_PHYSICAL_COMPARISON.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BATHE_9_METHODS_36_REAL_VARIATIONAL_SUBSTEPS_VERIFIED")
    for row in rows:print(row)


if __name__=="__main__":
    main()
