"""Exclusive CUF TE4 MVI runner: its own CASE, validation and physical files.

No MKR, shared variational dispatcher, or legacy solver is imported.
Only --run invokes a physical shell calculation; --validate-only does not.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from independent_mvi.linear_te4 import VariationalCase, VariationalTE4

SCHEMA="INDEPENDENT_MVI_CUF_TE4_V1"
METHOD="MVI_XY_VARIATIONAL_ITERATIONS"
INTEGRATORS=("BATHE","NEWMARK","GAUSS_IRK4")


def load_case(path: Path):
    document=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document,dict) or set(document)!={
            "schema","physics","time","rank","output_dir"}:
        raise ValueError("MVI_CASE_KEYS_INVALID")
    if document["schema"]!=SCHEMA:
        raise ValueError("MVI_REQUIRES_OWN_INDEPENDENT_CASE_SCHEMA")
    physics=VariationalCase(**document["physics"])
    rank=document["rank"]
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("MVI_CASE_RANK_INVALID")
    if rank>min(physics.nx-2,physics.ns):
        raise ValueError("MVI_CASE_RANK_EXCEEDS_CLAMPED_CAPACITY")
    timing=document["time"]
    if not isinstance(timing,dict) or set(timing)!={
            "integrator","dt_s","steps"}:
        raise ValueError("MVI_TIME_KEYS_INVALID")
    scheme=timing["integrator"]
    if scheme not in INTEGRATORS:
        raise ValueError("MVI_UNKNOWN_INDEPENDENT_TIME_SCHEME")
    dt_s=float(timing["dt_s"])
    steps=timing["steps"]
    if (not np.isfinite(dt_s) or not 0<dt_s<=.1 or
            type(steps) is not int or not 1<=steps<=120):
        raise ValueError("MVI_INDEPENDENT_TIME_GRID_INVALID")
    output=Path(document["output_dir"])
    if str(output) in (".","/") or not str(output).strip():
        raise ValueError("MVI_OUTPUT_DIRECTORY_REQUIRED")
    return physics,rank,scheme,dt_s,steps,output,document


def run(path: Path, *, validate_only: bool):
    physics,rank,scheme,dt_s,steps,output,document=load_case(path)
    if validate_only:
        return {
            "method":METHOD,"schema":SCHEMA,"rank":rank,
            "integrator":scheme,"validated_only":True,
            "steps":steps,"dt_s":dt_s,"output_dir":str(output),
            "physics":asdict(physics)
        }

    # Instantiate ONLY the independent MVI. A separate, genuinely rank
    # constrained spatial X/Y solve occurs inside every physical time stage.
    solver=VariationalTE4(physics,rank)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MVI_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    shape=(15,physics.nx,physics.ns)
    q=np.zeros(shape);v=np.zeros(shape);a=np.zeros(shape)
    profiles=None
    summary={
        "method":METHOD,"schema":SCHEMA,"integrator":scheme,"rank":rank,
        "status":"IN_PROGRESS","accepted_steps":0,
        "requested_steps":steps,"dt_s":dt_s,"time_s":0.,
        "solver_module":"independent_mvi.linear_te4"
    }
    def save_summary():
        (output/"MVI_OWN_RUN_SUMMARY.json").write_text(
            json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    save_summary()
    table=[]
    for k in range(1,steps+1):
        start=(k-1)*dt_s
        if scheme=="BATHE":
            first,final,profiles,details=solver.step_bathe(
                q,v,a,start,dt_s,previous_profiles=profiles)
            stages=(("GAMMA",start+(2-np.sqrt(2))*dt_s,first),
                    ("END",k*dt_s,final))
        elif scheme=="NEWMARK":
            final,profiles,details=solver.step_newmark(
                q,v,a,start,dt_s,previous_profiles=profiles)
            stages=(("END",k*dt_s,final),)
        else:
            physical,final,profiles,details=solver.step_runge_kutta_gauss4(
                q,v,a,start,dt_s,previous_profiles=profiles)
            offsets=(.5-np.sqrt(3)/6,.5+np.sqrt(3)/6)
            stages=tuple((f"GAUSS_{i+1}",start+offsets[i]*dt_s,physical[i])
                         for i in range(2))+(("END",k*dt_s,final),)
        for stage,time_s,(mechanical,velocity,acceleration) in stages:
            if any(field.shape!=shape or not np.isfinite(field).all()
                   for field in (mechanical,velocity,acceleration)):
                raise RuntimeError("MVI_NONFINITE_OR_INVALID_ACCEPTED_FIELD")
            name=output/f"MVI_R{rank}_STEP_{k:04d}_{stage}.npz"
            np.savez_compressed(
                name,time_s=time_s,mechanical=mechanical,
                velocity=velocity,acceleration=acceleration)
            table.append({
                "step":k,"stage":stage,"time_s":time_s,
                "central_w0_m":float(mechanical[10,physics.nx//2,physics.ns//2]),
                "maximum_abs_w0_m":float(np.max(np.abs(mechanical[10])))
            })
        q,v,a=tuple(np.array(x,copy=True) for x in final)
        np.savez_compressed(output/f"MVI_R{rank}_STEP_{k:04d}_FACTORS.npz",
                            x_profiles=profiles[0],y_profiles=profiles[1])
        summary["accepted_steps"]=k
        summary["time_s"]=k*dt_s
        save_summary()
    with (output/"MVI_OWN_PHYSICAL_HISTORY.csv").open(
            "w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    summary["status"]="CONVERGED"
    save_summary()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",type=Path,required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--validate-only",action="store_true")
    choice.add_argument("--run",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.case,validate_only=args.validate_only),indent=2))


if __name__=="__main__":
    main()
