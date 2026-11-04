"""Method-specific executable for the isolated CUF TE4 finite-difference solver.

This command does not import, dispatch, read, or call the MVI solver.
Use --validate-only for preflight; --run must be requested explicitly.
Physical calculations are not performed during import or validation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from independent_mkr.linear_te4 import PhysicalCase, StrongDifferenceTE4

SCHEMA = "INDEPENDENT_MKR_CUF_TE4_V1"
METHOD = "MKR_STRONG_FINITE_DIFFERENCES"


def load_case(path: Path):
    document=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document,dict) or set(document)!={"schema","physics","time","output_dir"}:
        raise ValueError("MKR_CASE_KEYS_INVALID")
    if document["schema"]!=SCHEMA:
        raise ValueError("MKR_REQUIRES_OWN_INDEPENDENT_CASE_SCHEMA")
    raw=document["physics"]
    if not isinstance(raw,dict):
        raise ValueError("MKR_PHYSICS_REQUIRES_OBJECT")
    physics=PhysicalCase(**raw)
    timing=document["time"]
    if not isinstance(timing,dict) or set(timing)!={"dt_s","steps"}:
        raise ValueError("MKR_TIME_KEYS_INVALID")
    dt_s=float(timing["dt_s"])
    steps=timing["steps"]
    if (not np.isfinite(dt_s) or not 0<dt_s<=.1 or
            type(steps) is not int or not 1<=steps<=120):
        raise ValueError("MKR_INDEPENDENT_BATHE_TIME_INVALID")
    output=Path(document["output_dir"])
    if not str(output).strip() or str(output) in (".","/"):
        raise ValueError("MKR_OUTPUT_DIRECTORY_REQUIRED")
    return physics,dt_s,steps,output,document


def run(path: Path, *, validate_only: bool):
    physics,dt_s,steps,output,document=load_case(path)
    if validate_only:
        return {
            "schema":SCHEMA,"method":METHOD,
            "validated_only":True,"steps":steps,"dt_s":dt_s,
            "output_dir":str(output),"physics":asdict(physics)
        }

    # Only the MKR/FDM implementation is constructed in this route.
    solver=StrongDifferenceTE4(physics)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MKR_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    shape=(15,physics.nx,physics.ns)
    q=np.zeros(shape,float);v=np.zeros(shape,float);a=np.zeros(shape,float)
    summary={
        "method":METHOD,"schema":SCHEMA,"status":"IN_PROGRESS",
        "accepted_steps":0,"requested_steps":steps,"dt_s":dt_s,
        "time_s":0.,"solver_module":"independent_mkr.linear_te4"
    }
    def save_summary():
        (output/"MKR_OWN_RUN_SUMMARY.json").write_text(
            json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    save_summary()
    table=[]
    for k in range(1,steps+1):
        t=(k-1)*dt_s
        # The solver must return *two* accepted physical Bathe stages.
        (qg,vg,ag),(qn,vn,an)=solver.step_bathe(q,v,a,t,dt_s)
        gamma=2-np.sqrt(2.)
        # Independent filenames; never load or write any MVI output.
        for stage,time_s,fields in (
            ("GAMMA",t+gamma*dt_s,(qg,vg,ag)),
            ("END",k*dt_s,(qn,vn,an))):
            pos=output/f"MKR_STEP_{k:04d}_{stage}.npz"
            np.savez_compressed(pos,time_s=time_s,mechanical=fields[0],
                                velocity=fields[1],acceleration=fields[2])
            table.append({
                "step":k,"stage":stage,"time_s":time_s,
                "central_w0_m":float(fields[0][10,physics.nx//2,physics.ns//2]),
                "maximum_abs_w0_m":float(np.max(np.abs(fields[0][10])))
            })
        q,v,a=qn.copy(),vn.copy(),an.copy()
        summary["accepted_steps"]=k
        summary["time_s"]=k*dt_s
        save_summary()
    with (output/"MKR_OWN_PHYSICAL_HISTORY.csv").open(
            "w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    summary["status"]="CONVERGED"
    save_summary()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    result=run(args.case,validate_only=args.validate_only)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
