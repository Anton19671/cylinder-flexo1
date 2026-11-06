"""Exclusive CUF TE4 MVI runner: its own CASE, validation and physical files.

No MKR, shared variational dispatcher, or legacy solver is imported.
Only --run invokes a physical shell calculation; --validate-only does not.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np

from independent_mvi.linear_models import VariationalModelCase, IndependentVariationalModel

SCHEMA="INDEPENDENT_MVI_SEVEN_MODELS_V1"
METHOD="MVI_XY_VARIATIONAL_ITERATIONS"
INTEGRATORS=("BATHE","NEWMARK","GAUSS_IRK4")


def load_case(path: Path):
    document=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document,dict) or set(document)!={
            "schema","physics","time","rank","output_dir"}:
        raise ValueError("MVI_CASE_KEYS_INVALID")
    if document["schema"]!=SCHEMA:
        raise ValueError("MVI_REQUIRES_OWN_INDEPENDENT_CASE_SCHEMA")
    physics=VariationalModelCase(**document["physics"])
    rank=document["rank"]
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("MVI_CASE_RANK_INVALID")
    if rank>min(physics.nx-(4 if physics.theory=="KL" else 2),physics.ns):
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


def _save_npz_atomic(destination: Path, **arrays):
    """Commit a complete NPZ only after compression succeeds; never expose partial data."""
    temporary = destination.with_name(destination.stem + ".tmp.npz")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


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
    solver=IndependentVariationalModel(physics,rank)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MVI_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    shape=(solver.n,physics.nx,physics.ns)
    q=np.zeros(shape);v=np.zeros(shape);a=np.zeros(shape)
    profiles=None
    summary={
        "method":METHOD,"schema":SCHEMA,"integrator":scheme,"rank":rank,"theory":physics.theory,
        "constitutive_model":("PLANE_STRESS_SHELL" if
                              physics.theory in ("KL","FSDT","REDDY")
                              else "THREE_DIMENSIONAL_CUF"),
        "load_surface":"OUTER_REFERENCE_FACE",
        "load_sign":"POSITIVE_PRESSURE_INWARD",
        "pressure_pa":physics.pressure_pa,
        "outer_reference_area_factor":1.+physics.thickness_m/(2.*physics.radius_m),
        "status":"IN_PROGRESS","accepted_steps":0,
        "requested_steps":steps,"dt_s":dt_s,"time_s":0.,
        "solver_module":"independent_mvi.linear_models"
    }
    summary_path=output/"MVI_OWN_RUN_SUMMARY.json"
    def save_summary():
        temporary=summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        temporary.replace(summary_path)
    save_summary()
    radial=(2 if physics.theory in ("KL","FSDT","REDDY") else
            2*(int(physics.theory[-1])+1))
    pause_signal=Path(path).with_suffix(".pause")
    history=output/"MVI_OWN_PHYSICAL_HISTORY.csv"
    fields=("step","stage","time_s","central_w0_m","maximum_abs_w0_m")
    failed_step=None
    committed_steps=0
    with history.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        stream.flush()
        committed_csv_position=stream.tell()
        try:
            for k in range(1,steps+1):
                failed_step=k
                if pause_signal.is_file():
                    summary["status"]="PAUSED"
                    save_summary()
                    print(f"MVI PAUSED before step {k}/{steps}",flush=True)
                    while pause_signal.is_file():
                        time.sleep(.2)
                    summary["status"]="IN_PROGRESS"
                    save_summary()
                    print(f"MVI RESUMED step {k}/{steps}",flush=True)
                start=(k-1)*dt_s
                if scheme=="BATHE":
                    first,final,next_profiles,details=solver.step_bathe(
                        q,v,a,start,dt_s,previous=profiles)
                    stages=(("GAMMA",start+(2-np.sqrt(2))*dt_s,first),
                            ("END",k*dt_s,final))
                elif scheme=="NEWMARK":
                    final,next_profiles,details=solver.step_newmark(
                        q,v,a,start,dt_s,previous=profiles)
                    stages=(("END",k*dt_s,final),)
                else:
                    physical,final,next_profiles,details=solver.step_gauss_irk4(
                        q,v,a,start,dt_s,previous=profiles)
                    offsets=(.5-np.sqrt(3)/6,.5+np.sqrt(3)/6)
                    stages=tuple(
                        (f"GAUSS_{i+1}",start+offsets[i]*dt_s,physical[i])
                        for i in range(2))+(("END",k*dt_s,final),)
                if (not isinstance(next_profiles,(tuple,list)) or
                        len(next_profiles)!=2):
                    raise RuntimeError("MVI_MISSING_XY_FACTORS")
                X,Y=(np.asarray(z,float) for z in next_profiles)
                if (X.shape!=(solver.n,rank,physics.nx) or
                        Y.shape!=(solver.n,rank,physics.ns) or
                        not np.isfinite(X).all() or not np.isfinite(Y).all()):
                    raise RuntimeError("MVI_NONFINITE_OR_WRONG_SHAPE_FACTORS")
                pending=[]
                for stage,time_s,fields_at_stage in stages:
                    mechanical,velocity,acceleration=fields_at_stage
                    if any(z.shape!=shape or not np.isfinite(z).all()
                           for z in (mechanical,velocity,acceleration)):
                        raise RuntimeError("MVI_NONFINITE_OR_WRONG_SHAPE_STAGE")
                    pending.append((
                        output/f"MVI_R{rank}_STEP_{k:04d}_{stage}.npz",
                        time_s,mechanical,velocity,acceleration,
                        {"step":k,"stage":stage,"time_s":time_s,
                         "central_w0_m":float(mechanical[
                             radial,physics.nx//2,physics.ns//2]),
                         "maximum_abs_w0_m":float(np.max(np.abs(
                             mechanical[radial])))}
                    ))
                for filename,stage_time,mechanical,velocity,acceleration,row in pending:
                    _save_npz_atomic(filename,time_s=stage_time,
                            mechanical=mechanical,velocity=velocity,
                            acceleration=acceleration)
                _save_npz_atomic(output/f"MVI_R{rank}_STEP_{k:04d}_FACTORS.npz",
                                    x_profiles=X,y_profiles=Y)
                for *_,row in pending:
                    writer.writerow(row)
                stream.flush()
                q,v,a=tuple(np.array(z,copy=True) for z in final)
                profiles=(X.copy(),Y.copy())
                summary["accepted_steps"]=k
                summary["time_s"]=k*dt_s
                save_summary()
                committed_steps=k
                committed_csv_position=stream.tell()
                print(f"MVI ACCEPTED {k}/{steps} t={k*dt_s:.9g} s",flush=True)
            summary["status"]="CONVERGED"
            save_summary()
            print(f"MVI COMPLETED {steps}/{steps}",flush=True)
        except Exception as exc:
            # The summary is the commit marker. Discard any partially written
            # stages and CSV lines belonging to the uncommitted time step.
            if failed_step is not None and failed_step > committed_steps:
                for orphan in output.glob(f"MVI_R{rank}_STEP_{failed_step:04d}_*.npz"):
                    orphan.unlink(missing_ok=True)
                stream.seek(committed_csv_position)
                stream.truncate()
                stream.flush()
            summary["accepted_steps"]=committed_steps
            summary["time_s"]=committed_steps*dt_s
            summary.update(status="FAILED",failed_step=failed_step,
                           error_type=type(exc).__name__,detail=str(exc))
            save_summary()
            print(f"MVI FAILED step={failed_step} error={type(exc).__name__}: {exc}",
                  flush=True)
            raise
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
