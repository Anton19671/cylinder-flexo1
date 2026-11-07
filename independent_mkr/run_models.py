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
import time
from pathlib import Path

import numpy as np

from independent_mkr.linear_models import StrongModelCase, IndependentStrongModel

SCHEMA = "INDEPENDENT_MKR_SEVEN_MODELS_V1"
METHOD = "MKR_STRONG_FINITE_DIFFERENCES"
INTEGRATORS = ("BATHE","NEWMARK","GAUSS_IRK4")


SPATIAL_GRID="AXIAL_CIRCUMFERENTIAL_2D"
THICKNESS_SCHEME="ANALYTIC_KINEMATICS_GAUSS12_NO_Z_GRID"
PHYSICS_SCOPE="LINEAR_MACRO_HOOKE_ELECTRIC_OFF_GRADIENT_OFF"


def _central_grid_value(field):
    """Sample x=L/2, theta=pi with linear interpolation on the grid."""
    field=np.asarray(field)
    if field.ndim!=2 or min(field.shape)<1:
        raise ValueError("CENTRAL_SAMPLE_REQUIRES_2D_FIELD")
    nx,ns=field.shape
    ix=(nx-1)/2.
    js=ns/2.
    i0=int(np.floor(ix)); i1=int(np.ceil(ix))
    j0=int(np.floor(js)); j1=int(np.ceil(js))%ns
    tx=ix-i0; ts=js-int(np.floor(js))
    a=(1.-tx)*field[i0,j0]+tx*field[i1,j0]
    b=(1.-tx)*field[i0,j1]+tx*field[i1,j1]
    return float((1.-ts)*a+ts*b)


def _outer_radial_field(mechanical, theory):
    """Recover w(x,theta,z=+h/2), without a third spatial grid.

    For the normalized CUF thickness basis F_k(+h/2)=1. The leading
    radial coefficient w_0 is NOT generally the physical outer-face
    displacement for TE1-TE4. Classical shell models have only w_0.
    This operation is local to the MKR export path.
    """
    if theory in ("KL","FSDT","REDDY"):
        return mechanical[2]
    if theory not in ("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
        raise ValueError("MKR_UNSUPPORTED_OUTER_FACE_THEORY")
    order=int(theory[-1])+1
    return np.sum(mechanical[2*order:3*order],axis=0)


def load_case(path: Path):
    document=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document,dict) or set(document)!={"schema","physics","time","output_dir"}:
        raise ValueError("MKR_CASE_KEYS_INVALID")
    if document["schema"]!=SCHEMA:
        raise ValueError("MKR_REQUIRES_OWN_INDEPENDENT_CASE_SCHEMA")
    raw=document["physics"]
    if not isinstance(raw,dict):
        raise ValueError("MKR_PHYSICS_REQUIRES_OBJECT")
    physics=StrongModelCase(**raw)
    timing=document["time"]
    if not isinstance(timing,dict) or set(timing)!={"integrator","dt_s","steps"}:
        raise ValueError("MKR_TIME_KEYS_INVALID")
    scheme=timing["integrator"]
    if scheme not in INTEGRATORS:
        raise ValueError("MKR_UNKNOWN_INDEPENDENT_TIME_SCHEME")
    dt_s=float(timing["dt_s"])
    steps=timing["steps"]
    if (not np.isfinite(dt_s) or not 0<dt_s<=.1 or
            type(steps) is not int or not 1<=steps<=120):
        raise ValueError("MKR_INDEPENDENT_BATHE_TIME_INVALID")
    output=Path(document["output_dir"])
    if not str(output).strip() or str(output) in (".","/"):
        raise ValueError("MKR_OUTPUT_DIRECTORY_REQUIRED")
    return physics,scheme,dt_s,steps,output,document


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
    physics,scheme,dt_s,steps,output,document=load_case(path)
    if validate_only:
        return {
            "schema":SCHEMA,"method":METHOD,
            "validated_only":True,"integrator":scheme,"steps":steps,"dt_s":dt_s,
            "output_dir":str(output),"physics":asdict(physics),
            "spatial_grid":SPATIAL_GRID,
            "thickness_scheme":THICKNESS_SCHEME,
            "physics_scope":PHYSICS_SCOPE
        }

    # Only the MKR/FDM implementation is constructed in this route.
    solver=IndependentStrongModel(physics)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MKR_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    shape=(solver.n,physics.nx,physics.ns)
    q=np.zeros(shape,float);v=np.zeros(shape,float);a=np.zeros(shape,float)
    summary={
        "method":METHOD,"schema":SCHEMA,"integrator":scheme,"theory":physics.theory,
        "constitutive_model":("PLANE_STRESS_SHELL" if
                              physics.theory in ("KL","FSDT","REDDY")
                              else "THREE_DIMENSIONAL_CUF"),
        "load_surface":physics.pressure_face+"_REFERENCE_FACE",
        "load_sign":"POSITIVE_PRESSURE_INWARD",
        "pressure_pa":physics.pressure_pa,
        "outer_reference_area_factor":1.+physics.thickness_m/(2.*physics.radius_m),
        "applied_reference_area_factor":(
            1.+physics.thickness_m/(2.*physics.radius_m)
            if physics.pressure_face=="OUTER" else 1.),
        "spatial_grid":SPATIAL_GRID,
        "thickness_scheme":THICKNESS_SCHEME,
        "physics_scope":PHYSICS_SCOPE,
        "status":"IN_PROGRESS","converged":False,
        "accepted_steps":0,"requested_steps":steps,"dt_s":dt_s,
        "time_s":0.,"solver_module":"independent_mkr.linear_models"
    }
    summary_path=output/"MKR_OWN_RUN_SUMMARY.json"
    def save_summary():
        partial=summary_path.with_suffix(".tmp")
        partial.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        partial.replace(summary_path)
    save_summary()
    radial=(2 if physics.theory in ("KL","FSDT","REDDY") else
            2*(int(physics.theory[-1])+1))
    # Method-owned cooperative pause signal: one isolated CASE, one process.
    # Pause takes effect BETWEEN accepted steps, never in the middle of Newton.
    pause_signal=Path(path).with_suffix(".pause")
    history_path=output/"MKR_OWN_PHYSICAL_HISTORY.csv"
    fields=("step","stage","time_s","central_w0_m",
            "maximum_abs_w0_m","central_outer_w_m",
            "maximum_abs_outer_w_m")
    failed_step=None
    committed_steps=0
    with history_path.open("x",newline="",encoding="utf-8") as stream:
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
                    print(f"MKR PAUSED before step {k}/{steps}",flush=True)
                    while pause_signal.is_file():
                        time.sleep(.2)
                    summary["status"]="IN_PROGRESS"
                    save_summary()
                    print(f"MKR RESUMED step {k}/{steps}",flush=True)
                t=(k-1)*dt_s
                if scheme=="BATHE":
                    first,final=solver.step_bathe(q,v,a,t,dt_s)
                    gamma=2-np.sqrt(2.)
                    stages=(("GAMMA",t+gamma*dt_s,first),
                            ("END",k*dt_s,final))
                elif scheme=="NEWMARK":
                    final=solver.step_newmark(q,v,a,t,dt_s)
                    stages=(("END",k*dt_s,final),)
                else:
                    gauss,final=solver.step_gauss_irk4(q,v,a,t,dt_s)
                    offsets=(.5-np.sqrt(3)/6,.5+np.sqrt(3)/6)
                    stages=tuple((f"GAUSS_{i+1}",t+offsets[i]*dt_s,gauss[i])
                                 for i in range(2))+(("END",k*dt_s,final),)
                pending=[]
                for stage,time_s,fields_at_stage in stages:
                    mechanical,velocity,acceleration=fields_at_stage
                    if any(z.shape!=shape or not np.isfinite(z).all()
                           for z in (mechanical,velocity,acceleration)):
                        raise RuntimeError("MKR_NONFINITE_OR_WRONG_SHAPE_STAGE")
                    outer=_outer_radial_field(mechanical,physics.theory)
                    pending.append((
                        output/f"MKR_STEP_{k:04d}_{stage}.npz",
                        time_s,mechanical,velocity,acceleration,
                        {"step":k,"stage":stage,"time_s":time_s,
                         "central_w0_m":_central_grid_value(mechanical[radial]),
                         "maximum_abs_w0_m":float(np.max(np.abs(
                             mechanical[radial]))),
                         "central_outer_w_m":_central_grid_value(outer),
                         "maximum_abs_outer_w_m":float(
                             np.max(np.abs(outer)))}
                    ))
                for filename,stage_time,mechanical,velocity,acceleration,row in pending:
                    _save_npz_atomic(
                        filename,time_s=stage_time,mechanical=mechanical,
                        velocity=velocity,acceleration=acceleration)
                for *_,row in pending:
                    writer.writerow(row)
                stream.flush()
                q,v,a=tuple(np.array(z,copy=True) for z in final)
                summary["accepted_steps"]=k
                summary["time_s"]=k*dt_s
                save_summary()
                committed_steps=k
                committed_csv_position=stream.tell()
                print(f"MKR ACCEPTED {k}/{steps} t={k*dt_s:.9g} s",flush=True)
            summary["status"]="CONVERGED"
            summary["converged"]=True
            save_summary()
            print(f"MKR COMPLETED {steps}/{steps}",flush=True)
        except Exception as exc:
            # The summary is the commit marker. Discard any partially written
            # stages and CSV lines belonging to the uncommitted time step.
            if failed_step is not None and failed_step > committed_steps:
                for orphan in output.glob(f"MKR_STEP_{failed_step:04d}_*.npz"):
                    orphan.unlink(missing_ok=True)
                stream.seek(committed_csv_position)
                stream.truncate()
                stream.flush()
            summary["accepted_steps"]=committed_steps
            summary["time_s"]=committed_steps*dt_s
            summary.update(status="FAILED",converged=False,failed_step=failed_step,
                           error_type=type(exc).__name__,detail=str(exc))
            save_summary()
            print(f"MKR FAILED step={failed_step} error={type(exc).__name__}: {exc}",
                  flush=True)
            raise
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
