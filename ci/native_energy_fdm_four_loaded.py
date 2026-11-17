"""Four REAL loaded native energy-FDM KL/G3D,Gz STATIC/DYNAMIC pilot cases.

No toy residual is injected. The nonlinear cylindrical thickness kernel,
true curved outer-face pressure, and coupled electric potential are used.
A tiny 7x7 physical grid limits diagnostic runtime. This pilot makes NO
mesh accuracy, free-face boundary, MVI parity, or production claims.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from cylinder_paths import activate
activate()
from independent_mkr import run_energy_fdm_research as energy
from native_gradient_fdm_mvi_campaign import MATERIAL_BASE,pressure_profile

NX=7
NS=7
STEPS=2
DT_S=1.e-4
PRESSURE_PA=100.


def _save(path,source):
    path.write_text(json.dumps(source,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")


def create_case(folder,gradient,kind):
    folder.mkdir(parents=True,exist_ok=False)
    q=np.zeros((3,NX,NS),dtype=float)
    p=np.zeros((2,NX,NS),dtype=float)
    state=dict(mechanical=q,electrical=p)
    if kind=="DYNAMIC":
        state["velocity"]=np.zeros_like(q)
    np.savez_compressed(folder/"initial.npz",**state)
    np.savez_compressed(folder/"loads.npz",mechanical_load=q,
                        electrical_charge=p)
    pressure=-pressure_profile(NS)
    if not np.any(pressure) or not np.isfinite(pressure).all():
        raise AssertionError("REAL_PRESSURE_MISSING")
    settings=dict(
        kind=kind,integrator="NEWMARK" if kind=="DYNAMIC" else None,
        dt_s=DT_S if kind=="DYNAMIC" else None,
        steps=STEPS if kind=="DYNAMIC" else 0,
        max_dofs=1400,max_iterations=35,
        linear_solver="AUTO",dense_lu_dof_limit=650,
        mechanical_state_scale=1.e-4,
        mechanical_equation_scale=1.e4,
        electrical_state_scale=1.,electrical_equation_scale=1.,
        potential_scale_v=1.)
    case=dict(
        schema=energy.SCHEMA,research_opt_in=True,
        physics=dict(theory="KL",geometry_model="GREEN_LAGRANGE",
                     material=dict(MATERIAL_BASE,gradient_model=gradient),
                     thickness_m=.08,density_kg_m3=2500.,
                     eta_s=0.,mass_damping_rate_per_s=0.,
                     n_gauss=12,potential_order=1),
        grid=dict(nx=NX,ns=NS,length_m=1.2,radius_m=2.),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                      electric="GROUNDED_BOTH",axial_orders=[2,2,2]),
        solver=settings,initial_state_npz="initial.npz",
        loads_npz="loads.npz",output_dir="RESULTS",
        kl_radial_face_traction=dict(face="OUTER",
             traction_pa=np.broadcast_to(pressure,(NX,NS)).tolist()))
    _save(folder/"CASE.json",case)
    return folder/"CASE.json"


def run(gradient,kind,output):
    if gradient not in ("G3D","Gz") or kind not in ("STATIC","DYNAMIC"):
        raise ValueError("REAL_ENERGY_PILOT_UNSUPPORTED_CASE")
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    report=dict(gradient=gradient,kind=kind,git_sha=os.getenv("GITHUB_SHA"),
                method=energy.METHOD,physics="REAL_KL_GL_FLEXO_COUPLED",
                grid=[NX,NS],outer_face_pressure_pa=PRESSURE_PA,
                scientific_accuracy_certified=False,
                physical_boundary_closure_certified=False,
                mesh_convergence_certified=False,status="IN_PROGRESS")
    marker=root/"RESULT.json"
    _save(marker,report)
    try:
        case=create_case(root/"input",gradient,kind)
        preview=energy.run(case,validate_only=True)
        if preview["validated_only"] is not True or preview["schema"]!=energy.SCHEMA:
            raise AssertionError("REAL_ENERGY_PILOT_PREFLIGHT")
        solved=energy.run(case,validate_only=False)
        expected=1 if kind=="STATIC" else STEPS
        if solved["accepted_steps"]!=expected:
            raise AssertionError("REAL_ENERGY_PILOT_UNACCEPTED_STEPS")
        state=root/"input"/"RESULTS"/(
            "STATIC_ACCEPTED.npz" if kind=="STATIC" else f"STEP_{STEPS:04d}.npz")
        with np.load(state,allow_pickle=False) as f:
            mechanical=np.asarray(f["mechanical"],float)
            electrical=np.asarray(f["electrical"],float)
            if mechanical.shape!=(3,NX,NS) or electrical.shape!=(2,NX,NS):
                raise AssertionError("REAL_ENERGY_PILOT_FIELD_SHAPE")
            if not np.isfinite(mechanical).all() or not np.isfinite(electrical).all():
                raise AssertionError("REAL_ENERGY_PILOT_NONFINITE")
            if kind=="DYNAMIC":
                for name in ("velocity","acceleration"):
                    if f[name].shape!=(3,NX,NS) or not np.isfinite(f[name]).all():
                        raise AssertionError("REAL_ENERGY_PILOT_"+name.upper())
                if not math.isclose(float(f["time_s"]),STEPS*DT_S,
                                    rel_tol=1.e-12,abs_tol=1.e-14):
                    raise AssertionError("REAL_ENERGY_PILOT_WRONG_TIME")
        max_w=float(np.max(np.abs(mechanical[2])))
        max_phi=float(np.max(np.abs(electrical)))
        if max_w<=1.e-14 or max_phi<=1.e-14:
            raise AssertionError("REAL_ENERGY_PILOT_NO_NONZERO_RESPONSE")
        report.update(status="ACCEPTED_UNCERTIFIED",accepted_steps=expected,
                      max_midradial_m=max_w,max_potential_v=max_phi,
                      output=str(state),
                      newton=solved.get("static_nonlinear_diagnostics",
                                        solved.get("initial_nonlinear_diagnostics")))
        _save(marker,report)
        print("REAL_ENERGY_FDM_ACCEPTED",json.dumps(report),flush=True)
        return report
    except BaseException as exc:
        report.update(status="FAILED",error_type=type(exc).__name__,
                      error=str(exc))
        _save(marker,report)
        (root/"TRACEBACK.txt").write_text(traceback.format_exc(),
                                          encoding="utf-8")
        print("REAL_ENERGY_FDM_FAILED",gradient,kind,
              type(exc).__name__,str(exc),flush=True)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gradient",choices=("G3D","Gz"),required=True)
    parser.add_argument("--kind",choices=("STATIC","DYNAMIC"),required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    run(a.gradient,a.kind,a.output)


if __name__=="__main__":
    main()
