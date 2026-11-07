"""Physical grid/time/rank studies for coupled G3D/Gz cylindrical shells.

Never replace a nonconverged physical result with a synthetic state or
another spatial method. Every comparison requires an identical continuum
physics signature and persisted NPZ fields from this exact GitHub commit.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_final_stage11a_dispatcher import run_final_case
from cylinder_reference_case import prepare_reference_case
from ci.scientific_real_coupled_case import build_case

STUDIES = ("KL_GRID","TE1_GRID","TE1_TIME","TE1_RANK")
PILOT_IDS = ("KL_GRID_5x5","KL_GRID_7x7",
             "TE1_GRID_4x4","TE1_GRID_5x5",
             "TE1_TIME_N2","TE1_TIME_N4",
             "TE1_RANK_FDM","TE1_RANK_MVI_R1")
ROOT = Path(__file__).resolve().parents[1]


def catalog(scope="full"):
    if scope not in ("full","pilot"):
        raise ValueError("SCIENTIFIC_SCOPE_INVALID")
    entries = []
    def add(study,variant,theory,gradient,on,kind,method,nx,ns,
            *,rank=1,steps=1,dt=.01,patch=False):
        entries.append(dict(
            id=study+"_"+variant,study=study,variant=variant,
            theory=theory,gradient=gradient,on=on,kind=kind,method=method,
            nx=nx,ns=ns,rank=rank,steps=steps,dt=dt,patch=patch))
    for n in (5,7,9):
        add("KL_GRID",f"{n}x{n}","KL","G3D",True,"STATIC","FDM",n,n)
    for n in (4,5,6):
        add("TE1_GRID",f"{n}x{n}","CUF_TE1","Gz",True,"STATIC","FDM",n,n)
    for steps in (2,4,8):
        add("TE1_TIME",f"N{steps}","CUF_TE1","Gz",True,
            "DYNAMIC","MVI",4,4,steps=steps,dt=.02/steps)
    for rank in (0,1,2,3):
        method="FDM" if rank==0 else "MVI"
        add("TE1_RANK","FDM" if rank==0 else f"MVI_R{rank}",
            "CUF_TE1","Gz",True,"STATIC",method,5,5,
            rank=max(1,rank),patch=True)
    assert len(entries)==13 and len({e["id"] for e in entries})==13
    return ([e for e in entries if e["id"] in PILOT_IDS]
            if scope=="pilot" else entries)


def case_from_entry(entry,folder):
    case=build_case(entry,folder)
    grid=case["fdm_grid" if entry["method"]=="FDM" else "mvi_grid"]
    grid["nx"],grid["ns"]=entry["nx"],entry["ns"]
    # Keep one spatial operator family across mesh levels: otherwise
    # grid refinement would be confounded with changing discretization.
    grid["operator_scheme"]="LEGACY_REFERENCE"
    if entry["method"]=="MVI":
        case["rank"]=entry["rank"]
        case["reference_mvi"]={"max_sweeps":45}
    else:
        # The dense benchmark has a documented 512 free-DOF upper bound.
        case["reference_newton"]["max_free_dofs"]=512
    if entry["kind"]=="DYNAMIC":
        case["dynamics"]["time_step_s"]=entry["dt"]
        case["dynamics"]["n_steps"]=entry["steps"]
        assert math.isclose(entry["steps"]*entry["dt"],.02,abs_tol=1e-14)
    if entry["patch"]:
        case["load_definition"].update(
            type="PATCH_RADIAL_FACE",full_circumference=False,
            x1=.15,x2=1.05,edge_width_x=.30,
            c1=.30,c2=5.40,edge_width_c=1.00)
    return case


def physical_identity(case):
    """Same continuum problem; method, grid, rank and dt are NOT physics."""
    keys=("theory","cuf_model","scale_model","mode","geometry_model",
          "material_model","material","micro","geometry","quadrature",
          "boundary_conditions","load_definition","units","calculation_kind")
    fields={key:case[key] for key in keys if key in case}
    if case["calculation_kind"]=="DYNAMIC":
        dynamic=dict(case["dynamics"])
        steps=dynamic.pop("n_steps")
        dt=dynamic.pop("time_step_s")
        fields["time_horizon_s"]=round(steps*dt,14)
        fields["dynamics_except_discretization"]=dynamic
    return sha256(json.dumps(fields,sort_keys=True,
                             separators=(",",":")).encode()).hexdigest()


def digest(path):
    h=sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _probe(field,x,theta,length):
    """Same physical (L/2, theta=pi) point on every nodal grid."""
    data=np.asarray(field,float)
    if data.ndim!=2 or not np.isfinite(data).all():
        raise ValueError("SCIENTIFIC_PROBE_REQUIRES_FINITE_2D_FIELD")
    nx,ns=data.shape
    axial=np.linspace(0.,length,nx)
    along=np.array([np.interp(x,axial,data[:,j])
                    for j in range(ns)])
    phi=np.arange(ns)*2*np.pi/ns
    return float(np.interp(theta,np.r_[phi,2*np.pi],
                           np.r_[along,along[0]]))


def _radial_index(theory):
    return (2*(int(theory[-1])+1) if theory.startswith("CUF_TE")
            else 2)


def classify_accepted_steps(summary,entry):
    """Interpret accepted fixed-rank MVI as rank-limited, never full-space.

    For STATIC MVI, CONVERGED_PROJECTED is an accepted *rank-constrained*
    solution. Its ability to represent the true full-space problem remains
    unverified and must not be reported as CONVERGED_FULL.
    """
    steps=summary.get("steps",[])
    n=entry["steps"]
    if (summary.get("converged") is not True or
        summary.get("accepted_steps")!=n or len(steps)!=n or
        any(step.get("converged") is not True for step in steps)):
        raise ValueError("SCIENTIFIC_CONVERGENCE_STEPS_NOT_ACCEPTED")
    actual_status=summary.get("status")
    if entry["method"]=="MVI" and entry["kind"]=="STATIC":
        if actual_status not in ("CONVERGED_FULL","CONVERGED_PROJECTED"):
            raise ValueError("SCIENTIFIC_MVI_UNRECOGNIZED_ACCEPTED_STATUS")
        cert=steps[-1].get("mvi_certificate",{})
        is_full=actual_status=="CONVERGED_FULL"
        if (steps[-1].get("status")!=actual_status or
            cert.get("solution_status")!=actual_status or
            cert.get("accepted") is not True or
            cert.get("projected_accepted") is not True or
            cert.get("boundary_constraint_satisfied") is not True or
            cert.get("profile_reconstruction_satisfied") is not True or
            cert.get("projected_equilibrium_satisfied") is not True or
            cert.get("full_equilibrium_satisfied") is not is_full or
            steps[-1].get("full_equilibrium_satisfied") is not is_full):
            raise ValueError("SCIENTIFIC_MVI_CERTIFICATE_INCONSISTENT")
        return {
            "solution_class":("FULL_SPACE_STATIONARY" if is_full
                              else "RANK_CONSTRAINED_ONLY"),
            "full_space_stationarity_confirmed":is_full,
            "fixed_rank_stationarity_confirmed":True,
            "spatial_rank_limit_detected":not is_full,
        }
    if actual_status!="CONVERGED":
        raise ValueError("SCIENTIFIC_FULL_SPACE_SOLVE_NOT_CONVERGED")
    return {
        "solution_class":("DYNAMIC_STEP_ACCEPTED" if entry["kind"]=="DYNAMIC"
                          else "FULL_SPACE_STATIONARY"),
        # A dynamic MVI step is a time-advance acceptance, not an external
        # full-space rank certification; do not infer such a claim.
        "full_space_stationarity_confirmed":(
            True if entry["kind"]=="STATIC" else None),
        "fixed_rank_stationarity_confirmed":None,
        "spatial_rank_limit_detected":None,
    }


def verify(case,entry,folder):
    out=folder/"RESULTS"
    summary=json.loads((out/"FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    qualification=classify_accepted_steps(summary,entry)
    if summary["theory"]!=entry["theory"] or summary["method"]!=entry["method"]:
        raise ValueError("SCIENTIFIC_CONVERGENCE_IDENTITY_MISMATCH")
    if (summary["gradient_model"]!=entry["gradient"] or
        summary["electrical_enabled"] is not entry["on"]):
        raise ValueError("SCIENTIFIC_CONVERGENCE_MATERIAL_MISMATCH")
    final=summary["steps"][-1]["state_file"]
    prefix=("REFERENCE_DYNAMIC_STEP_" if entry["kind"]=="DYNAMIC"
            else "REFERENCE_STEP_")
    if final!=prefix+f"{entry['steps']:04d}.npz":
        raise ValueError("SCIENTIFIC_CONVERGENCE_FINAL_FILE_MISMATCH")
    file=out/final
    length=case["geometry"]["L"]
    with np.load(file,allow_pickle=False) as a:
        q=np.asarray(a["mechanical"],float)
        if q.shape[1:]!=(entry["nx"],entry["ns"]):
            raise ValueError("SCIENTIFIC_CONVERGENCE_FIELD_SHAPE")
        if entry["kind"]=="DYNAMIC":
            if not np.isclose(float(a["time_s"]),.02,rtol=0,atol=1.e-12):
                raise ValueError("SCIENTIFIC_CONVERGENCE_FINAL_TIME")
            if not {"velocity","acceleration"}.issubset(a.files):
                raise ValueError("SCIENTIFIC_CONVERGENCE_DYNAMIC_FIELDS")
        if ("electrical" in a.files)!=entry["on"]:
            raise ValueError("SCIENTIFIC_CONVERGENCE_ELECTRIC_FIELDS")
        for key in a.files:
            if np.issubdtype(a[key].dtype,np.number) and not np.isfinite(a[key]).all():
                raise ValueError("SCIENTIFIC_CONVERGENCE_NONFINITE_"+key)
        if entry["method"]=="MVI":
            x,y,scales=a["x_profiles"],a["y_profiles"],a["state_scales"]
            field=np.einsum("crx,crs->cxs",x,y)
            target=np.concatenate((q,a["electrical"])) if entry["on"] else q
            np.testing.assert_allclose(scales*field,target,rtol=2e-8,atol=2e-11)
        radial=q[_radial_index(entry["theory"])]
        observed=dict(
            w_mid_probe_m=_probe(radial,length/2,np.pi,length),
            w_mid_max_abs_m=float(np.max(np.abs(radial))))
        if observed["w_mid_max_abs_m"]<=1.e-15:
            raise ValueError("SCIENTIFIC_CONVERGENCE_ZERO_RADIAL_RESPONSE")
        if "face_outer_radial_displacement_m" in a.files:
            observed["w_outer_probe_m"]=_probe(
                a["face_outer_radial_displacement_m"],length/2,np.pi,length)
        if entry["on"]:
            observed["potential_c0_probe_v"]=_probe(
                a["electrical"][0],length/2,np.pi,length)
            if "face_outer_electric_outward_induction_c_per_m2" in a.files:
                observed["face_outer_induction_probe_c_per_m2"]=_probe(
                    a["face_outer_electric_outward_induction_c_per_m2"],
                    length/2,np.pi,length)
        if "stress6_pa" in a.files:
            z=np.asarray(a["z_m"])
            middle=int(np.argmin(np.abs(z)))
            observed["sigma_xx_mid_probe_pa"]=_probe(
                a["stress6_pa"][middle,:,:,0],length/2,np.pi,length)
        if entry["kind"]=="DYNAMIC":
            observed["velocity_mid_probe_m_s"]=_probe(
                a["velocity"][_radial_index(entry["theory"])],
                length/2,np.pi,length)
    return dict(accepted=True,state_file=final,state_sha256=digest(file),
                observations=observed,**qualification,
                scientific_accuracy_certified=False)


def run(entry,folder):
    folder=Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("SCIENTIFIC_CONVERGENCE_OUTPUT_ALREADY_EXISTS")
    folder.mkdir(parents=True,exist_ok=True)
    case=case_from_entry(entry,folder)
    filename=folder/"CASE.json"
    filename.write_text(json.dumps(case,indent=2)+"\n",encoding="utf-8")
    report=folder/"CONVERGENCE_EVIDENCE.json"
    record=dict(case_id=entry["id"],study=entry["study"],
                variant=entry["variant"],method=entry["method"],
                grid=[entry["nx"],entry["ns"]],
                rank=entry["rank"] if entry["method"]=="MVI" else None,
                dt_s=entry["dt"] if entry["kind"]=="DYNAMIC" else None,
                steps=entry["steps"],
                physics_sha256=physical_identity(case),
                case_sha256=digest(filename),
                git_sha=os.environ.get("GITHUB_SHA","UNKNOWN"),
                status="IN_PROGRESS",
                note="not an external benchmark or accuracy certificate")
    def save():
        report.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    save()
    try:
        prepared=prepare_reference_case(case,base_dir=folder)
        if prepared.method!=entry["method"] or prepared.calculation_kind!=entry["kind"]:
            raise ValueError("SCIENTIFIC_PREPARED_CASE_MISMATCH")
        run_final_case(filename)
        record.update(verify(case,entry,folder))
        record["status"]="ACCEPTED"
        save()
        print("SCIENTIFIC_CONVERGENCE_ACCEPTED",entry["id"],flush=True)
    except BaseException as exc:
        record.update(status="FAILED",failure_type=type(exc).__name__,
                      failure_message=str(exc))
        save()
        (folder/"TRACEBACK.txt").write_text(traceback.format_exc())
        raise


def compare(root,out,scope="full"):
    """Compare only requested campaign entries; missing ones remain failures."""
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    candidates=list(root.rglob("CONVERGENCE_EVIDENCE.json"))
    for entry in catalog(scope):
        matches=[]
        for path in candidates:
            try:
                record=json.loads(path.read_text())
            except (ValueError,OSError):
                continue
            if record.get("case_id")==entry["id"]:
                matches.append((path,record))
        if len(matches)!=1:
            rows.append(dict(case_id=entry["id"],study=entry["study"],
                             status="MISSING_OR_DUPLICATED",method=entry["method"]))
            continue
        path,record=matches[0]
        row={k:v for k,v in record.items() if k not in ("observations",)}
        row.update(record.get("observations",{}))
        rows.append(row)
    groups={}
    for row in rows:
        groups.setdefault(row["study"],[]).append(row)
    report={"records":rows,"series":{},"all_accepted":False,
            "all_static_cases_full_space_stationary":False,
            "rank_constrained_cases":[],
            "external_accuracy_certified":False,
            "independent_coupled_fdm_mvi_certified":False}
    for study,group in groups.items():
        accepted=[x for x in group if x["status"]=="ACCEPTED"]
        signatures={x["physics_sha256"] for x in accepted}
        if len(signatures)>1:
            raise ValueError("SCIENTIFIC_PHYSICS_PASSPORT_MISMATCH_"+study)
        comparisons=[]
        for left,right in zip(accepted[:-1],accepted[1:]):
            for metric in ("w_mid_probe_m","w_outer_probe_m",
                           "potential_c0_probe_v","sigma_xx_mid_probe_pa"):
                if metric in left and metric in right:
                    a,b=left[metric],right[metric]
                    comparisons.append(dict(from_case=left["case_id"],
                        to_case=right["case_id"],observable=metric,
                        absolute_change=abs(b-a),
                        relative_to_later=abs(b-a)/max(abs(b),1.e-15)))
        report["series"][study]={
            "accepted_count":len(accepted),
            "requested_count":len(group),
            "rank_constrained_case_ids":[
                x["case_id"] for x in accepted
                if x.get("solution_class")=="RANK_CONSTRAINED_ONLY"],
            "same_continuum_physics":len(signatures)<=1,
            "adjacent_changes_not_error_bounds":comparisons}
    report["all_accepted"]=all(row["status"]=="ACCEPTED" for row in rows)
    report["rank_constrained_cases"]=[
        row["case_id"] for row in rows
        if row.get("solution_class")=="RANK_CONSTRAINED_ONLY"]
    static_rows=[row for row in rows
                 if row.get("solution_class") in (
                     "FULL_SPACE_STATIONARY","RANK_CONSTRAINED_ONLY")]
    report["all_static_cases_full_space_stationary"]=bool(static_rows) and all(
        row.get("full_space_stationarity_confirmed") is True
        for row in static_rows) and report["all_accepted"]
    (out/"SCIENTIFIC_COMPARISON.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    cols=["study","case_id","status","solution_class",
          "full_space_stationarity_confirmed","method","grid","rank","dt_s","steps",
          "physics_sha256","w_mid_probe_m","w_mid_max_abs_m",
          "w_outer_probe_m","potential_c0_probe_v",
          "face_outer_induction_probe_c_per_m2","sigma_xx_mid_probe_pa",
          "velocity_mid_probe_m_s","failure_type","failure_message"]
    with (out/"SCIENTIFIC_CONVERGENCE_DATA.csv").open(
            "w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=cols,extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v
                             for k,v in row.items()})
    if not report["all_accepted"]:
        raise RuntimeError("SCIENTIFIC_CONVERGENCE_HAS_FAILED_OR_MISSING_CASES")
    return report


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest="command",required=True)
    listing=commands.add_parser("matrix")
    listing.add_argument("--scope",choices=("pilot","full"),default="pilot")
    p=commands.add_parser("run")
    p.add_argument("--id",required=True)
    p.add_argument("--output",type=Path,required=True)
    p=commands.add_parser("compare")
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scope",choices=("pilot","full"),default="pilot")
    a=parser.parse_args()
    if a.command=="matrix":
        print(json.dumps({"include":catalog(a.scope)},separators=(",",":")))
    elif a.command=="run":
        found=[x for x in catalog("full") if x["id"]==a.id]
        if len(found)!=1:raise ValueError("SCIENTIFIC_CASE_NOT_FOUND")
        run(found[0],a.output)
    else:
        compare(a.root,a.output,scope=a.scope)


if __name__=="__main__":
    main()
