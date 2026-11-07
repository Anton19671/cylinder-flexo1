"""Real coupled CUF TE1 Gz nonlinear Newmark: same-grid FDM vs MVI 1..8.

Default real GitHub Actions push campaign: FDM and MVI ranks 1,2,3.
Opt-in full GitHub dispatch: FDM and all MVI ranks 1..8.

All cases use the same 8x8 shell mesh and one non-axisymmetric outer-face
load; two Newmark steps of .01 s, with identical actual physical time
coordinates. Every accepted step is checked and hashed independently.
There is no electrical inertia and no postprocessing substitute for
a failed coupled dynamic solve.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import traceback

import numpy as np
from cylinder_paths import activate
activate()

from cylinder_reference_case import prepare_reference_case
from cylinder_final_stage11a_dispatcher import run_final_case
from scientific_coupled_convergence import case_from_entry, physical_identity, digest, verify
from scientific_coupled_field_compare import matching_fields, metric
from scientific_coupled_rank_1_8 import catalog as static_rank_catalog

STUDY="TE1_NONAX_DYNAMIC_8X8"
DT=.01
STEPS=2
NX=NS=8
FIELDS_DYNAMIC=("mechanical","velocity","acceleration","electrical",
                "face_outer_radial_displacement_m",
                "face_outer_electric_outward_induction_c_per_m2",
                "stress6_pa","hyperstress6x3_pa_m",
                "induction3_c_per_m2")
PILOT_RANKS=(1,2,3)


def catalog(scope="pilot"):
    if scope not in ("pilot","full"):
        raise ValueError("DYNAMIC_RANK_SCOPE_INVALID")
    result=[]
    for source in static_rank_catalog():
        rank=source["rank"]
        if scope=="pilot" and source["method"]=="MVI" and rank not in PILOT_RANKS:
            continue
        entry=dict(source)
        entry.update(
            id=("TE1_DYNAMIC_8X8_FDM" if source["method"]=="FDM"
                else f"TE1_DYNAMIC_8X8_MVI_R{rank}"),
            study=STUDY,kind="DYNAMIC",steps=STEPS,dt=DT,
            patch=True,nx=NX,ns=NS,
        )
        result.append(entry)
    assert len(result)==(4 if scope=="pilot" else 9)
    assert len({x["id"] for x in result})==len(result)
    return result


def build_case(entry,output):
    case=case_from_entry(entry,output)
    case["reference_newton"]["max_free_dofs"]=512
    if entry["method"]=="MVI":
        case["reference_mvi"]["max_sweeps"]=55
    case["postprocess"]={"dynamic_analytics":False}
    if (case["dynamics"]["n_steps"]!=STEPS
            or case["dynamics"]["time_step_s"]!=DT
            or case["load_definition"]["type"]!="PATCH_RADIAL_FACE"):
        raise AssertionError("DYNAMIC_RANK_PHYSICS_AND_TIME_CHANGED")
    return case


def accepted_time_states(record,folder):
    """Read each accepted physical Newmark state, verify its exact hash."""
    folder=Path(folder)
    summary=json.loads((folder/"RESULTS"/"FINAL_RUN_SUMMARY.json").read_text())
    if (summary.get("status")!="CONVERGED" or
        summary.get("converged") is not True or
        summary.get("accepted_steps")!=STEPS or
        summary.get("requested_steps")!=STEPS or
        summary.get("electric_inertia") is not False or
        len(summary.get("steps",[]))!=STEPS):
        raise ValueError("DYNAMIC_RANK_INCOMPLETE_TIME_HISTORY")
    checks=[]
    for i,step in enumerate(summary["steps"],1):
        t=i*DT
        if (step.get("converged") is not True or
            not np.isclose(step.get("time_s"),t,rtol=0,atol=1.e-12) or
            step.get("state_file")!=f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz"):
            raise ValueError("DYNAMIC_RANK_STEP_NOT_ACCEPTED")
        file=folder/"RESULTS"/step["state_file"]
        if not file.is_file():
            raise ValueError("DYNAMIC_RANK_STEP_FILE_MISSING")
        with np.load(file,allow_pickle=False) as saved:
            for name in FIELDS_DYNAMIC:
                if name not in saved:
                    raise ValueError("DYNAMIC_RANK_MISSING_PHYSICAL_FIELD_"+name)
                if not np.isfinite(saved[name]).all():
                    raise ValueError("DYNAMIC_RANK_NONFINITE_FIELD_"+name)
            if not np.isclose(float(saved["time_s"]),t,rtol=0,atol=1.e-12):
                raise ValueError("DYNAMIC_RANK_TIME_MISMATCH")
            if saved["electrical"].shape!=(2,NX,NS):
                raise ValueError("DYNAMIC_RANK_ELECTRIC_ORDER_MISMATCH")
            if saved["mechanical"].shape!=(6,NX,NS):
                raise ValueError("DYNAMIC_RANK_MECHANICAL_SHAPE_MISMATCH")
            for name in ("velocity","acceleration"):
                if saved[name].shape!=(6,NX,NS):
                    raise ValueError("DYNAMIC_RANK_MOTION_FIELD_SHAPE_MISMATCH")
        checks.append({"index":i,"time_s":t,
                       "file":step["state_file"],
                       "sha256":digest(file),
                       "mvi_certificate":step.get("mvi_certificate")})
    if checks[-1]["sha256"]!=record["state_sha256"]:
        raise ValueError("DYNAMIC_RANK_FINAL_STATE_HASH_MISMATCH")
    return checks


def run(entry,directory):
    directory=Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("DYNAMIC_RANK_OUTPUT_DIRECTORY_NOT_EMPTY")
    directory.mkdir(parents=True,exist_ok=True)
    case=build_case(entry,directory)
    casefile=directory/"CASE.json"
    casefile.write_text(json.dumps(case,indent=2)+"\n",encoding="utf-8")
    record=dict(
        case_id=entry["id"],study=STUDY,
        rank=entry["rank"] if entry["method"]=="MVI" else None,
        method=entry["method"],grid=[NX,NS],
        calculation_kind="DYNAMIC",integrator="NEWMARK",
        total_time_s=STEPS*DT,steps_requested=STEPS,
        case_sha256=digest(casefile),
        physics_sha256=physical_identity(case),
        git_sha=os.getenv("GITHUB_SHA","UNKNOWN"),
        status="IN_PROGRESS",
        spatial_accuracy_certified=False,
        independent_methods_certified=False,
        even_periodic_grid_accuracy_certified=False)
    evidence=directory/"DYNAMIC_RANK_EVIDENCE.json"
    def persist():
        evidence.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",
                            encoding="utf-8")
    persist()
    try:
        prepared=prepare_reference_case(case,base_dir=directory)
        if (prepared.calculation_kind!="DYNAMIC" or
            prepared.method!=entry["method"] or
            prepared.problem.functional.theory!="CUF_TE1" or
            prepared.electrical is None or
            prepared.electrical.shape!=(2,NX,NS) or
            prepared.dynamics.settings.n_steps!=STEPS or
            not np.isclose(prepared.dynamics.settings.dt_s,DT)):
            raise ValueError("DYNAMIC_RANK_CASE_DISPATCH_MISMATCH")
        if entry["method"]=="MVI" and prepared.mvi_settings.rank!=entry["rank"]:
            raise ValueError("DYNAMIC_RANK_MVI_FIXED_RANK_MISMATCH")
        run_final_case(casefile)
        record.update(verify(case,entry,directory))
        record["accepted_states"]=accepted_time_states(record,directory)
        record["status"]="ACCEPTED"
        persist()
        print("DYNAMIC_RANK_REAL_ACCEPTED",entry["id"],flush=True)
    except BaseException as exc:
        record.update(status="FAILED",failure_type=type(exc).__name__,
                      failure_message=str(exc))
        persist()
        (directory/"FAILURE_TRACEBACK.txt").write_text(
            traceback.format_exc(),encoding="utf-8")
        raise


def archived_case(root,entry):
    """Check provenance of exact CASE and every real accepted time state."""
    matches=[]
    for file in root.rglob("DYNAMIC_RANK_EVIDENCE.json"):
        try: e=json.loads(file.read_text(encoding="utf-8"))
        except (ValueError,OSError):continue
        if e.get("case_id")==entry["id"]:
            matches.append((file,e))
    if len(matches)!=1:
        raise ValueError("DYNAMIC_RANK_CASE_MISSING_OR_DUPLICATE")
    file,record=matches[0]
    if record.get("status")!="ACCEPTED":
        raise ValueError("DYNAMIC_RANK_CASE_NOT_ACCEPTED")
    folder=file.parent
    cfile=folder/"CASE.json"
    if not cfile.is_file() or record.get("case_sha256")!=digest(cfile):
        raise ValueError("DYNAMIC_RANK_CASE_HASH_INVALID")
    case=json.loads(cfile.read_text())
    if (record.get("physics_sha256")!=physical_identity(case) or
        record.get("method")!=entry["method"] or
        record.get("grid")!=[NX,NS] or
        case.get("calculation_kind")!="DYNAMIC" or
        record.get("integrator")!="NEWMARK"):
        raise ValueError("DYNAMIC_RANK_CASE_PASSPORT_INVALID")
    grid=case["fdm_grid" if entry["method"]=="FDM" else "mvi_grid"]
    if (grid["nx"],grid["ns"])!=(NX,NS):
        raise ValueError("DYNAMIC_RANK_GRID_MISMATCH")
    if entry["method"]=="MVI" and case.get("rank")!=entry["rank"]:
        raise ValueError("DYNAMIC_RANK_FIXED_RANK_MISMATCH")
    checks=accepted_time_states(record,folder)
    saved=record.get("accepted_states")
    if not isinstance(saved,list) or len(saved)!=len(checks):
        raise ValueError("DYNAMIC_RANK_EVIDENCE_STEP_COUNT_MISMATCH")
    for original,actual in zip(saved,checks):
        for key in ("index","time_s","file","sha256"):
            if original.get(key)!=actual[key]:
                raise ValueError("DYNAMIC_RANK_STEP_HASH_OR_TIME_CHANGED")
    return record,folder


def compare(root,destination,scope="pilot"):
    root,destination=Path(root),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    entries=catalog(scope)
    failures=[]
    records={}
    for entry in entries:
        try:records[entry["id"]]=archived_case(root,entry)
        except (ValueError,KeyError,OSError) as exc:
            failures.append({"id":entry["id"],"reason":str(exc)})
    fdm=records.get(entries[0]["id"])
    comparisons=[]
    if fdm:
        ref,ref_dir=fdm
        for entry in entries[1:]:
            candidate=records.get(entry["id"])
            if candidate is None:continue
            record,directory=candidate
            if (ref["physics_sha256"]!=record["physics_sha256"] or
                ref["git_sha"]!=record["git_sha"]):
                raise ValueError("DYNAMIC_RANK_PHYSICS_OR_REVISION_DIFFER")
            steps=[]
            for index in range(1,STEPS+1):
                name=f"REFERENCE_DYNAMIC_STEP_{index:04d}.npz"
                with np.load(ref_dir/"RESULTS"/name,allow_pickle=False) as a, np.load(
                    directory/"RESULTS"/name,allow_pickle=False) as b:
                    metrics=matching_fields(a,b)
                    for field in ("velocity","acceleration"):
                        metrics[field]=metric(a[field],b[field])
                steps.append({"index":index,"time_s":index*DT,
                              "fields":metrics})
            comparisons.append({"rank":entry["rank"],"id":entry["id"],
                                "solution_class":record["solution_class"],
                                "per_time_step":steps})
    result={
        "study":STUDY,"scope":scope,
        "time_integrator":"NEWMARK",
        "time_step_s":DT,"n_steps":STEPS,"mesh":[NX,NS],
        "requested_cases":len(entries),"accepted_cases":len(records),
        "failed_or_missing":failures,
        "comparisons":comparisons,
        "all_requested_ranks_compared":len(comparisons)==len(entries)-1,
        "physical_accuracy_certified":False,
        "time_accuracy_certified":False,
        "spatial_independence_of_coupled_methods_certified":False,
    }
    (destination/"DYNAMIC_MVI_VS_FDM.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    fields=("mechanical","velocity","acceleration","electrical",
            "face_outer_radial_displacement_m","stress6_pa",
            "induction3_c_per_m2",
            "face_outer_electric_outward_induction_c_per_m2")
    with (destination/"DYNAMIC_MVI_VS_FDM.csv").open(
        "w",newline="",encoding="utf-8") as io:
        writer=csv.DictWriter(io,fieldnames=(
            "rank","step","time_s","field","relative_l2",
            "max_abs_difference","reference_peak_abs","solution_class"))
        writer.writeheader()
        for candidate in comparisons:
            for step in candidate["per_time_step"]:
                for name in fields:
                    m=step["fields"][name]
                    writer.writerow({
                        "rank":candidate["rank"],
                        "step":step["index"],"time_s":step["time_s"],
                        "field":name,"relative_l2":m["relative_l2"],
                        "max_abs_difference":m["max_abs_difference"],
                        "reference_peak_abs":m["reference_peak_abs"],
                        "solution_class":candidate["solution_class"],
                    })
    if not result["all_requested_ranks_compared"]:
        raise RuntimeError("DYNAMIC_RANK_REAL_CASES_INCOMPLETE")
    return result


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="command",required=True)
    c=sub.add_parser("matrix")
    c.add_argument("--scope",choices=("pilot","full"),default="pilot")
    c=sub.add_parser("run")
    c.add_argument("--id",required=True)
    c.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("compare")
    c.add_argument("--root",type=Path,required=True)
    c.add_argument("--output",type=Path,required=True)
    c.add_argument("--scope",choices=("pilot","full"),default="pilot")
    a=ap.parse_args()
    if a.command=="matrix":
        print(json.dumps({"include":catalog(a.scope)},separators=(",",":")))
    elif a.command=="run":
        selected=[x for x in catalog("full") if x["id"]==a.id]
        if len(selected)!=1:raise ValueError("DYNAMIC_RANK_CASE_ID_NOT_FOUND")
        run(selected[0],a.output)
    else:
        compare(a.root,a.output,scope=a.scope)


if __name__=="__main__":
    main()
