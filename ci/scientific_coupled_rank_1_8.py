"""Real TE1 flexoelectric patch: exactly one FDM and MVI ranks 1..8.

Rank-one results on 5x5 cannot be extrapolated to rank 8. This campaign
uses the *same* 8x8 nodal grid, exact constitutive/face/support/electrode
settings, same nonlinear GL model and per-rank accepted NPZ states.

8x8 is a deliberately bounded rank-capacity experiment at the 512-DOF
dense substep safety limit; its even angular grid does NOT certify
periodic/Nyquist grid convergence or physical accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import traceback

from cylinder_paths import activate
activate()

from cylinder_final_stage11a_dispatcher import run_final_case
from cylinder_reference_case import prepare_reference_case
from ci.scientific_coupled_convergence import (
    catalog as base_catalog,case_from_entry,physical_identity,
    digest,verify,
)
from ci.scientific_coupled_field_compare import (
    matching_fields,_trusted_archived_entry,
)
import numpy as np

STUDY="TE1_NONAX_RANK_1_8"
NX=8
NS=8
REFERENCE_ID="TE1_NONAX_8X8_FDM"
RANKS=tuple(range(1,9))


def catalog():
    template=next(x for x in base_catalog("full")
                  if x["study"]=="TE1_RANK" and x["method"]=="FDM")
    entries=[]
    for rank in (0,)+RANKS:
        is_fdm=rank==0
        entry=dict(template)
        entry.update(
            id=(REFERENCE_ID if is_fdm
                else f"TE1_NONAX_8X8_MVI_R{rank}"),
            study=STUDY,
            variant="FDM" if is_fdm else f"MVI_{rank}",
            method="FDM" if is_fdm else "MVI",
            rank=max(1,rank),
            nx=NX,ns=NS,
            steps=1,patch=True,
        )
        entries.append(entry)
    assert len(entries)==9 and len({x["id"] for x in entries})==9
    return entries


def build_case(entry,folder):
    case=case_from_entry(entry,folder)
    # A rank-8 halfstep can span 8*(6 mechanical+2 electric)*8 = 512
    # coefficient directions. Never silently relax the configured core cap.
    case["reference_newton"]["max_free_dofs"]=512
    if entry["method"]=="MVI":
        case["reference_mvi"]["max_sweeps"]=55
    return case


def solve(entry,folder):
    folder=Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("RANK_SERIES_OUTPUT_NOT_EMPTY")
    folder.mkdir(parents=True,exist_ok=True)
    case=build_case(entry,folder)
    case_file=folder/"CASE.json"
    case_file.write_text(json.dumps(case,indent=2)+"\n",encoding="utf-8")
    record=dict(
        case_id=entry["id"],study=STUDY,variant=entry["variant"],
        method=entry["method"],rank=entry["rank"] if entry["method"]=="MVI" else None,
        grid=[NX,NS],
        physics_sha256=physical_identity(case),
        case_sha256=digest(case_file),
        git_sha=os.getenv("GITHUB_SHA","UNKNOWN"),
        status="IN_PROGRESS",
        rank_capacity_only=True,
        even_periodic_grid_not_validated=True,
        external_accuracy_certified=False,
        independent_spatial_methods_certified=False,
    )
    marker=folder/"CONVERGENCE_EVIDENCE.json"
    def save():
        marker.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",
                          encoding="utf-8")
    save()
    try:
        prepared=prepare_reference_case(case,base_dir=folder)
        if prepared.method!=entry["method"] or prepared.calculation_kind!="STATIC":
            raise ValueError("RANK_SERIES_PREPARED_DISPATCH_MISMATCH")
        if prepared.problem.functional.theory!="CUF_TE1":
            raise ValueError("RANK_SERIES_WRONG_THEORY")
        if not prepared.problem.functional.electrical_on:
            raise ValueError("RANK_SERIES_WRONG_ELECTRIC_SWITCH")
        if prepared.electrical is None or prepared.electrical.shape!=(2,NX,NS):
            raise ValueError("RANK_SERIES_WRONG_ELECTRIC_ORDER")
        if entry["method"]=="MVI" and prepared.mvi_settings.rank!=entry["rank"]:
            raise ValueError("RANK_SERIES_WRONG_MVI_RANK")
        print("REAL_CUF_TE1_FLEXO_RANK_BEGIN",entry["id"],flush=True)
        run_final_case(case_file)
        verified=verify(case,entry,folder)
        record.update(verified)
        record["status"]="ACCEPTED"
        save()
        print("REAL_CUF_TE1_FLEXO_RANK_ACCEPTED",entry["id"],
              record["solution_class"],flush=True)
    except BaseException as exc:
        record.update(status="FAILED",failure_type=type(exc).__name__,
                      failure_message=str(exc))
        save()
        (folder/"TRACEBACK.txt").write_text(traceback.format_exc(),
                                             encoding="utf-8")
        raise


def compare(root,out):
    """Compare every available genuine rank; record failures without hiding."""
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    entries=catalog()
    rows=[]
    stored={}
    for entry in entries:
        try:
            record,state=_trusted_archived_entry(root,entry)
            if record.get("study")!=STUDY or record.get("grid")!=[NX,NS]:
                raise ValueError("RANK_SERIES_STUDY_OR_GRID_PASSPORT_MISMATCH")
            if record.get("even_periodic_grid_not_validated") is not True:
                raise ValueError("RANK_SERIES_GRID_LIMITATION_UNDOCUMENTED")
            rows.append(dict(id=entry["id"],method=entry["method"],
                             rank=entry["rank"] if entry["method"]=="MVI" else None,
                             status="ACCEPTED",
                             solution_class=record.get("solution_class"),
                             full_space_stationarity_confirmed=record.get(
                                 "full_space_stationarity_confirmed") is True))
            stored[entry["id"]]=(record,state)
        except (ValueError,FileNotFoundError,KeyError) as exc:
            rows.append(dict(id=entry["id"],method=entry["method"],
                             rank=entry["rank"] if entry["method"]=="MVI" else None,
                             status="MISSING_FAILED_OR_INVALID",
                             reason=str(exc)))
    references=stored.get(REFERENCE_ID)
    field_tables=[]
    if references is not None:
        baseline,base_state=references
        for entry in entries[1:]:
            pair=stored.get(entry["id"])
            if pair is None:
                continue
            record,state=pair
            if (record.get("physics_sha256")!=baseline.get("physics_sha256")
                    or record.get("git_sha")!=baseline.get("git_sha")):
                raise ValueError("RANK_SERIES_PHYSICS_OR_CODE_REVISION_MISMATCH")
            with np.load(base_state,allow_pickle=False) as fdm, np.load(
                    state,allow_pickle=False) as mvi:
                metrics=matching_fields(fdm,mvi)
            field_tables.append(dict(
                case_id=entry["id"],rank=entry["rank"],
                solution_class=record["solution_class"],
                full_space_stationarity_confirmed=record.get(
                    "full_space_stationarity_confirmed") is True,
                metrics=metrics))
    report={
        "study":STUDY,"grid":[NX,NS],
        "theory":"CUF_TE1","gradient_model":"Gz",
        "electric_thickness_order":1,"flexoelectric_enabled":True,
        "reference":REFERENCE_ID,
        "source_code_sha256":references[0]["git_sha"] if references else None,
        "physical_model_sha256":(
            references[0]["physics_sha256"] if references else None),
        "cases":rows,
        "relative_field_differences_to_same_grid_fdm":field_tables,
        "all_nine_states_accepted":len(stored)==len(entries),
        "all_eight_ranks_compared":len(field_tables)==8,
        "all_ranks_fully_stationary":(
            len(field_tables)==8 and all(row["full_space_stationarity_confirmed"]
                                        for row in field_tables)),
        "even_angular_grid_nyquist_accuracy_certified":False,
        "independent_coupled_spatial_discretizations_certified":False,
        "scientific_accuracy_certified":False,
        "notes":[
            "MVI rank constraints are distinct from full-space physical accuracy",
            "Relative L2 differences are referenced to FDM, not an independent benchmark",
            "Rank 8 spans the algebraic 8x8 grid; spatial refinement is a separate gate",
        ],
    }
    (out/"SCIENTIFIC_MVI_RANKS_1_TO_8.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (out/"SCIENTIFIC_MVI_RANKS_1_TO_8_FIELDS.csv").open(
            "w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=[
            "rank","field","rank_status","full_space_stationarity_confirmed",
            "relative_l2","relative_to_reference_peak",
            "max_abs_difference","reference_peak_abs"])
        writer.writeheader()
        for row in field_tables:
            for field,item in row["metrics"].items():
                writer.writerow({
                    "rank":row["rank"],
                    "field":field,
                    "rank_status":row["solution_class"],
                    "full_space_stationarity_confirmed":row[
                        "full_space_stationarity_confirmed"],
                    **{k:item[k] for k in (
                        "relative_l2","relative_to_reference_peak",
                        "max_abs_difference","reference_peak_abs")},
                })
    if not report["all_eight_ranks_compared"]:
        raise RuntimeError("RANK_SERIES_INCOMPLETE_REAL_CASES")
    print("REAL_MVI_RANKS_ONE_THROUGH_EIGHT_COMPARED",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("matrix")
    a=sub.add_parser("run")
    a.add_argument("--id",required=True)
    a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("compare")
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="matrix":
        print(json.dumps({"include":catalog()},separators=(",",":")))
    elif args.command=="run":
        selected=[e for e in catalog() if e["id"]==args.id]
        if len(selected)!=1:
            raise ValueError("RANK_SERIES_UNKNOWN_CASE")
        solve(selected[0],args.output)
    else:
        compare(args.root,args.output)


if __name__=="__main__":
    main()
