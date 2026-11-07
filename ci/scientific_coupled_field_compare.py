"""Strict field-by-field FDM versus fixed-rank MVI comparison on REAL NPZ.

This is a postprocessor. It NEVER solves, modifies or regularizes a
physical field. It does not claim independent spatial discretization;
both coupled variants currently share the variational reference kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from scientific_coupled_convergence import catalog, digest, physical_identity


REQUIRED_FIELDS=(
    "mechanical",
    "electrical",
    "face_outer_radial_displacement_m",
    "face_outer_electric_outward_induction_c_per_m2",
    "stress6_pa",
    "hyperstress6x3_pa_m",
    "induction3_c_per_m2",
)
GRID_KEYS=("x_m","s_m","z_m")


def metric(reference,candidate):
    """Relative norms use only the nonzero actual reference field."""
    a=np.asarray(reference,float)
    b=np.asarray(candidate,float)
    if a.shape!=b.shape or not (
            np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("SCIENTIFIC_FIELD_SHAPE_OR_FINITE_MISMATCH")
    if a.ndim==0:
        raise ValueError("SCIENTIFIC_FIELD_REQUIRES_ARRAY")
    difference=b-a
    peak=float(np.max(np.abs(a),initial=0))
    peak_delta=float(np.max(np.abs(difference),initial=0))
    reference_norm=float(np.linalg.norm(a.ravel()))
    difference_norm=float(np.linalg.norm(difference.ravel()))
    if not all(np.isfinite(x) for x in
               (peak,peak_delta,reference_norm,difference_norm)):
        raise ValueError("SCIENTIFIC_FIELD_NORM_NOT_FINITE")
    return {
        "shape":list(a.shape),
        "reference_peak_abs":peak,
        "max_abs_difference":peak_delta,
        "reference_l2":reference_norm,
        "difference_l2":difference_norm,
        "relative_l2":(
            difference_norm/reference_norm if reference_norm>0 else None),
        "relative_to_reference_peak":(
            peak_delta/peak if peak>0 else None),
        "reference_identically_zero":reference_norm==0.0,
    }


def matching_fields(reference,candidate):
    """Reject missing physics or incompatible physical coordinate vectors."""
    for key in GRID_KEYS:
        if key not in reference or key not in candidate:
            raise ValueError("SCIENTIFIC_FIELD_GRID_COORDINATE_MISSING_"+key)
        a,b=np.asarray(reference[key]),np.asarray(candidate[key])
        if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=1e-13):
            raise ValueError("SCIENTIFIC_FIELD_DIFFERENT_GRID_"+key)
    result={}
    for field in REQUIRED_FIELDS:
        if field not in reference or field not in candidate:
            raise ValueError("SCIENTIFIC_FIELD_MISSING_"+field)
        result[field]=metric(reference[field],candidate[field])
    return result


def _trusted_archived_entry(root,entry):
    """Verify actual CASE, NPZ and source-code evidence, not JSON claims."""
    matches=[]
    for evidence in root.rglob("CONVERGENCE_EVIDENCE.json"):
        try:
            content=json.loads(evidence.read_text(encoding="utf-8"))
        except (ValueError,OSError):
            continue
        if content.get("case_id")==entry["id"]:
            matches.append((evidence,content))
    if len(matches)!=1:
        raise ValueError("SCIENTIFIC_FIELD_EVIDENCE_MISSING_OR_DUPLICATED_"+
                         entry["id"])
    path,record=matches[0]
    if record.get("status")!="ACCEPTED":
        raise ValueError("SCIENTIFIC_FIELD_PHYSICAL_CASE_NOT_ACCEPTED_"+
                         entry["id"])
    folder=path.parent
    file=folder/"CASE.json"
    if (not file.is_file() or
            record.get("case_sha256")!=digest(file)):
        raise ValueError("SCIENTIFIC_FIELD_CASE_HASH_INCONSISTENT")
    case=json.loads(file.read_text(encoding="utf-8"))
    if (record.get("physics_sha256")!=physical_identity(case) or
            case.get("method")!=entry["method"] or
            record.get("method")!=entry["method"]):
        raise ValueError("SCIENTIFIC_FIELD_PHYSICS_OR_METHOD_MISMATCH")
    grid=case.get("mvi_grid" if entry["method"]=="MVI" else "fdm_grid")
    if not isinstance(grid,dict) or (
        grid.get("nx"),grid.get("ns"))!=(entry["nx"],entry["ns"]):
        raise ValueError("SCIENTIFIC_FIELD_GRID_PASSPORT_MISMATCH")
    if entry["method"]=="MVI":
        if case.get("rank")!=entry["rank"]:
            raise ValueError("SCIENTIFIC_FIELD_RANK_PASSPORT_MISMATCH")
        if record.get("solution_class") not in (
                "RANK_CONSTRAINED_ONLY","FULL_SPACE_STATIONARY"):
            raise ValueError("SCIENTIFIC_FIELD_RANK_CERTIFICATE_MISSING")
    name=record.get("state_file")
    if (not isinstance(name,str) or Path(name).name!=name or
            not name.endswith(".npz")):
        raise ValueError("SCIENTIFIC_FIELD_ACCEPTED_STATE_NAME_INVALID")
    state=folder/"RESULTS"/name
    if not state.is_file() or record.get("state_sha256")!=digest(state):
        raise ValueError("SCIENTIFIC_FIELD_ACCEPTED_STATE_HASH_INCONSISTENT")
    return record,state


def compare_rank_fields(root,out):
    """Compare rank 1..3 against the accepted *same-grid* full FDM state."""
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    candidates=[e for e in catalog("full") if e["study"]=="TE1_RANK"]
    base=next(e for e in candidates if e["method"]=="FDM")
    fdm,fdm_state=_trusted_archived_entry(root,base)
    ranks=[]
    for entry in sorted((e for e in candidates if e["method"]=="MVI"),
                        key=lambda e:e["rank"]):
        # The pilot has only rank 1; higher ranks are optional physical runs.
        evidences=list(root.rglob(
            entry["id"]+"/CONVERGENCE_EVIDENCE.json"))
        if not evidences:
            continue
        record,state=_trusted_archived_entry(root,entry)
        if (record.get("physics_sha256")!=fdm.get("physics_sha256") or
                record.get("git_sha")!=fdm.get("git_sha")):
            raise ValueError("SCIENTIFIC_FIELD_MISMATCHED_PHYSICS_OR_REVISION")
        with np.load(fdm_state,allow_pickle=False) as reference, np.load(
                state,allow_pickle=False) as candidate:
            result=matching_fields(reference,candidate)
        ranks.append(dict(
            case_id=entry["id"],rank=entry["rank"],
            solution_class=record["solution_class"],
            full_space_stationarity_confirmed=record.get(
                "full_space_stationarity_confirmed") is True,
            measured_field_differences=result))
    if not ranks:
        raise ValueError("SCIENTIFIC_FIELD_NO_MVI_RANK_STATE_AVAILABLE")
    report={
        "study":"TE1_RANK",
        "fdm_case_id":base["id"],
        "physical_model_sha256":fdm["physics_sha256"],
        "git_sha":fdm["git_sha"],
        "same_grid_and_physics_verified":True,
        "all_ranks_full_space_stationary":all(
            x["full_space_stationarity_confirmed"] for x in ranks),
        "independent_spatial_discretizations_certified":False,
        "published_accuracy_certified":False,
        "ranks":ranks,
    }
    (out/"SCIENTIFIC_RANK_FULL_FIELDS.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (out/"SCIENTIFIC_RANK_FULL_FIELDS.csv").open(
            "w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=(
            "rank","field","solution_class",
            "reference_peak_abs","max_abs_difference",
            "relative_l2","relative_to_reference_peak",
            "reference_identically_zero"))
        writer.writeheader()
        for entry in ranks:
            for field,measured in entry["measured_field_differences"].items():
                writer.writerow(dict(
                    rank=entry["rank"],field=field,
                    solution_class=entry["solution_class"],
                    **{k:measured[k] for k in (
                        "reference_peak_abs","max_abs_difference",
                        "relative_l2","relative_to_reference_peak",
                        "reference_identically_zero")}))
    print("SCIENTIFIC_FIELD_COMPARISON_REAL_RANKS",
          len(ranks),"FULL_QUALIFIED",report["all_ranks_full_space_stationary"],
          flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    arguments=parser.parse_args()
    compare_rank_fields(arguments.root,arguments.output)


if __name__=="__main__":
    main()
