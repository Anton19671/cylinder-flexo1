"""Explicit STATIC CASE runner for the independent MKR finite-difference solver.

This module never imports MVI, the reference variational spatial assembly,
a time integrator or an electrical/flexoelectric approximation.
Validation is preflight ONLY, with no numerical operator construction.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from independent_mkr.linear_models import StrongModelCase,IndependentStrongModel
from independent_mkr.run_models import (
    METHOD,SPATIAL_GRID,THICKNESS_SCHEME,PHYSICS_SCOPE,
    _outer_radial_field,_save_npz_atomic)

SCHEMA="INDEPENDENT_MKR_STATIC_SEVEN_MODELS_V1"


def load_case(path):
    path=Path(path)
    doc=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc,dict) or set(doc)!={
            "schema","physics","static","output_dir"}:
        raise ValueError("MKR_STATIC_CASE_KEYS_INVALID")
    if doc["schema"]!=SCHEMA:
        raise ValueError("MKR_STATIC_OWN_SCHEMA_REQUIRED")
    try:
        model=StrongModelCase(**doc["physics"])
    except TypeError as exc:
        raise ValueError("MKR_STATIC_PHYSICS_KEYS_INVALID") from exc
    settings=doc["static"]
    if not isinstance(settings,dict) or set(settings)!={"load_factor"}:
        raise ValueError("MKR_STATIC_LOAD_KEYS_INVALID")
    factor=settings["load_factor"]
    if type(factor) not in (int,float) or not np.isfinite(factor) or factor<0:
        raise ValueError("MKR_STATIC_LOAD_FACTOR_INVALID")
    output=doc["output_dir"]
    if not isinstance(output,str) or not output.strip() or output in (".","/"):
        raise ValueError("MKR_STATIC_OUTPUT_DIRECTORY_INVALID")
    directory=Path(output).expanduser()
    if not directory.is_absolute():
        directory=(path.resolve().parent/directory).resolve()
    return model,float(factor),directory,doc


def _save_json_atomic(path,data):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    temp.replace(path)


def run(path,*,validate_only):
    case,factor,output,document=load_case(path)
    if validate_only:
        return {"schema":SCHEMA,"method":METHOD,
                "calculation_kind":"STATIC","validated_only":True,
                "physics":asdict(case),"load_factor":factor,
                "output_dir":str(output),"spatial_grid":SPATIAL_GRID,
                "thickness_scheme":THICKNESS_SCHEME,
                "physics_scope":PHYSICS_SCOPE,
                "scientific_accuracy_certified":False}
    # No dynamic surrogate or reference calculation can enter this route.
    model=IndependentStrongModel(case)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MKR_STATIC_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    record={"schema":SCHEMA,"method":METHOD,
            "calculation_kind":"STATIC","theory":case.theory,
            "solver_module":"independent_mkr.linear_models",
            "status":"IN_PROGRESS","converged":False,
            "spatial_grid":SPATIAL_GRID,
            "thickness_scheme":THICKNESS_SCHEME,
            "physics_scope":PHYSICS_SCOPE,
            "load_factor":factor,
            "scientific_accuracy_certified":False,
            "independent_method":"MKR"}
    summary=output/"MKR_STATIC_SUMMARY.json"
    _save_json_atomic(summary,record)
    try:
        field=model.solve_static(factor)
        expected=(model.n,case.nx,case.ns)
        if field.shape!=expected or not np.isfinite(field).all():
            raise RuntimeError("MKR_STATIC_RESULT_SHAPE_OR_FINITE_INVALID")
        radial=(2 if case.theory in ("KL","FSDT","REDDY")
                else 2*(int(case.theory[-1])+1))
        outer=_outer_radial_field(field,case.theory)
        _save_npz_atomic(output/"MKR_STATIC_FIELD.npz",
                         mechanical=field,radial_mid_m=field[radial],
                         radial_outer_m=outer)
        record.update(status="CONVERGED",converged=True,
                      central_w0_m=float(field[radial,case.nx//2,case.ns//2]),
                      central_outer_w_m=float(outer[case.nx//2,case.ns//2]),
                      maximum_abs_outer_w_m=float(np.max(np.abs(outer))))
        _save_json_atomic(summary,record)
    except Exception as exc:
        (output/"MKR_STATIC_FIELD.npz").unlink(missing_ok=True)
        record.update(status="FAILED",converged=False,
                      error_type=type(exc).__name__,detail=str(exc))
        _save_json_atomic(summary,record)
        raise
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",type=Path,required=True)
    operation=parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--validate-only",action="store_true")
    operation.add_argument("--run",action="store_true")
    options=parser.parse_args()
    print(json.dumps(run(options.case,validate_only=options.validate_only),
                     indent=2))


if __name__=="__main__":
    main()
