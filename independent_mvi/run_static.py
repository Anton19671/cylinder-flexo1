"""Explicit STATIC CASE runner for independent rank-fixed MVI X/Y.

This module never imports MKR, a reference spatial assembler, or
any physical time integrator. Validation performs NO numerical solve.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from independent_mvi.linear_models import VariationalModelCase,IndependentVariationalModel
from independent_mvi.run_models import (
    METHOD,SPATIAL_GRID,THICKNESS_SCHEME,PHYSICS_SCOPE,
    _outer_radial_field,_save_npz_atomic)

SCHEMA="INDEPENDENT_MVI_STATIC_SEVEN_MODELS_V1"


def load_case(path):
    path=Path(path)
    doc=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc,dict) or set(doc)!={
            "schema","physics","static","rank","output_dir"}:
        raise ValueError("MVI_STATIC_CASE_KEYS_INVALID")
    if doc["schema"]!=SCHEMA:
        raise ValueError("MVI_STATIC_OWN_SCHEMA_REQUIRED")
    try:
        model=VariationalModelCase(**doc["physics"])
    except TypeError as exc:
        raise ValueError("MVI_STATIC_PHYSICS_KEYS_INVALID") from exc
    rank=doc["rank"]
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("MVI_STATIC_RANK_INVALID")
    capacity=min(model.nx-(4 if model.theory=="KL" else 2),model.ns)
    if rank>capacity:
        raise ValueError("MVI_STATIC_RANK_EXCEEDS_CLAMPED_CAPACITY")
    settings=doc["static"]
    if not isinstance(settings,dict) or set(settings)!={
            "load_factor","max_sweeps"}:
        raise ValueError("MVI_STATIC_LOAD_KEYS_INVALID")
    factor=settings["load_factor"]
    sweeps=settings["max_sweeps"]
    if type(factor) not in (int,float) or not np.isfinite(factor) or factor<0:
        raise ValueError("MVI_STATIC_LOAD_FACTOR_INVALID")
    if type(sweeps) is not int or not 1<=sweeps<=200:
        raise ValueError("MVI_STATIC_MAX_SWEEPS_INVALID")
    output=doc["output_dir"]
    if not isinstance(output,str) or not output.strip() or output in (".","/"):
        raise ValueError("MVI_STATIC_OUTPUT_DIRECTORY_INVALID")
    directory=Path(output).expanduser()
    if not directory.is_absolute():
        directory=(path.resolve().parent/directory).resolve()
    return model,rank,float(factor),sweeps,directory,doc


def _save_json_atomic(path,data):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    temp.replace(path)


def run(path,*,validate_only):
    case,rank,factor,sweeps,output,document=load_case(path)
    if validate_only:
        return {"schema":SCHEMA,"method":METHOD,
                "calculation_kind":"STATIC","validated_only":True,
                "rank":rank,"physics":asdict(case),
                "load_factor":factor,"max_sweeps":sweeps,
                "output_dir":str(output),"spatial_grid":SPATIAL_GRID,
                "thickness_scheme":THICKNESS_SCHEME,
                "physics_scope":PHYSICS_SCOPE,
                "scientific_accuracy_certified":False}
    model=IndependentVariationalModel(case,rank)
    output.mkdir(parents=True,exist_ok=False)
    (output/"MVI_STATIC_OWN_CASE.json").write_text(
        json.dumps(document,indent=2)+"\n",encoding="utf-8")
    record={"schema":SCHEMA,"method":METHOD,
            "calculation_kind":"STATIC","theory":case.theory,
            "solver_module":"independent_mvi.linear_models",
            "rank":rank,"status":"IN_PROGRESS","converged":False,
            "spatial_grid":SPATIAL_GRID,
            "thickness_scheme":THICKNESS_SCHEME,
            "physics_scope":PHYSICS_SCOPE,
            "load_factor":factor,"max_sweeps":sweeps,
            "scientific_accuracy_certified":False,
            "independent_method":"MVI"}
    summary=output/"MVI_STATIC_SUMMARY.json"
    _save_json_atomic(summary,record)
    filename=f"MVI_R{rank}_STATIC"
    try:
        field,factors,history=model.solve_static(
            factor,max_sweeps=sweeps)
        expected=(model.n,case.nx,case.ns)
        if field.shape!=expected or not np.isfinite(field).all():
            raise RuntimeError("MVI_STATIC_RESULT_SHAPE_OR_FINITE_INVALID")
        X,Y=(np.asarray(a,float) for a in factors)
        if (X.shape!=(model.n,rank,case.nx) or
                Y.shape!=(model.n,rank,case.ns) or
                not np.isfinite(X).all() or not np.isfinite(Y).all()):
            raise RuntimeError("MVI_STATIC_FACTORS_INVALID")
        reconstructed=np.einsum("krx,krs->kxs",X,Y,optimize=True)
        if not np.allclose(field,reconstructed,rtol=2.e-8,atol=0.):
            raise RuntimeError("MVI_STATIC_FACTORS_INCONSISTENT")
        radial=(2 if case.theory in ("KL","FSDT","REDDY")
                else 2*(int(case.theory[-1])+1))
        outer=_outer_radial_field(field,case.theory)
        _save_npz_atomic(output/(filename+"_FIELD.npz"),
                         mechanical=field,radial_mid_m=field[radial],
                         radial_outer_m=outer)
        _save_npz_atomic(output/(filename+"_FACTORS.npz"),
                         x_profiles=X,y_profiles=Y)
        record.update(status="CONVERGED",converged=True,
                      completed_xy_halfsteps=len(history),
                      central_w0_m=float(field[radial,case.nx//2,case.ns//2]),
                      central_outer_w_m=float(outer[case.nx//2,case.ns//2]),
                      maximum_abs_outer_w_m=float(np.max(np.abs(outer))))
        _save_json_atomic(summary,record)
    except Exception as exc:
        for filename_to_remove in output.glob(filename+"_*.npz"):
            filename_to_remove.unlink(missing_ok=True)
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
