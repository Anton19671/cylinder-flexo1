#!/usr/bin/env python3
"""Only coupled elastic G3D/Gz cylindrical-shell calculations are supported."""
from __future__ import annotations
import json
from pathlib import Path
from cylinder_case_io import load_case_file
from cylinder_reference_case import (
    BACKEND, prepare_reference_case, run_reference_case, run_reference_mvi_series
)
THEORIES=("KL","FSDT","REDDY","CUF")
CUF_MODELS=("TE1","TE2","TE3","TE4")
METHODS=("FDM","MVI")
MODES=("NONLINEAR_ELASTIC",)
class FinalCaseError(ValueError):
    pass
def _load_case(case_json):
    path=Path(case_json).expanduser().resolve()
    try:
        case=load_case_file(path)
    except (OSError,ValueError,RuntimeError) as exc:
        raise FinalCaseError(f"Cannot read CASE: {path}: {exc}") from exc
    if not isinstance(case,dict):
        raise FinalCaseError("CASE must be a JSON object")
    if str(case.get("solver_backend","")).upper()!=BACKEND:
        raise FinalCaseError(
            "Only VARIATIONAL_REFERENCE is supported; legacy CASE is not converted")
    return path,case
def validate_final_case(case_json,*,dry_run=True):
    path,case=_load_case(case_json)
    if str(case.get("calculation_kind","")).upper()=="MODAL":
        from cylinder_modal_reference import prepare_modal_case
        return dict(valid=True,solver_backend=BACKEND,case=str(path),
            **prepare_modal_case(case,base_dir=path.parent),material_model="HOOKE")
    validated=prepare_reference_case(case,base_dir=path.parent)
    f=validated.problem.functional
    return dict(valid=True,solver_backend=BACKEND,case=str(path),
        method=validated.method,theory=f.theory,
        calculation_kind=validated.calculation_kind,
        gradient_model=f.material.gradient_model_label,material_model="HOOKE")
def run_final_case(case_json):
    path,case=_load_case(case_json)
    if str(case.get("calculation_kind","")).upper()=="MODAL":
        from cylinder_modal_reference import run_modal_case
        return run_modal_case(case,path)
    return run_reference_case(case,path)
def run_mvi_rank_series(case_json,ranks=range(1,9)):
    path,case=_load_case(case_json)
    if str(case.get("method","")).upper()!="MVI":
        raise FinalCaseError("MVI rank series requires method=MVI")
    return run_reference_mvi_series(case,path,ranks=ranks)
__all__=("THEORIES","CUF_MODELS","METHODS","MODES","FinalCaseError",
         "validate_final_case","run_final_case","run_mvi_rank_series")
