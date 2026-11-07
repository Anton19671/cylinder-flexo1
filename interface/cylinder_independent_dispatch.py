"""Strict own-method native CASE dispatcher for independent linear solvers.

Supported: KL/FSDT/Reddy/CUF TE1..4, LINEAR MACRO HOOKE, electrical OFF,
Bathe/Newmark/Gauss IRK4. Independent MKR and MVI own different model
assembly, solver, documents and output directories. This code does NOT
call reference variational assembly or the opposite solver.

Unsupported physics is rejected. No fallback, hidden approximation,
conversion of nonlinear physics or accidental physical solve on validation.
"""
from __future__ import annotations
from pathlib import Path

SCHEMAS={
    "INDEPENDENT_MKR_SEVEN_MODELS_V1":"INDEPENDENT_MKR",
    "INDEPENDENT_MVI_SEVEN_MODELS_V1":"INDEPENDENT_MVI",
}


def independent_method(case):
    if not isinstance(case,dict):
        raise ValueError("INDEPENDENT_CASE_EXPECTED_OBJECT")
    schema=case.get("schema")
    if schema not in SCHEMAS:
        raise ValueError("INDEPENDENT_CASE_UNSUPPORTED_SCHEMA")
    if case.get("solver_backend") is not None:
        raise ValueError("INDEPENDENT_NATIVE_CASE_MUST_NOT_MASK_BACKEND")
    return SCHEMAS[schema]


def _load_native(path):
    """Only explicit independent JSON, never an implicit reference bridge."""
    from cylinder_case_io import load_case_file
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!=".json":
        raise ValueError("INDEPENDENT_RUNNER_REQUIRES_NATIVE_JSON")
    case=load_case_file(path)
    return path,case,independent_method(case)


def _own_runner(method):
    if method=="INDEPENDENT_MKR":
        from independent_mkr import run_models
        return run_models
    if method=="INDEPENDENT_MVI":
        from independent_mvi import run_models
        return run_models
    raise ValueError("INDEPENDENT_UNKNOWN_METHOD_NO_FALLBACK")


def validate_independent_case(case_file):
    path,case,method=_load_native(case_file)
    own=_own_runner(method)
    result=own.run(path,validate_only=True)
    if result.get("validated_only") is not True:
        raise RuntimeError("INDEPENDENT_CASE_VALIDATION_MUST_NOT_SOLVE")
    if result["schema"]!=case["schema"]:
        raise RuntimeError("INDEPENDENT_CASE_METHOD_SCHEMA_MISMATCH")
    return dict(result,solver_backend=method,calculation_kind="DYNAMIC",
                theory=result["physics"]["theory"],case=str(path),
                valid=True,physical_steps_computed=0,
                no_reference_fallback=True)


def run_independent_case(case_file):
    path,case,method=_load_native(case_file)
    # The only branch that advances any numerical step. One user-requested
    # method-specific implementation is selected from the validated schema.
    own=_own_runner(method)
    own.load_case(path)
    result=own.run(path,validate_only=False)
    if result.get("method")!=own.METHOD:
        raise RuntimeError("INDEPENDENT_RUNNER_METHOD_MISMATCH")
    return dict(result,solver_backend=method,no_reference_fallback=True)


__all__=("SCHEMAS","independent_method",
         "validate_independent_case","run_independent_case")
