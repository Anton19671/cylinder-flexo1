"""Canonical physical CASE passport for Newmark versus Gauss-IRK4 MVI.

Only solver/integrator controls are removed. Boundary conditions, material,
geometry, loading, density, grid and integration interval stay in the digest.
A matching passport proves identical *declared inputs*, not that arbitrary
user-provided load callbacks obeyed the declaration.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

_OMIT_SOLVER = frozenset({
    "method", "rank", "reference_mvi", "reference_newton",
    "output_dir", "fdm_grid", "mvi_grid", "grid", "fdm_backend",
})
_OMIT_TIME_ALGORITHM = frozenset({
    "newmark_beta", "newmark_gamma", "residual_tolerance_relative",
    "integrator", "integration_method", "gauss2",
})


def canonical_dynamic_physics(case):
    """Require a complete MVI DYNAMIC CASE and return canonical physics."""
    if not isinstance(case, dict):
        raise ValueError("DYNAMIC_PASSPORT_REQUIRES_CASE_OBJECT")
    if str(case.get("calculation_kind", "")).upper() != "DYNAMIC":
        raise ValueError("DYNAMIC_PASSPORT_REQUIRES_DYNAMIC_CASE")
    if str(case.get("method", "")).upper() != "MVI":
        raise ValueError("DYNAMIC_PASSPORT_REQUIRES_MVI_METHOD")
    raw_grid=case.get("mvi_grid") or case.get("grid")
    if not isinstance(raw_grid, dict):
        raise ValueError("DYNAMIC_PASSPORT_REQUIRES_SPATIAL_GRID")
    nx, ns=raw_grid.get("nx"), raw_grid.get("ns")
    if (type(nx) is not int or type(ns) is not int or nx < 2 or ns < 2):
        raise ValueError("DYNAMIC_PASSPORT_INVALID_SPATIAL_GRID")
    scheme=str(raw_grid.get("operator_scheme", "LEGACY_REFERENCE")).upper()
    if scheme not in (
            "LEGACY_REFERENCE", "SBP21_PERIODIC_ODD", "SBP21_PERIODIC_ODD_CSR",
            "SBP21_FOURIER_ODD", "SBP42_FOURIER_ODD"):
        raise ValueError("DYNAMIC_PASSPORT_INVALID_SPATIAL_OPERATOR")
    dynamics=case.get("dynamics")
    if not isinstance(dynamics, dict):
        raise ValueError("DYNAMIC_PASSPORT_REQUIRES_DYNAMICS")
    n=dynamics.get("n_steps")
    if type(n) is not int or n < 1:
        raise ValueError("DYNAMIC_PASSPORT_INVALID_TIME_GRID")
    for label in ("time_step_s", "density_kg_m3"):
        v=dynamics.get(label)
        if (isinstance(v,bool) or not isinstance(v,(int,float)) or
                not math.isfinite(v) or v<=0):
            raise ValueError("DYNAMIC_PASSPORT_INVALID_"+label.upper())
    t0=dynamics.get("start_time_s",0)
    if (isinstance(t0,bool) or not isinstance(t0,(int,float)) or
            not math.isfinite(t0)):
        raise ValueError("DYNAMIC_PASSPORT_INVALID_START_TIME")
    document=deepcopy({k:v for k,v in case.items() if k not in _OMIT_SOLVER})
    document["_comparison_grid"]={
        "nx":nx,"ns":ns,"operator_scheme":scheme}
    document["dynamics"]={k:v for k,v in dynamics.items()
                          if k not in _OMIT_TIME_ALGORITHM}
    # Enforce JSON portability; no NaN, infinity, hidden numpy objects.
    try:
        packed=json.dumps(document,sort_keys=True,ensure_ascii=False,
                          separators=(",",":"),allow_nan=False)
        return json.loads(packed)
    except (TypeError,ValueError) as exc:
        raise ValueError("DYNAMIC_PASSPORT_NOT_FINITE_JSON") from exc


def dynamic_physics_sha256(case):
    canonical=canonical_dynamic_physics(case)
    value=json.dumps(canonical,sort_keys=True,ensure_ascii=False,
                     separators=(",",":"),allow_nan=False)
    return sha256(value.encode("utf-8")).hexdigest()


__all__=("canonical_dynamic_physics","dynamic_physics_sha256")
