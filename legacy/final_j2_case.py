#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASE parser for the independent incremental J2 material model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from final_j2_incremental import J2Hardening, J2MaterialError, J2Parameters, J2State
from final_j2_finite_strain import FiniteJ2State
import final_j2_history_io as j2io


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise J2MaterialError(f"{name} must be an object")
    return value


def _positive(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise J2MaterialError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(x) or x <= 0.0:
        raise J2MaterialError(f"{name} must be finite and positive")
    return x


def _nonnegative(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise J2MaterialError(f"{name} must be a finite non-negative number") from exc
    if not np.isfinite(x) or x < 0.0:
        raise J2MaterialError(f"{name} must be finite and non-negative")
    return x


def parse_j2_parameters(case: Dict[str, Any]) -> J2Parameters:
    """Build canonical J2 parameters from CASE.j2 and CASE.material."""

    j = _object(case.get("j2"), "CASE.j2")
    material = _object(case.get("material"), "CASE.material")
    E = _positive(material.get("E0"), "material.E0")
    nu = float(material.get("nu0"))
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise J2MaterialError("material.nu0 must satisfy -1 < nu < 0.5")
    kind = str(j.get("hardening_kind", "LINEAR")).upper()
    sigma_y0 = _positive(j.get("sigma_y0"), "j2.sigma_y0")
    H = _nonnegative(j.get("H", 0.0), "j2.H")
    if kind == "TABLE":
        table = j.get("hardening_table")
        if not isinstance(table, list) or len(table) < 2:
            raise J2MaterialError("j2.hardening_table must have at least two [kappa,sigma_y] rows")
        try:
            kappa = [float(row[0]) for row in table]
            sigma_y = [float(row[1]) for row in table]
        except Exception as exc:
            raise J2MaterialError("j2.hardening_table rows must be [kappa,sigma_y]") from exc
        hardening = J2Hardening(
            sigma_y0=sigma_y0, kind="TABLE", H=0.0,
            kappa_table=kappa, sigma_y_table=sigma_y,
        )
    else:
        hardening = J2Hardening(sigma_y0=sigma_y0, kind=kind, H=H)
    try:
        max_local_iterations = int(j.get("max_local_iterations", 48))
    except (TypeError, ValueError) as exc:
        raise J2MaterialError("j2.max_local_iterations must be an integer") from exc
    if max_local_iterations < 4:
        raise J2MaterialError("j2.max_local_iterations must be at least 4")
    return J2Parameters(
        E=E, nu=nu, hardening=hardening,
        yield_tolerance=_nonnegative(j.get("yield_tolerance", 1.0e-10), "j2.yield_tolerance"),
        local_tolerance=_positive(j.get("local_tolerance", 1.0e-12), "j2.local_tolerance"),
        max_local_iterations=max_local_iterations,
    )


def initial_j2_state(
    case: Dict[str, Any],
    case_path: Path,
    *,
    finite: bool,
    shape: Tuple[int, ...],
    quadrature,
    h: float,
):
    """Load an accepted history or create an initially elastic one.

    The source is explicit.  A NPZ may contain only J2 history written by this
    branch; displacement state, GB and arbitrary NPZ files are rejected.
    """

    j = _object(case.get("j2"), "CASE.j2")
    initial = _object(j.get("initial_history", {"source": "ELASTIC_ZERO"}), "j2.initial_history")
    source = str(initial.get("source", "ELASTIC_ZERO")).upper()
    if source == "ELASTIC_ZERO":
        return FiniteJ2State.zeros(shape) if finite else J2State.zeros(shape)
    if source != "NPZ":
        raise J2MaterialError("j2.initial_history.source must be ELASTIC_ZERO or NPZ")
    raw = initial.get("npz")
    if not raw:
        raise J2MaterialError("j2.initial_history.npz is required when source=NPZ")
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(case_path).parent / path
    if finite:
        return j2io.load_finite_history(path, shape=shape, quadrature=quadrature, h=h)
    return j2io.load_small_history(path, shape=shape, quadrature=quadrature, h=h)


__all__ = ["parse_j2_parameters", "initial_j2_state"]
