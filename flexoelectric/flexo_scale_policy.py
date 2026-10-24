#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict macro/micro routing contract; no silent physics substitutions.

MACRO: existing classical elastic cylinder, no strain-gradient/flexo terms.
MICRO: G3D gradient dielectric with bidirectional flexoelectric coupling.
The MICRO numerical solver is not connected at this stage; inputs can be
validated and saved, but never dispatched as an existing macro solution.
"""
from __future__ import annotations

from math import isfinite
from flexo_gradient_g3d import G3DMaterial

SCALES = ("MACRO", "MICRO")
MICRO_GRADIENT_MODEL = "G3D"


def scale_from_case(case):
    name = str(case.get("scale_model", "MACRO") or "").strip().upper()
    if name not in SCALES:
        raise ValueError("scale_model must be explicitly MACRO or MICRO")
    return name


def micro_material_from_case(case):
    if scale_from_case(case) != "MICRO":
        raise ValueError("micro material requested for a non-MICRO case")
    if (case.get("units") or {}).get("system") != "SI_N_M_PA":
        raise ValueError("MICRO G3D requires SI_N_M_PA; no silent unit conversions")
    if str(case.get("geometry_model", "")).upper() != "GREEN_LAGRANGE":
        raise ValueError("MICRO G3D derives from full Green-Lagrange kinematics")
    micro = case.get("micro")
    if not isinstance(micro, dict):
        raise ValueError("MICRO case requires explicit 'micro' object")
    if str(micro.get("gradient_model", "")).upper() != MICRO_GRADIENT_MODEL:
        raise ValueError("Only full covariant G3D is derived; Gz is a distinct model")
    m = case.get("material")
    if not isinstance(m, dict):
        raise ValueError("MICRO case requires material object")
    keys = ("ell_m", "mu1_c_per_m", "mu2_c_per_m", "kappa_f_per_m")
    for name in keys:
        if name not in micro or micro[name] is None or str(micro[name]).strip() == "":
            raise ValueError(f"micro.{name} is required; no material defaults")
    material = G3DMaterial(
        young_pa=m["E0"], poisson=m["nu0"],
        ell_m=micro["ell_m"], mu1_c_per_m=micro["mu1_c_per_m"],
        mu2_c_per_m=micro["mu2_c_per_m"],
        kappa_f_per_m=micro["kappa_f_per_m"],
    )
    for key in ("rho_kg_m3", "eta_s"):
        if key not in micro or micro[key] is None or str(micro[key]).strip() == "":
            raise ValueError(f"micro.{key} is required; no dynamic defaults")
        v = float(micro[key])
        if not isfinite(v) or (key == "rho_kg_m3" and v <= 0) or (key == "eta_s" and v < 0):
            raise ValueError(f"invalid micro.{key}")
    p = micro.get("potential_order")
    if isinstance(p, bool) or str(p).strip() == "":
        raise ValueError("micro.potential_order must be a nonnegative integer")
    try:
        n = int(p)
    except (TypeError, ValueError) as exc:
        raise ValueError("micro.potential_order must be an integer") from exc
    if n < 0 or str(n) != str(p).strip():
        raise ValueError("micro.potential_order must be a nonnegative integer")
    return material


def numerical_status(case):
    scale = scale_from_case(case)
    if scale == "MACRO":
        return "EXISTING_CLASSICAL_ELASTIC"
    micro_material_from_case(case)
    return "NOT_IMPLEMENTED_COUPLED_G3D"


__all__ = ("SCALES", "MICRO_GRADIENT_MODEL", "scale_from_case",
           "micro_material_from_case", "numerical_status")
