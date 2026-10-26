#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict macro/micro routing contract; no silent physics substitutions.

MACRO: existing classical elastic cylinder, no strain-gradient/flexo terms.
MICRO: selectable full G3D or thickness-only Gz gradient elasticity,
with optional bidirectional flexoelectric coupling.
MICRO calculations require the explicit VARIATIONAL_REFERENCE backend. This
module reports the selected route; full CASE validation belongs to
prepare_reference_case and is not implied by the status string.
"""
from __future__ import annotations

from math import isfinite
from flexo_gradient_g3d import G3DMaterial

SCALES = ("MACRO", "MICRO")
MICRO_GRADIENT_MODEL = "G3D"
MICRO_GRADIENT_MODELS = ("G3D", "GZ")


def flexoelectric_enabled(case):
    """Physical flexoelectric coupling switch; legacy MICRO defaults to ON."""
    micro = case.get("micro")
    if not isinstance(micro, dict):
        raise ValueError("MICRO case requires explicit micro object")
    flag = micro.get("enable_flexoelectric", True)
    if not isinstance(flag, bool):
        raise ValueError("micro.enable_flexoelectric must be true or false")
    return flag


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
    gradient_model=micro.get("gradient_model")
    if not isinstance(gradient_model,str) or gradient_model.strip().upper() not in MICRO_GRADIENT_MODELS:
        raise ValueError("micro.gradient_model must be G3D (full 18 components) or Gz (6 thickness components)")
    gradient_model=gradient_model.strip().upper()
    m = case.get("material")
    if not isinstance(m, dict):
        raise ValueError("MICRO case requires material object")
    if "ell_m" not in micro or str(micro["ell_m"]).strip() == "":
        raise ValueError("micro.ell_m is required; no material defaults")
    flexo_on = flexoelectric_enabled(case)
    if flexo_on:
        for name in ("mu1_c_per_m", "mu2_c_per_m", "kappa_f_per_m"):
            if name not in micro or micro[name] is None or str(micro[name]).strip() == "":
                raise ValueError(f"micro.{name} required when flexoelectricity is ON")
    # When OFF, exact constitutive limit: electric and cross-coupling terms = 0.
    # No fictional dielectric constant is substituted for inactive electricity.
    material = G3DMaterial(
        young_pa=m["E0"], poisson=m["nu0"],
        ell_m=micro["ell_m"],
        mu1_c_per_m=micro["mu1_c_per_m"] if flexo_on else 0.0,
        mu2_c_per_m=micro["mu2_c_per_m"] if flexo_on else 0.0,
        kappa_f_per_m=micro["kappa_f_per_m"] if flexo_on else 0.0,
        gradient_model=gradient_model,
    )
    if material.ell_m <= 0:
        raise ValueError("MICRO G3D/Gz requires ell_m>0; use MACRO to turn gradient elasticity OFF")
    if flexo_on and material.kappa_f_per_m <= 0:
        raise ValueError("kappa must be positive when flexoelectricity is ON")
    for key in ("rho_kg_m3", "eta_s"):
        if key not in micro or micro[key] is None or str(micro[key]).strip() == "":
            raise ValueError(f"micro.{key} is required; no dynamic defaults")
        v = float(micro[key])
        if not isfinite(v) or (key == "rho_kg_m3" and v <= 0) or (key == "eta_s" and v < 0):
            raise ValueError(f"invalid micro.{key}")
    p = micro.get("potential_order") if flexo_on else 0
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
    model=micro_material_from_case(case).gradient_model
    if str(case.get("solver_backend","")).upper()!="VARIATIONAL_REFERENCE":
        return "EXPLICIT_VARIATIONAL_BACKEND_REQUIRED"
    return ("REFERENCE_COUPLED_"+model if flexoelectric_enabled(case)
            else "REFERENCE_GRADIENT_"+model)


__all__ = ("SCALES", "MICRO_GRADIENT_MODEL", "MICRO_GRADIENT_MODELS", "scale_from_case",
           "micro_material_from_case", "flexoelectric_enabled", "numerical_status")

