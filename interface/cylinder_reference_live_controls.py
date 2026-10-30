"""Lossless, UI-independent edits for active variational CASE controls.

The main-window selectors create an in-memory draft. The saved CASE is
immutable until an explicit Save action; a Run action uses a separately
saved launch CASE. No numerical or physical calculation is done here.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite


def _case(original):
    if not isinstance(original, dict) or original.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("Select a saved VARIATIONAL_REFERENCE CASE")
    return deepcopy(original)


def _count(value, name, lower=1):
    if isinstance(value, bool) or str(value).strip() == "":
        raise ValueError(f"{name}: enter an integer >= {lower}")
    try:
        result = int(str(value).strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: enter an integer >= {lower}") from exc
    if str(result) != str(value).strip() or result < lower:
        raise ValueError(f"{name}: enter an integer >= {lower}")
    return result


def selected_grid_case(original, *, nx, ns, n_gauss, fdm_backend=None,
                       operator_scheme=None):
    """Update exactly the active spatial method grid and thickness quadrature."""
    draft = _case(original)
    nx = _count(nx, "nx", 4)
    ns = _count(ns, "ns", 4)
    ng = _count(n_gauss, "n_gauss", 2)
    if ng > 32:
        raise ValueError("n_gauss must be <= 32")
    method = str(draft.get("method", "")).upper()
    if method not in ("FDM", "MVI"):
        raise ValueError("Unknown spatial method")
    grid_key = "mvi_grid" if method == "MVI" else "fdm_grid"
    grid = deepcopy(draft.get(grid_key) or draft.get("grid") or {})
    scheme = str(operator_scheme or grid.get("operator_scheme", "LEGACY_REFERENCE")).upper()
    allowed = ("LEGACY_REFERENCE", "SBP21_PERIODIC_ODD",
               "SBP21_PERIODIC_ODD_CSR", "SBP21_FOURIER_ODD",
               "SBP42_FOURIER_ODD", "SBP42_FOURIER_ODD_CSR")
    if scheme not in allowed:
        raise ValueError("Unknown spatial operator scheme")
    if scheme in ("SBP21_PERIODIC_ODD", "SBP21_PERIODIC_ODD_CSR",
                  "SBP21_FOURIER_ODD", "SBP42_FOURIER_ODD", "SBP42_FOURIER_ODD_CSR") and ns % 2 != 1:
        raise ValueError("SBP21 and Fourier periodic grids require odd ns")
    backend = str(fdm_backend or draft.get("fdm_backend", "DENSE_REFERENCE")).upper()
    if method == "MVI":
        if backend != "DENSE_REFERENCE" or scheme in ("SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR"):
            raise ValueError("MVI uses a separate dense Y/X formulation, not matrix-free FDM")
        if _count(draft.get("rank"), "MVI rank") > min(nx, ns):
            raise ValueError("MVI rank exceeds the selected grid")
    elif backend == "MATRIX_FREE_GMRES":
        if scheme not in ("SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR"):
            raise ValueError("MATRIX_FREE_GMRES needs SBP21_PERIODIC_ODD_CSR")
        if (str(draft.get("calculation_kind", "")).upper() == "MODAL"
                and str(draft.get("scale_model", "")).upper() != "MACRO"):
            raise ValueError("Large-grid MODAL FDM currently supports MACRO only")
    elif backend != "DENSE_REFERENCE" or scheme in ("SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR"):
        raise ValueError("DENSE_REFERENCE cannot use a CSR operator scheme")
    grid.update(nx=nx, ns=ns, operator_scheme=scheme)
    draft.pop("fdm_grid", None)
    draft.pop("mvi_grid", None)
    draft.pop("grid", None)
    draft[grid_key] = grid
    draft["fdm_backend"] = backend
    draft["quadrature"] = {"type": "GAUSS", "n_gauss": ng}
    options = deepcopy(draft.get("reference_newton") or {})
    if backend == "MATRIX_FREE_GMRES":
        options.pop("max_free_dofs", None)
        options.pop("min_reciprocal_condition", None)
        # Only numerical controls of the same already selected method.
        options.setdefault("max_total_dofs", 100_000)
        options.setdefault("max_surface_nodes", 8192)
        options.setdefault("max_constraint_rows", 20_000)
    else:
        for name in (
            "max_total_dofs", "max_surface_nodes", "max_constraint_rows",
            "krylov_rtol", "krylov_restart", "krylov_max_matvecs",
            "krylov_preconditioner", "ilu_drop_tolerance", "ilu_fill_factor",
            "ilu_max_surface_nodes"):
            options.pop(name, None)
        options.setdefault("max_free_dofs", 512)
    draft["reference_newton"] = options
    return draft


_MICRO_FIELDS = {
    "micro_ell_m": "ell_m",
    "micro_mu1": "mu1_c_per_m",
    "micro_mu2": "mu2_c_per_m",
    "micro_kappa": "kappa_f_per_m",
    "micro_rho": "rho_kg_m3",
    "micro_eta": "eta_s",
    "micro_potential_order": "potential_order",
}


def selected_micro_case(original, *, scale_model, gradient_model="G3D",
                        flexo=False, values=None):
    """Preserve physical MICRO input instead of inventing material coefficients.

    A MACRO->MICRO draft may be incomplete: it must be filled in and
    preflight-validated before launch. OFF is a constitutive switch that
    retains past coupling inputs separately without using them in physics.
    """
    draft = _case(original)
    scale = str(scale_model).upper()
    if scale not in ("MACRO", "MICRO"):
        raise ValueError("Scale must be MACRO or MICRO")
    model = str(gradient_model).strip().upper()
    if model not in ("G3D", "GZ"):
        raise ValueError("Gradient model must be G3D or Gz")
    if type(flexo) is not bool:
        raise ValueError("Flexoelectric switch must be boolean")
    micro = deepcopy(draft.get("micro") or {})
    current_on = bool(micro.get("enable_flexoelectric", False))
    electric = (draft.get("boundary_conditions") or {}).get("variational", {}).get("electrical")
    if scale == "MACRO" or not flexo:
        if current_on:
            remembered = {key: micro.get(key) for key in
                          ("mu1_c_per_m", "mu2_c_per_m", "kappa_f_per_m", "potential_order")
                          if key in micro}
            if remembered:
                micro["editor_saved_flexo"] = remembered
        if electric:
            micro["editor_saved_electrical_bc"] = deepcopy(electric)
            draft["boundary_conditions"]["variational"]["electrical"] = {}
    if scale == "MACRO":
        micro["enable_flexoelectric"] = False
        draft["micro"] = micro
        draft["scale_model"] = "MACRO"
        return draft
    draft["scale_model"] = "MICRO"
    # Full Green--Lagrange is a physical prerequisite in MICRO, not
    # a free choice of nonlinear material model.
    draft["geometry_model"] = "GREEN_LAGRANGE"
    draft["mode"] = ("LINEAR_ELASTIC" if str(draft.get("calculation_kind", "")).upper() == "MODAL"
                     else "NONLINEAR_ELASTIC")
    micro["gradient_model"] = "Gz" if model == "GZ" else "G3D"
    micro["enable_flexoelectric"] = flexo
    if not current_on and flexo:
        remembered = micro.get("editor_saved_flexo") or {}
        for key in ("mu1_c_per_m", "mu2_c_per_m", "kappa_f_per_m", "potential_order"):
            if key in remembered:
                micro[key] = remembered[key]
    if "rho_kg_m3" not in micro:
        if isinstance(draft.get("modal"), dict):
            micro["rho_kg_m3"] = draft["modal"].get("density_kg_m3")
        elif isinstance(draft.get("dynamics"), dict):
            micro["rho_kg_m3"] = draft["dynamics"].get("density_kg_m3")
    micro.setdefault("eta_s", 0.0)
    if not flexo:
        micro["potential_order"] = 0
    elif "potential_order" not in micro:
        # Incomplete coupled CASE: user must choose potential order explicitly.
        micro.pop("potential_order", None)
    for input_name, physical_key in (values or {}).items():
        if input_name not in _MICRO_FIELDS:
            raise ValueError("Unknown MICRO parameter: " + input_name)
        key = _MICRO_FIELDS[input_name]
        raw = str(physical_key).strip().replace(",", ".")
        if not raw:
            micro.pop(key, None)
            continue
        if key == "potential_order":
            micro[key] = _count(raw, key, 0)
        else:
            try:
                number = float(raw)
            except ValueError as exc:
                raise ValueError(f"{key}: enter a finite number") from exc
            if not isfinite(number):
                raise ValueError(f"{key}: enter a finite number")
            micro[key] = number
    if "rho_kg_m3" in micro and micro["rho_kg_m3"] is not None:
        if isinstance(draft.get("modal"), dict):
            draft["modal"]["density_kg_m3"] = micro["rho_kg_m3"]
        if isinstance(draft.get("dynamics"), dict):
            draft["dynamics"]["density_kg_m3"] = micro["rho_kg_m3"]
    if flexo and not current_on:
        saved_bc = micro.get("editor_saved_electrical_bc")
        if saved_bc and isinstance(draft.get("boundary_conditions", {}).get("variational"), dict):
            draft["boundary_conditions"]["variational"]["electrical"] = deepcopy(saved_bc)
    draft["micro"] = micro
    return draft


__all__ = ("selected_grid_case", "selected_micro_case")
