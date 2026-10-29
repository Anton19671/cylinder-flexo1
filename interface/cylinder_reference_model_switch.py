"""Lossless model/kinematics selection for an already loaded variational CASE.

A GUI click creates an in-memory draft only. The original file, boundary
passport, load, grid, material and units are not silently rewritten.
Actual solver support is checked by preflight at Save/Run time.
"""
from __future__ import annotations

import copy

THEORIES = ("KL", "FSDT", "REDDY", "CUF")
CUF_ORDERS = ("TE1", "TE2", "TE3", "TE4")
GEOMETRIES = ("LINEAR", "VON_KARMAN", "GREEN_LAGRANGE")


def selected_model_case(original, *, theory=None, cuf_model=None, geometry_model=None):
    """Produce an isolated CASE draft when a top-level model selector changes."""
    if not isinstance(original, dict) or original.get("solver_backend") != "VARIATIONAL_REFERENCE":
        raise ValueError("Select a saved VARIATIONAL_REFERENCE CASE")

    result = copy.deepcopy(original)
    selected_theory = str(theory if theory is not None else result.get("theory", "")).upper()
    if selected_theory not in THEORIES:
        raise ValueError("Theory must be KL, FSDT, REDDY or CUF")
    result["theory"] = selected_theory

    if selected_theory == "CUF":
        selected_order = str(
            cuf_model if cuf_model is not None else result.get("cuf_model", "")
        ).upper()
        if selected_order not in CUF_ORDERS:
            raise ValueError("CUF model must be TE1, TE2, TE3 or TE4")
        result["cuf_model"] = selected_order
    else:
        result.pop("cuf_model", None)

    geometry = str(
        geometry_model if geometry_model is not None
        else result.get("geometry_model", "")
    ).upper()
    if geometry not in GEOMETRIES:
        raise ValueError("Geometry model must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")
    result["geometry_model"] = geometry

    # MODAL is a linearization of the chosen kinematics. Never turn a
    # free-vibration eigenproblem into a nonlinear time-stepping case.
    kind = str(result.get("calculation_kind", "")).upper()
    if kind == "MODAL" or geometry == "LINEAR":
        result["mode"] = "LINEAR_ELASTIC"
    else:
        result["mode"] = "NONLINEAR_ELASTIC"
    return result


__all__ = ("selected_model_case", "THEORIES", "CUF_ORDERS", "GEOMETRIES")
