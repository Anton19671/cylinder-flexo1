#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict persistence contract for incremental J2 integration-point history.

J2 history belongs to the material model, never to a displacement state or to
Birger GB.  This module stores/loads only accepted history and verifies the
through-thickness quadrature before reuse.  It is deliberately solver-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from final_j2_incremental import J2MaterialError, J2State
from final_j2_finite_strain import FiniteJ2State


FORMAT = "CYLINDER_J2_HISTORY_V1"
SMALL = "SMALL_STRAIN"
FINITE = "FINITE_STRAIN"


def _quad_tolerance(a: np.ndarray, b: np.ndarray) -> float:
    return 256.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(a))) if a.size else 1.0,
        float(np.max(np.abs(b))) if b.size else 1.0,
    )


def _check_quadrature(z_saved, w_saved, h_saved, z, w, h) -> None:
    z_saved = np.asarray(z_saved, float).reshape(-1)
    w_saved = np.asarray(w_saved, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)
    if z_saved.shape != z.shape or w_saved.shape != w.shape:
        raise J2MaterialError("J2 history quadrature shape does not match current quadrature")
    htol = 128.0 * np.finfo(float).eps * max(1.0, abs(float(h_saved)), abs(float(h)))
    if abs(float(h_saved) - float(h)) > htol:
        raise J2MaterialError("J2 history thickness_h does not match the CASE")
    if not np.allclose(z_saved, z, rtol=0.0, atol=_quad_tolerance(z_saved, z)):
        raise J2MaterialError("J2 history quadrature_z does not match the CASE")
    if not np.allclose(w_saved, w, rtol=0.0, atol=_quad_tolerance(w_saved, w)):
        raise J2MaterialError("J2 history quadrature_w does not match the CASE")


def _metadata(kind: str, quadrature, h: float) -> dict:
    if kind not in (SMALL, FINITE):
        raise J2MaterialError("J2 history kind must be SMALL_STRAIN or FINITE_STRAIN")
    z = np.asarray(quadrature.z, float)
    w = np.asarray(quadrature.w, float)
    if z.ndim != 1 or w.ndim != 1 or z.shape != w.shape:
        raise J2MaterialError("J2 quadrature must expose matching 1-D z and w")
    return {
        "j2_history_format": np.asarray(FORMAT),
        "j2_history_kind": np.asarray(kind),
        "quadrature_z": z,
        "quadrature_w": w,
        "thickness_h": np.asarray(float(h)),
    }


def save_small_history(path: Path, state: J2State, *, quadrature, h: float) -> None:
    """Save one accepted small-strain J2 state with immutable quadrature identity."""

    state.validate()
    data = _metadata(SMALL, quadrature, h)
    data.update(
        j2_plastic_strain=np.asarray(state.plastic_strain, float),
        j2_kappa=np.asarray(state.kappa, float),
        j2_equivalent_stress=np.asarray(state.equivalent_stress, float),
    )
    np.savez_compressed(Path(path), **data)


def save_finite_history(path: Path, state: FiniteJ2State, *, quadrature, h: float) -> None:
    """Save one accepted finite-strain J2 state with immutable quadrature identity."""

    state.validate()
    data = _metadata(FINITE, quadrature, h)
    data.update(
        j2_Fp=np.asarray(state.Fp, float),
        j2_kappa=np.asarray(state.kappa, float),
        j2_equivalent_kirchhoff_stress=np.asarray(state.equivalent_kirchhoff_stress, float),
    )
    np.savez_compressed(Path(path), **data)


def _read_common(path: Path, kind: str, quadrature, h: float):
    try:
        z = np.load(Path(path), allow_pickle=False)
    except Exception as exc:
        raise J2MaterialError(f"cannot read J2 history NPZ: {exc}") from exc
    required = ("j2_history_format", "j2_history_kind", "quadrature_z", "quadrature_w", "thickness_h")
    missing = [key for key in required if key not in z.files]
    if missing:
        z.close()
        raise J2MaterialError("J2 history NPZ lacks metadata: " + ", ".join(missing))
    try:
        fmt = str(np.asarray(z["j2_history_format"]).reshape(()))
        got = str(np.asarray(z["j2_history_kind"]).reshape(()))
        if fmt != FORMAT:
            raise J2MaterialError(f"unsupported J2 history format: {fmt!r}")
        if got != kind:
            raise J2MaterialError(f"J2 history kind {got!r} cannot be used as {kind!r}")
        _check_quadrature(
            z["quadrature_z"], z["quadrature_w"], z["thickness_h"],
            quadrature.z, quadrature.w, h,
        )
        return z
    except Exception:
        z.close()
        raise


def load_small_history(path: Path, *, shape: Tuple[int, ...], quadrature, h: float) -> J2State:
    z = _read_common(path, SMALL, quadrature, h)
    try:
        req = ("j2_plastic_strain", "j2_kappa", "j2_equivalent_stress")
        missing = [key for key in req if key not in z.files]
        if missing:
            raise J2MaterialError("small-strain J2 NPZ lacks: " + ", ".join(missing))
        state = J2State(
            np.asarray(z["j2_plastic_strain"], float),
            np.asarray(z["j2_kappa"], float),
            np.asarray(z["j2_equivalent_stress"], float),
        )
        if state.validate() != tuple(shape):
            raise J2MaterialError(f"small-strain J2 history shape must be {tuple(shape)}")
        return state
    finally:
        z.close()


def load_finite_history(path: Path, *, shape: Tuple[int, ...], quadrature, h: float) -> FiniteJ2State:
    z = _read_common(path, FINITE, quadrature, h)
    try:
        req = ("j2_Fp", "j2_kappa", "j2_equivalent_kirchhoff_stress")
        missing = [key for key in req if key not in z.files]
        if missing:
            raise J2MaterialError("finite J2 NPZ lacks: " + ", ".join(missing))
        state = FiniteJ2State(
            np.asarray(z["j2_Fp"], float),
            np.asarray(z["j2_kappa"], float),
            np.asarray(z["j2_equivalent_kirchhoff_stress"], float),
        )
        if state.validate() != tuple(shape):
            raise J2MaterialError(f"finite J2 history shape must be {tuple(shape)}")
        return state
    finally:
        z.close()


__all__ = [
    "FORMAT", "SMALL", "FINITE",
    "save_small_history", "save_finite_history",
    "load_small_history", "load_finite_history",
]
