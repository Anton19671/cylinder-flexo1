#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression guard for acceptance after the last permitted MVI correction."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np

import cylinder_te1_mvi_core as core
import cylinder_te1_stage3c_internal_mvi as te1
import cylinder_cuf_stage7d_general_internal_mvi as cuf
import cylinder_kl_stage8b5_strict_mvi as kl


class _ScalarState:
    def __init__(self, value=1.0):
        self.value = float(value)

    def copy(self):
        return _ScalarState(self.value)


@contextmanager
def _patched(items):
    old = [(obj, name, getattr(obj, name)) for obj, name, _new in items]
    try:
        for obj, name, new in items:
            setattr(obj, name, new)
        yield
    finally:
        for obj, name, value in reversed(old):
            setattr(obj, name, value)


def _test_kl():
    labels = (("q", 0),)
    active = np.array([True])
    norms = np.array([1.0])

    def direction_data(state, *_args, **_kwargs):
        r = np.array([state.value])
        return r.copy(), r.copy(), labels, norms.copy(), active.copy(), r.copy()

    def tangent(state, *_args, **_kwargs):
        r = np.array([state.value])
        return np.array([[1.0]]), r, r.copy(), labels, norms.copy(), active.copy()

    def set_parameters(state, _ops, _maps, _direction, _labels, values):
        state.value = float(np.asarray(values)[0])

    with _patched([
        (kl, "_direction_data", direction_data),
        (kl, "directional_tangent_fd", tangent),
        (kl, "_set_raw_parameters", set_parameters),
        (kl, "normalized_projected_inf", lambda state, *_a, **_k: abs(state.value)),
    ]):
        settings = kl.KLInternalMVISettings(1e-12, 1e-12, 1e-7, 1, 1)
        result = kl.solve_strict_halfstep(
            _ScalarState(), None, None, "X", None, settings,
            quadrature=None, cycle=1,
        )
    if result.iterations != 1 or result.projected_inf > settings.projected_tolerance:
        raise AssertionError("KL did not accept convergence after the final correction")


def _test_kl_strict_tolerance():
    settings = kl.KLInternalMVISettings(1e-5, 1e-12, 1e-5, 1, 1)
    if not kl._projected_converged(1e-5, settings):
        raise AssertionError("KL rejected the exact requested tolerance")
    if kl._projected_converged(np.nextafter(1e-5, np.inf), settings):
        raise AssertionError("KL accepted a projected residual above the requested tolerance")


def _test_kl_relative_projected_measure():
    labels = (("q", 0),)
    active = np.array([True])
    norms = np.array([1.0])

    def direction_data(state, *_args, **_kwargs):
        r = np.array([state.value])
        return r.copy(), r.copy(), labels, norms.copy(), active.copy(), r.copy()

    with _patched([(kl, "_direction_data", direction_data)]):
        settings = kl.KLInternalMVISettings(1e-7, 1e-12, 1e-7, 1, 1)
        result = kl.solve_strict_halfstep(
            _ScalarState(1e-2), None, None, "X", None, settings,
            quadrature=None, cycle=1, projected_scale=1e6,
        )
    if result.iterations != 0:
        raise AssertionError("KL relative projected measure did not stop before tangent assembly")
    if not result.projected_inf > settings.projected_tolerance:
        raise AssertionError("KL relative test did not distinguish absolute and relative residual")
    if not result.projected_relative <= settings.projected_tolerance:
        raise AssertionError("KL relative projected residual was computed incorrectly")

    class Grid:
        nx = 3
        ns = 4
        dx = 5.0
        ds = 7.0

    p = np.zeros((3, Grid.ns, Grid.nx), float)
    p[2] = 2.0
    got = kl._surface_load_resultant_scale(Grid(), p)
    expected = 2.0 * ((Grid.nx - 1) * Grid.dx) * (Grid.ns * Grid.ds)
    if not np.isclose(got, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"KL surface-load force scale mismatch: {got} != {expected}")


def _test_kl_geometric_trace_roundoff_guard():
    ns, nx = 5, 7
    geometric = SimpleNamespace(kind="GEOMETRIC")
    end = SimpleNamespace(
        u=geometric, v=geometric, w=geometric, slope=geometric,
    )
    bc = SimpleNamespace(left=end, right=end)
    dU = np.zeros((3, ns, nx), float)
    dUx = np.zeros_like(dU)

    # A large admissible reconstructed variation can leave an end trace of
    # machine precision after direct elimination.  This is not a BC breach.
    dUx[2, :, 1:-1] = 1.0e8
    dUx[2, :, 0] = 5.0e-7
    dUx[2, :, -1] = 5.0e-7
    variation = SimpleNamespace(dU=dU, dUx=dUx)
    kl.kw._check_geometric_variation_traces(variation, bc, None)

    # A material end trace must still be rejected.
    dUx[2, :, -1] = 1.0e-3
    try:
        kl.kw._check_geometric_variation_traces(variation, bc, None)
    except kl.ke.KLEssentialConstraintError:
        pass
    else:
        raise AssertionError("KL geometric trace guard accepted a material BC violation")


def _test_cuf():
    mask = np.array([True])
    ids = np.array([0])
    norms = np.array([1.0])
    active = np.array([True])
    Tn = np.array([[1.0]])

    def direction_data(*_args, **_kwargs):
        return mask.copy(), Tn.copy(), ids.copy(), norms.copy(), active.copy(), Tn.copy()

    def tangent(*_args, **_kwargs):
        return Tn.copy(), Tn.copy(), mask.copy(), ids.copy(), norms.copy(), active.copy()

    def set_parameters(state, _direction, _ids, values):
        state.value = float(np.asarray(values)[0])

    with _patched([
        (cuf, "_residual_vector", lambda _fn, state, _grid: np.array([state.value])),
        (cuf, "_direction_data", direction_data),
        (cuf, "directional_tangent_action_fd", tangent),
        (cuf, "projected_inf_for_direction", lambda state, *_a, **_k: abs(state.value)),
        (cuf.g8, "n_full_dofs", lambda _state, _grid: 1),
        (cuf.g8, "residual_projection_weights", lambda _state, _grid: np.ones(1)),
        (cuf.g8, "get_active_parameters", lambda state, _d, _m: (np.array([state.value]), ids.copy())),
        (cuf.g8, "set_active_parameters", set_parameters),
    ]):
        settings = cuf.GeneralInternalMVISettings(1e-12, 1e-12, 1e-7, 1, 1)
        result = cuf.solve_strict_halfstep(
            _ScalarState(), None, None, "X", None, settings,
            quadrature=None, cycle=1,
        )
    if result.iterations != 1 or result.projected_inf > settings.projected_tolerance:
        raise AssertionError("CUF did not accept convergence after the final correction")


def _test_cuf_relative_projected_measure():
    mask = np.array([True])
    ids = np.array([0])
    norms = np.array([1.0])
    active = np.array([True])
    Tn = np.array([[1.0]])

    def direction_data(*_args, **_kwargs):
        return mask.copy(), Tn.copy(), ids.copy(), norms.copy(), active.copy(), Tn.copy()

    def tangent(*_args, **_kwargs):
        return Tn.copy(), Tn.copy(), mask.copy(), ids.copy(), norms.copy(), active.copy()

    def set_parameters(state, _direction, _ids, values):
        state.value = float(np.asarray(values)[0])

    with _patched([
        (cuf, "_residual_vector", lambda _fn, state, _grid: np.array([state.value])),
        (cuf, "_direction_data", direction_data),
        (cuf, "directional_tangent_action_fd", tangent),
        (cuf, "projected_inf_for_direction", lambda state, *_a, **_k: abs(state.value)),
        (cuf.g8, "n_full_dofs", lambda _state, _grid: 1),
        (cuf.g8, "residual_projection_weights", lambda _state, _grid: np.ones(1)),
        (cuf.g8, "get_active_parameters", lambda state, _d, _m: (np.array([state.value]), ids.copy())),
        (cuf.g8, "set_active_parameters", set_parameters),
    ]):
        settings = cuf.GeneralInternalMVISettings(1e-7, 1e-12, 1e-7, 1, 1)
        result = cuf.solve_strict_halfstep(
            _ScalarState(1e-2), None, None, "X", None, settings,
            quadrature=None, cycle=1, projected_scale=1e6,
        )
    if result.iterations != 1:
        raise AssertionError("CUF half-step was stopped by the distinct V-4 cycle scale")
    if not result.projected_relative <= settings.projected_tolerance:
        raise AssertionError("CUF original reduced equation did not converge")

    def reference_residual(state):
        return np.array([state.value])

    reference_residual.mvi_projected_reference_residual = np.array([2.0])
    with _patched([
        (cuf, "_residual_vector", lambda fn, state, _grid: np.asarray(fn(state), float)),
        (cuf, "_direction_data", direction_data),
        (cuf.g8, "n_full_dofs", lambda _state, _grid: 1),
    ]):
        scale = cuf._fixed_projected_scale(
            _ScalarState(1e-9), None, None, reference_residual, np.ones(1),
        )
    if scale != 2.0:
        raise AssertionError(f"CUF applied-load projected scale was not retained: {scale}")


def _test_te1_strict():
    mask = np.array([True])
    ids = np.array([0])
    norms = np.array([1.0])
    active = np.array([True])
    Tn = np.array([[1.0]])

    class Grid:
        n_full_dofs = 1

        @staticmethod
        def residual_projection_weights():
            return np.ones(1)

    def direction_data(*_args, **_kwargs):
        return mask.copy(), Tn.copy(), ids.copy(), norms.copy(), active.copy(), Tn.copy()

    def tangent(*_args, **_kwargs):
        return Tn.copy(), Tn.copy(), mask.copy(), ids.copy(), norms.copy(), active.copy()

    def set_parameters(state, _direction, _ids, values):
        state.value = float(np.asarray(values)[0])

    with _patched([
        (te1, "_residual_vector", lambda _fn, state, _grid: np.array([state.value])),
        (te1, "_direction_data", direction_data),
        (te1, "directional_tangent_action_fd", tangent),
        (te1, "projected_inf_for_direction", lambda state, *_a, **_k: abs(state.value)),
        (te1.mvi, "get_active_parameters", lambda state, _d, _m: (np.array([state.value]), ids.copy())),
        (te1.mvi, "set_active_parameters", set_parameters),
    ]):
        settings = te1.InternalMVISettings(1e-12, 1e-12, 1e-7, 1, 1)
        result = te1.solve_strict_halfstep(
            _ScalarState(), Grid(), None, "X", None, settings,
            quadrature=None,
        )
    if result.iterations != 1 or result.projected_inf > settings.projected_tolerance:
        raise AssertionError("TE1 strict MVI did not accept the final correction")


def _test_legacy_core():
    ids = np.array([0])

    def set_parameters(state, _direction, _ids, values):
        state.value = float(np.asarray(values)[0])

    with _patched([
        (core, "reconstruct_vector", lambda state, _grid: np.array([state.value])),
        (core, "build_direction_tangent", lambda *_a, **_k: (np.array([[1.0]]), ids.copy())),
        (core, "get_active_parameters", lambda state, _d, _m: (np.array([state.value]), ids.copy())),
        (core, "set_active_parameters", set_parameters),
    ]):
        result = core.solve_halfstep(
            _ScalarState(), None, "X",
            lambda U: (np.asarray(U), np.array([[1.0]])),
            row_weights=np.ones(1), projected_tol=1e-12,
            correction_tol=1e-12, max_iterations=1,
        )
    if not result.converged or result.iterations != 1 or result.projected_norm > 1e-12:
        raise AssertionError("legacy TE1 MVI did not accept the final correction")


def run_checks(verbose=True):
    _test_kl()
    _test_kl_strict_tolerance()
    _test_kl_relative_projected_measure()
    _test_kl_geometric_trace_roundoff_guard()
    _test_cuf()
    _test_cuf_relative_projected_measure()
    _test_te1_strict()
    _test_legacy_core()
    if verbose:
        print("MVI_LIMIT_SELF_CHECK: PASS")


if __name__ == "__main__":
    run_checks()
