"""Numerical X/Y MVI-1..8 regression: independent rank-r analytic solutions.

Unlike a zero-load smoke, this detects a missing or inactive second and higher
separation term. It tests the MVI algorithm, not physical shell discretization.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
from cylinder_functional_assembly import surface_trapezoid
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_variational_mvi import solve_static_mvi, StaticMVISettings
from cylinder_variational_fdm import StaticFDMSettings


class _QuadraticReferenceFunctional:
    theory = 'KL'
    n_mechanical = 3
    n_derivatives = 10
    electrical_on = False
    R = 2.
    h = .2

    def __init__(self, nodes):
        self.ops = surface_trapezoid(nodes, nodes, 1.2, self.R)
        self.W = self.ops.area_weights
        self.material = SimpleNamespace(young_pa=100.)

    def evaluate(self, mechanical, electrical=None, *, dead_load=None,
                 free_charge=None, **kwargs):
        if electrical is not None or free_charge is not None:
            raise ValueError('electric model is off')
        load = np.zeros_like(mechanical) if dead_load is None else dead_load
        return SimpleNamespace(
            total_enthalpy_j=float(np.sum(self.W * (
                mechanical * mechanical - load * mechanical))),
            mechanical_residual=self.W * (2. * mechanical - load),
            electrical_residual=None,
        )


@pytest.mark.parametrize('rank', range(1, 9))
def test_mvi_rank_r_activates_all_r_terms_and_recovers_exact_solution(rank):
    nodes = 9
    functional = _QuadraticReferenceFunctional(nodes)
    problem = BoundarySpatialProblem(functional, bc_spec={}, calculation_kind='STATIC')
    positions = np.arange(nodes)
    axial_bank = np.array([
        np.cos(np.pi * k * (positions + .5) / nodes)
        for k in range(nodes)
    ])
    axial = np.linalg.qr(axial_bank.T, mode='reduced')[0].T
    circumferential = np.linalg.qr(
        np.random.default_rng(20261030).normal(size=(nodes, nodes))
    )[0].T
    weights = .005 * .72 ** np.arange(rank)
    exact = np.zeros((3, nodes, nodes))
    exact[2] = sum(
        weights[k] * np.outer(axial[k], circumferential[k])
        for k in range(rank)
    )
    result = solve_static_mvi(
        problem, np.zeros_like(exact), dead_load=2. * exact,
        settings=StaticFDMSettings(
            relative_tolerance=1.e-8, absolute_tolerance_scaled=1.e-12,
            max_iterations=7, max_free_dofs=256),
        mvi_settings=StaticMVISettings(rank=rank, max_sweeps=8),
    )
    assert result.converged, result.status
    assert np.max(np.abs(result.mechanical - exact)) < 1.e-10
    assert np.max(np.abs(result.mechanical)) > 1.e-4
    singular_values = np.linalg.svd(result.mechanical[2], compute_uv=False)
    active_terms = np.sum(singular_values > max(1.e-12, singular_values[0] * 1.e-7))
    assert int(active_terms) == rank
    assert singular_values[rank-1] > 1.e-4
