"""Rank-1 through rank-8 modal regression on a known separated spectrum.

This exercises the real fixed-rank Y/X modal eigensolver, mass and stiffness
projections, factor recovery, and output rank. It replaces only the physical
shell matrices with a small exact symmetric reference operator. No shell
benchmark is claimed here.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_modal_mvi import _SeparatedModalProblem, solve_modal_mvi
from cylinder_modal_reference import _component_separation_ranks


@pytest.mark.parametrize("rank", range(1, 9))
def test_modal_rank_1_through_8_changes_eigenvalue_when_needed(rank):
    nx, ns = 9, 11
    rng = np.random.default_rng(182731)
    x_basis = np.linalg.qr(rng.standard_normal((nx, 8)))[0]
    s_basis = np.linalg.qr(rng.standard_normal((ns, 8)))[0]
    weights = np.array([.70, .45, .35, .29, .25, .20, .17, .13])
    weights /= np.linalg.norm(weights)
    true_mode = ((x_basis * weights) @ s_basis.T).ravel()
    stiffness = 4. * np.eye(nx * ns) - 3. * np.outer(true_mode, true_mode)
    exact_rank_r_eigenvalue = 4. - 3. * np.sum(weights[:rank] ** 2)

    # Keep the actual projected-basis and alternating Y/X algorithms, while
    # replacing only the constitutive stiffness and mass actions by a case
    # with an independently known fixed-rank variational minimum.
    operator = object.__new__(_SeparatedModalProblem)
    operator.rank = rank
    operator.nx, operator.ns = nx, ns
    operator.n_components, operator.nm, operator.ne = 1, nx * ns, 0
    operator.shape = (1, nx, ns)
    operator.scales = np.ones(operator.nm)
    operator.cm = np.empty((0, operator.nm))
    operator.ce = np.empty((0, 0))
    operator.n_mechanical_free_dofs = operator.nm
    operator.n_electric_free_dofs = 0
    operator.large_harmonic = False
    operator.f = SimpleNamespace(theory="SYNTHETIC")
    operator._previous_mass_cache = {}
    operator.mass = lambda q: np.asarray(q, float).copy()
    operator.stiffness = lambda q: stiffness @ np.asarray(q, float).ravel()
    prepared = SimpleNamespace(method="MVI", mvi_settings=SimpleNamespace(
        rank=rank, max_sweeps=80))

    result = solve_modal_mvi(prepared, 1., 1, _cached_operator=operator,
                             eigen_tolerance=1.e-9, mode_tolerance=2.e-6)
    np.testing.assert_allclose(result["omega_squared_rad2_s2"][0],
                               exact_rank_r_eigenvalue, rtol=1.e-8, atol=1.e-9)
    assert result["rank"] == rank
    assert _component_separation_ranks(result["mechanical_modes"]) == [[rank]]
    assert result["mvi_modal_algorithm"] == (
        "fixed_rank_constrained_variational_YX_stationarity")
