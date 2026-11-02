"""Source-level fixed-rank variational-space checks for modal MVI.

These checks do not solve a shell problem and do not claim to verify the
global spectrum. A full numerical benchmark remains necessary.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_modal_mvi import _SeparatedModalProblem, _initial_factors
from cylinder_modal_reference import _component_separation_ranks


@pytest.mark.parametrize("rank", range(1, 9))
def test_fixed_rank_yx_variation_spaces_have_distinct_dimensions(rank):
    nx, ns = 9, 11
    operator = object.__new__(_SeparatedModalProblem)
    operator.rank = rank
    operator.nx, operator.ns = nx, ns
    operator.n_components, operator.nm, operator.ne = 1, nx * ns, 0
    operator.shape = (1, nx, ns)
    operator.scales = np.ones(operator.nm)
    operator.cm = np.empty((0, operator.nm))
    operator.ce = np.empty((0, 0))
    operator.large_harmonic = False
    x, y = _initial_factors(1, rank, nx, ns, 0)
    bx, ex = operator.projected_bases(x, y, "X")
    by, ey = operator.projected_bases(x, y, "Y")
    assert bx.shape == (nx * ns, rank * nx)
    assert by.shape == (nx * ns, rank * ns)
    assert ex.shape == ey.shape == (0, 0)
    assert np.linalg.matrix_rank(bx) == rank * nx
    assert np.linalg.matrix_rank(by) == rank * ns
    field = np.einsum("crx,crs->cxs", x, y)
    assert _component_separation_ranks(field[None]) == [[rank]]


@pytest.mark.parametrize("rank", range(1, 9))
def test_nonseparable_exact_reference_requires_higher_rank(rank):
    nx, ns = 9, 11
    rng = np.random.default_rng(182731)
    u = np.linalg.qr(rng.standard_normal((nx, 8)))[0]
    v = np.linalg.qr(rng.standard_normal((ns, 8)))[0]
    singular = np.array([.70, .45, .35, .29, .25, .20, .17, .13])
    singular /= np.linalg.norm(singular)
    exact = (u * singular) @ v.T
    truncated = (u[:, :rank] * singular[:rank]) @ v[:, :rank].T
    error_sq = np.linalg.norm(exact - truncated) ** 2
    np.testing.assert_allclose(error_sq, np.sum(singular[rank:] ** 2),
                               rtol=1.e-10, atol=1.e-12)
    # For K=4I-3qq^T and M=I, this is the exact global minimum of
    # the Rayleigh quotient over all rank<=rank normalized trial fields.
    lam = 4. - 3. * np.sum(singular[:rank] ** 2)
    assert 1. - 1.e-12 <= lam <= 4. + 1.e-12
    if rank == 8:
        np.testing.assert_allclose(lam, 1., atol=1.e-12)
    else:
        assert lam > 1.
