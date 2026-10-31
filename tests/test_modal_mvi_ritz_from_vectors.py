"""Thin-shell generalized eigenvalue accuracy: use the computed eigenvector.

No physical shell solve is run in this regression module.
"""
import numpy as np
import pytest

import cylinder_modal_mvi as modal


def test_ritz_value_is_not_low_accuracy_solver_eigenvalue():
    k = np.diag([1.e15, 14056558.9409])
    m = np.eye(2)
    vector = np.array([0., 1.])
    wrong_eigenvalue_reported_by_solver = 14056574.2392
    correct_value = modal._modal_ritz_value(k, m, vector)
    np.testing.assert_allclose(correct_value, 14056558.9409, rtol=0., atol=1.e-8)
    assert abs(correct_value - wrong_eigenvalue_reported_by_solver) > 15.


def test_ritz_generalized_quotient_uses_original_mass():
    k = np.diag([2., 8.])
    m = np.diag([2., 4.])
    vector = np.array([1., 2.])
    expected = (2. + 4. * 8.) / (2. + 4. * 4.)
    np.testing.assert_allclose(modal._modal_ritz_value(k, m, vector), expected)


def test_ritz_rejects_nonpositive_mass_and_bad_energies():
    with pytest.raises(ValueError, match="MODAL_MVI_INVALID_RITZ_QUOTIENT"):
        modal._modal_ritz_value(np.eye(2), np.zeros((2, 2)), np.array([1., 0.]))
    with pytest.raises(ValueError, match="MODAL_MVI_INVALID_RITZ_QUOTIENT"):
        modal._modal_ritz_value(np.eye(2) * np.nan, np.eye(2), np.array([1., 0.]))


def test_halfstep_uses_projected_ritz_of_selected_vector(monkeypatch):
    # Exact rank-one X space, two independent axial coefficients, no BC.
    op = object.__new__(modal._SeparatedModalProblem)
    x = np.array([[[1., 0.]]])
    y = np.array([[[1., 0.]]])
    op.nm, op.ne = 4, 0
    op.nx, op.ns = 2, 2
    op.n_components = 1
    op.shape = (1, 2, 2)
    op.scales = np.ones(4)
    op.cm = np.empty((0, 4))
    op.mass = lambda v: v
    op.stiffness = lambda v: np.array([14056558.9409, 1.e15,
                                      14056558.9409, 1.e15]) * v
    op.projected_bases = lambda _x, _y, _axis: (
        np.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]]),
        np.empty((0, 0)))

    def corrupted_lapack_eigenvalue(k, m, **_):
        return (np.array([14056574.2392, 14056574.24]), np.eye(2))
    monkeypatch.setattr(modal, "eigh", corrupted_lapack_eigenvalue)
    q, p, lam, old = op.halfstep(x, y, "X", [])
    assert p is None and old is None
    np.testing.assert_allclose(lam, 14056558.9409, rtol=0., atol=1.e-8)
    np.testing.assert_allclose(q, [1., 0., 0., 0.])
