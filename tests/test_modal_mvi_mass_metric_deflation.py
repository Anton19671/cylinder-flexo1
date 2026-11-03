"""Numerical mass-metric QR independence of projected modal constraints.

A pair of distinct full-field modes may have near dependent projections
in a rank-one X/Y subspace. Verify we never turn roundoff into a physical
constraint and never weaken the final orthogonality gate.
"""
import numpy as np
from scipy.linalg import cholesky
from cylinder_modal_mvi import _mass_orthogonal_null


def test_rotational_doublet_projection_preserves_axial_overtone():
    mass = np.diag([2., 3., 5., 7.])
    chol = cholesky(mass, lower=True)
    rows = np.array([[0.8, 0., 0., 0.],
                     [0.6, 1.36e-8, 0., 0.]])
    constraints = rows @ chol.T
    null = _mass_orthogonal_null(constraints, mass_factor=chol)
    assert null.shape == (4, 3)
    np.testing.assert_allclose(null.T @ mass @ null, np.eye(3),
                               rtol=0., atol=1.e-12)
    assert float(np.max(np.linalg.norm(constraints @ null, axis=1))) <= 2.e-7
    overtone = np.array([0., 1. / np.sqrt(3.), 0., 0.])
    projected = null @ (null.T @ mass @ overtone)
    np.testing.assert_allclose(projected, overtone, rtol=0., atol=1.e-7)


def test_physically_independent_modal_rows_are_never_dropped():
    mass = np.diag([2., 3., 5., 7.])
    chol = cholesky(mass, lower=True)
    rows = np.array([[0.8, 0., 0., 0.],
                     [0.6, 1.e-4, 0., 0.]])
    constraints = rows @ chol.T
    null = _mass_orthogonal_null(constraints, mass_factor=chol)
    assert null.shape == (4, 2)
    np.testing.assert_allclose(null.T @ mass @ null, np.eye(2),
                               rtol=0., atol=1.e-12)
    assert float(np.max(np.abs(constraints @ null))) < 1.e-12


def test_initial_seed_nullspace_without_mass_is_backward_compatible():
    c = np.array([[1., 0., 0.], [2., 0., 0.]])
    null = _mass_orthogonal_null(c)
    assert null.shape == (3, 2)
    assert float(np.max(np.abs(c @ null))) < 1.e-12
    np.testing.assert_array_equal(
        _mass_orthogonal_null(np.empty((0, 3))), np.eye(3))
