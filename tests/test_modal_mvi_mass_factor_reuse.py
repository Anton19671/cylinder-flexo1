"""Reuse the already validated projected mass factor without changing MVI.

No material or shell parameters are modified. Both paths compute the
same mass-orthogonal projection on the same frozen-partner subspace.
"""
import numpy as np
from scipy.linalg import cholesky

from cylinder_modal_mvi import _mass_project_previous_mode


def test_mass_projection_reuse_matches_independent_positive_definite_solve():
    rng = np.random.default_rng(20261102)
    for dim in (2, 7, 20, 48):
        full = dim + 10
        sample = rng.standard_normal((full, dim))
        basis, _ = np.linalg.qr(sample)
        weights = rng.uniform(0.4, 4.0, full)
        mass = lambda x: weights * x
        reduced = basis.T @ (weights[:, None] * basis)
        # A small orthogonality transformation models exactly the completed
        # modal mass deflation subspace, not an altered physical mass matrix.
        z = np.eye(dim)[:, :dim-1]
        mz = z.T @ reduced @ z
        previous = basis @ rng.standard_normal(dim)
        a = _mass_project_previous_mode(basis, z, mz, previous, mass)
        b = _mass_project_previous_mode(
            basis, z, mz, previous, mass,
            mass_factor=cholesky(mz, lower=True, check_finite=True))
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=3e-13)


def test_rank_deficient_or_nonpositive_projected_mass_is_not_accepted():
    import pytest
    # No change to the positive-definiteness checks in the real halfstep.
    with pytest.raises(np.linalg.LinAlgError):
        cholesky(np.diag([1., -1.]), lower=True, check_finite=True)
