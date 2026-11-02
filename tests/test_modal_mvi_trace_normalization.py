"""Algebraic regressions; no shell eigensolve or benchmark run."""
import unittest
import numpy as np

from cylinder_modal_separated_space import (
    fixed_trace_xy_basis, normalize_modal_partner)
from cylinder_variational_mvi import reconstruct_profiles


class ModalTraceNormalizationTests(unittest.TestCase):
    def test_nearly_dependent_factors_keep_exact_zero_traces_and_rank(self):
        rng = np.random.default_rng(92)
        for axis in ("X", "Y"):
            for rank in (1, 2, 3, 4, 8):
                with self.subTest(axis=axis, rank=rank):
                    nx, ns = 13, 21
                    x = rng.normal(size=(1, rank, nx))
                    y = rng.normal(size=(1, rank, ns))
                    mask = np.zeros((1, nx, ns), bool)
                    partner = x if axis == "Y" else y
                    partner[:, :, [0, -1]] = 0.
                    if rank > 1:
                        partner[0, 1] = partner[0, 0]
                        partner[0, 1, 1:-1] += (
                            1.e-10 * rng.normal(size=partner.shape[2] - 2))
                    if axis == "Y":
                        mask[:, [0, -1], :] = True
                    else:
                        mask[:, :, [0, -1]] = True
                    before = reconstruct_profiles(x, y).ravel()
                    xn, yn = normalize_modal_partner(x, y, axis, mask.ravel())
                    after = reconstruct_profiles(xn, yn).ravel()
                    np.testing.assert_allclose(after, before, rtol=1.e-12, atol=1.e-12)
                    normalized = xn if axis == "Y" else yn
                    np.testing.assert_array_equal(normalized[:, :, [0, -1]], 0.)
                    np.testing.assert_allclose(
                        normalized[0] @ normalized[0].T, np.eye(rank), atol=1.e-14)
                    basis = fixed_trace_xy_basis(
                        xn, yn, axis, mask.ravel(), np.ones(mask.size))
                    self.assertEqual(basis.shape[1], rank * (ns if axis == "Y" else nx))
                    np.testing.assert_allclose(
                        basis @ (basis.T @ after), after, rtol=1.e-12, atol=1.e-12)

    def test_nonzero_factor_traces_are_not_clipped(self):
        # Admissibility can arise from cancellation of nonzero rank terms.
        x = np.array([[[1., 2., 3., 1.], [1., -1., 2., 1.]]])
        y = np.array([[[1., 2., 3.], [-1., -2., -3.]]])
        fixed = np.zeros((1, 4, 3), bool)
        fixed[:, [0, -1], :] = True
        before = reconstruct_profiles(x, y)
        xn, yn = normalize_modal_partner(x, y, "Y", fixed.ravel())
        np.testing.assert_allclose(reconstruct_profiles(xn, yn), before, atol=1.e-14)
        self.assertGreater(np.linalg.norm(xn[:, :, [0, -1]]), .1)
        b = fixed_trace_xy_basis(xn, yn, "Y", fixed.ravel(), np.ones(fixed.size))
        np.testing.assert_allclose(b @ (b.T @ before.ravel()), before.ravel(), atol=1.e-14)

    def test_zero_component_completion_stays_in_free_rows(self):
        x, y = np.zeros((1, 2, 5)), np.ones((1, 2, 7))
        mask = np.zeros((1, 5, 7), bool)
        mask[:, [0, -1], :] = True
        xn, yn = normalize_modal_partner(x, y, "Y", mask.ravel())
        np.testing.assert_array_equal(reconstruct_profiles(xn, yn), 0.)
        np.testing.assert_array_equal(xn[:, :, [0, -1]], 0.)
        np.testing.assert_allclose(xn[0] @ xn[0].T, np.eye(2), atol=1.e-14)


if __name__ == "__main__":
    unittest.main()
