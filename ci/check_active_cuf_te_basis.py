#!/usr/bin/env python3
"""No-solver verification of the active CUF TE1..TE4 thickness basis.

The active weak shell uses (2*z/h)**k, not the legacy final-interface
provider's dimensional z**k. Run before reporting physical face fields.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexoelectric.cuf_g3d_weak import mass_matrix_te, te_basis


def check() -> None:
    for h, radius in ((0.05, 1.0), (0.17, 0.43)):
        for order in range(1, 5):
            for zeta in (-1.0, -0.625, 0.0, 0.375, 1.0):
                z = 0.5 * h * zeta
                values, first, second = te_basis(z, h, order)
                expected = np.array([zeta**k for k in range(order + 1)])
                expected_first = np.array([
                    0.0 if k == 0 else 2 * k * zeta**(k - 1) / h
                    for k in range(order + 1)
                ])
                expected_second = np.array([
                    0.0 if k < 2 else 4 * k * (k - 1) * zeta**(k - 2) / h**2
                    for k in range(order + 1)
                ])
                np.testing.assert_allclose(values, expected, rtol=2e-14, atol=2e-14)
                np.testing.assert_allclose(first, expected_first, rtol=2e-14, atol=2e-12)
                np.testing.assert_allclose(second, expected_second, rtol=2e-14, atol=2e-10)

            # Independent Gauss integration of curved-shell mass moments.
            rho = 7800.0
            actual = mass_matrix_te(
                radius_m=radius, thickness_m=h, rho_kg_m3=rho,
                te_order=order,
            )
            xi, weights = np.polynomial.legendre.leggauss(12)
            basis = np.stack([xi**k for k in range(order + 1)], axis=1)
            weighted = weights * (0.5 * h * rho) * (1 + 0.5 * h * xi / radius)
            reference = basis.T @ (weighted[:, None] * basis)
            np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-12)
            np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-12)
            if np.linalg.eigvalsh(actual)[0] <= 0:
                raise AssertionError("Normalized TE mass matrix not positive definite")

            coefficients = np.array([
                [math.sin(k + 1) for k in range(order + 1)],
                [math.cos(k + 2) for k in range(order + 1)],
                [(-1)**k / (k + 1) for k in range(order + 1)],
            ])
            for zeta in (-1.0, 0.0, 1.0):
                observed = coefficients @ te_basis(0.5 * h * zeta, h, order)[0]
                expected = sum(coefficients[:, k] * zeta**k
                               for k in range(order + 1))
                np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)
            if not np.allclose(te_basis(h / 2, h, order)[0], 1.0):
                raise AssertionError("Outer face is not normalized to +1")
            if not np.allclose(te_basis(-h / 2, h, order)[0],
                               [(-1)**k for k in range(order + 1)]):
                raise AssertionError("Inner face is not normalized to -1")
    print("PASS: active CUF TE1..TE4 basis, derivatives, mass and physical faces")


if __name__ == "__main__":
    check()
