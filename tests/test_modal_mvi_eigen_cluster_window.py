"""Doublet-aware spectral window avoids redundant generalized solves.

The physical K/M matrices, all modal candidate validations, and handling of
higher multiplicity remain identical.
"""
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import eigh as actual_eigh

from cylinder_modal_mvi import _SeparatedModalProblem


@pytest.mark.parametrize(
    "spectrum,expected_calls,expected_first_window",
    [
        ([4.0, 4.0, 12.0, 20.0], 1, (0, 2)),
        ([4.0, 4.0, 4.0, 20.0], 2, (0, 2)),
        ([4.0, 5.0, 12.0, 20.0], 1, (0, 2)),
        ([4.0, 4.0], 1, (0, 1)),
    ],
)
def test_modal_eigen_cluster_window_without_changing_ritz_value(
        spectrum, expected_calls, expected_first_window):
    n = len(spectrum)
    op = object.__new__(_SeparatedModalProblem)
    op.nm, op.ne = n, 0
    op.shape = (1, 1, n)
    op.scales = np.ones(n)
    op.cm = np.empty((0, n))
    bm = np.eye(n)
    stiffness = np.diag(spectrum)
    mass = np.eye(n)
    op.projected_bases = lambda *_: (bm, np.empty((0, 0)))
    op.cached_cuf_projection = lambda _b: (stiffness.copy(), mass.copy())
    op.mass = lambda vec: vec
    windows = []

    def track_eigensolve(k, m, **kwargs):
        windows.append(kwargs["subset_by_index"])
        return actual_eigh(k, m, **kwargs)

    with patch("cylinder_modal_mvi.eigh", side_effect=track_eigensolve):
        q, p, value, old = op.halfstep(None, None, "X", [])

    assert len(windows) == expected_calls
    assert windows[0] == expected_first_window
    assert p is None and old is None
    np.testing.assert_allclose(value, spectrum[0], rtol=0., atol=1.e-10)
    np.testing.assert_allclose(float(q @ stiffness @ q / (q @ mass @ q)),
                               spectrum[0], rtol=0., atol=1.e-10)
    assert np.isfinite(q).all() and float(np.linalg.norm(q)) > 0.
