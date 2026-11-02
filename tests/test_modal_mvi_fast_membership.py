"""Large modal MVI membership checks must avoid redundant global dense QR.

The original membership certificates remain binding.  This test exercises the
real CUF TE1 13x21 frozen-partner spaces; it does not substitute FDM modes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import qr as real_qr

from cylinder_modal_mvi import _SeparatedModalProblem
from cylinder_modal_reference import _modal_spec
from cylinder_modal_separated_space import normalize_modal_partner


ROOT = Path(__file__).resolve().parents[1]
CASE = (ROOT / "config/verification/cammalleri2023/user_13x21"
        / "CAMMALLERI2023_CUF_TE1_13x21_MVI_5.json")


def _modal_trial():
    prepared, density, _ = _modal_spec(json.loads(CASE.read_text("utf-8")))
    operator = _SeparatedModalProblem(prepared, density, 5)
    assert operator.large_harmonic and operator.ne == 0
    x, y, q, p = operator.admissible_start(0, [])
    assert p is None
    return operator, x, y, q


@pytest.mark.parametrize("axis", ("X", "Y"))
def test_large_modal_mvi_basis_is_orthonormal_after_component_scaling(axis):
    operator, x, y, q = _modal_trial()
    x, y = normalize_modal_partner(x, y, axis, operator.fixed_mask)
    bm, _ = operator.projected_bases(x, y, axis)
    # The real fixed-trace basis is orthonormal in dimensionless coordinates.
    # Physical scales are uniform per mechanical component.
    normalized = bm / operator.scales[:operator.nm, None]
    gram = normalized.T @ normalized
    np.testing.assert_allclose(gram, np.eye(len(gram)), rtol=1e-10, atol=1e-10)
    coeff = (bm.T @ q) / np.einsum("ij,ij->j", bm, bm)
    fast = bm @ coeff
    orth, triangular, _ = real_qr(
        bm, mode="economic", pivoting=True, check_finite=False)
    diag = np.abs(np.diag(triangular))
    rank = int(np.count_nonzero(diag > 1e-12 * np.max(diag, initial=0.)))
    original = orth[:, :rank] @ (orth[:, :rank].T @ q)
    np.testing.assert_allclose(fast, original, atol=1e-8, rtol=1e-9)


@pytest.mark.parametrize("axis", ("X", "Y"))
def test_large_modal_mvi_halfstep_uses_no_global_qr(axis):
    operator, x, y, q = _modal_trial()
    x, y = normalize_modal_partner(x, y, axis, operator.fixed_mask)

    def reject_global_qr(a, *args, **kwargs):
        if a.ndim == 2 and a.shape[0] >= operator.nm and a.shape[1] > 50:
            raise AssertionError("Redundant full-nodal QR in MVI halfstep")
        return real_qr(a, *args, **kwargs)

    with patch("cylinder_modal_mvi.qr", side_effect=reject_global_qr):
        out, p, value, old_value = operator.halfstep(
            x, y, axis, [], current_q=q)
    assert p is None and np.isfinite(out).all()
    assert np.isfinite(value) and np.isfinite(old_value)
    assert value >= 0


def test_large_modal_mvi_rejects_field_not_in_frozen_partner_span():
    operator, x, y, q = _modal_trial()
    x, y = normalize_modal_partner(x, y, "Y", operator.fixed_mask)
    bm, _ = operator.projected_bases(x, y, "Y")
    rng = np.random.default_rng(1729)
    noise = rng.standard_normal(len(q))
    noise[operator.fixed_mask] = 0
    coeff = (bm.T @ noise) / np.einsum("ij,ij->j", bm, bm)
    noise -= bm @ coeff
    noise[operator.fixed_mask] = 0
    noise /= np.linalg.norm(noise)
    wrong = q + 0.01 * np.linalg.norm(q) * noise
    with pytest.raises(ValueError, match="frozen-partner span|BASIS_LOST"):
        operator.halfstep(x, y, "Y", [], current_q=wrong)
