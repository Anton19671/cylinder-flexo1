"""Carry accepted fixed-rank X/Y factors between load or Newmark steps.

The physical field is exactly the separated product, with no nodal lift.
Continuation must also preserve small physical mechanical/electrical fields;
comparisons are therefore made in dimensionless component scales.
"""
from __future__ import annotations

import numpy as np


def continue_accepted_mvi_profiles(previous, mechanical, electrical, *,
                                   new_scales, full_space, rank):
    """Reparameterize one accepted physical state into new component scales.

    The new prescribed traces may differ at the new physical time. Their
    defect is reported, not silently corrected or projected into a full
    nodal field. Subsequent X/Y half-steps must satisfy them at fixed rank.
    """
    if previous is None or not previous.converged:
        raise ValueError("MVI_CONTINUATION_REQUIRES_CONVERGED_PREVIOUS_STEP")
    cert = previous.mvi_certificate
    if not isinstance(cert, dict) or not cert.get("accepted", False):
        raise ValueError("MVI_CONTINUATION_REQUIRES_ACCEPTED_CERTIFICATE")
    if type(rank) is not int or not 1 <= rank <= 8:
        raise ValueError("MVI_CONTINUATION_INCOMPATIBLE_FIXED_RANK_OR_SCALE")
    if cert.get("fixed_rank", rank) != rank:
        raise ValueError("MVI_CONTINUATION_CERTIFICATE_RANK_MISMATCH")

    q = np.asarray(mechanical, float)
    p = None if electrical is None else np.asarray(electrical, float)
    if (q.ndim != 3 or q.size == 0 or not np.isfinite(q).all() or
            (p is not None and (p.ndim != 3 or p.shape[1:] != q.shape[1:] or
                                not np.isfinite(p).all()))):
        raise ValueError("MVI_CONTINUATION_INVALID_PHYSICAL_FIELDS")
    actual = q if p is None else np.concatenate((q, p))
    x = np.asarray(previous.x_profiles, float)
    y = np.asarray(previous.y_profiles, float)
    old = np.asarray(previous.scales, float)
    if (x.shape != (actual.shape[0], rank, actual.shape[1]) or
            y.shape != (actual.shape[0], rank, actual.shape[2]) or
            old.shape != actual.shape):
        raise ValueError("MVI_CONTINUATION_INCOMPATIBLE_FIXED_RANK_OR_SCALE")
    new = np.asarray(new_scales, float)
    if new.size != actual.size:
        raise ValueError("MVI_CONTINUATION_INCOMPATIBLE_FIXED_RANK_OR_SCALE")
    new = new.reshape(actual.shape)
    if (not all(np.isfinite(v).all() for v in (x, y, old, new)) or
            np.any(old <= 0) or np.any(new <= 0)):
        raise ValueError("MVI_CONTINUATION_INVALID_SCALES")

    # Compare in the dimensionless coordinates used by MVI. The former
    # fixed 1e-10 physical tolerance would hide smaller real displacements.
    represented = np.einsum("crx,crs->cxs", x, y)
    scaled_actual = actual / old
    tolerance = 1.e-10 * np.maximum(1., np.maximum(np.abs(represented),
                                                    np.abs(scaled_actual)))
    if (not np.isfinite(represented).all() or
            not np.isfinite(scaled_actual).all() or
            np.any(np.abs(represented - scaled_actual) > tolerance)):
        raise ValueError("MVI_CONTINUATION_PREVIOUS_FIELD_MISMATCH")
    if hasattr(previous, "mechanical"):
        stored_q = np.asarray(previous.mechanical, float)
        stored_p = getattr(previous, "electrical", None)
        if (stored_q.shape != q.shape or not np.isfinite(stored_q).all() or
                (p is None) != (stored_p is None)):
            raise ValueError("MVI_CONTINUATION_PREVIOUS_STATE_MISMATCH")
        stored = stored_q if p is None else np.concatenate(
            (stored_q, np.asarray(stored_p, float)))
        if (stored.shape != actual.shape or not np.isfinite(stored).all() or
                np.any(np.abs(stored / old - scaled_actual) > tolerance)):
            raise ValueError("MVI_CONTINUATION_PREVIOUS_STATE_MISMATCH")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        ratio = old / new
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("MVI_CONTINUATION_INVALID_SCALES")
    # A relative-only comparison is essential for very small positive ratios.
    if not np.allclose(ratio, ratio[:, :1, :1], rtol=1.e-12, atol=0.):
        raise ValueError("MVI_CONTINUATION_NONSEPARABLE_SCALE_CHANGE")
    xx = x * ratio[:, 0, 0, None, None]
    yy = y.copy()
    if not np.isfinite(xx).all():
        raise ValueError("MVI_CONTINUATION_INVALID_SCALES")
    defect = np.asarray(full_space.defect(
        np.einsum("crx,crs->cxs", xx, yy).ravel()), dtype=float)
    if not np.isfinite(defect).all():
        raise ValueError("MVI_CONTINUATION_INVALID_NEW_TRACES")
    return xx, yy, {
        "initialization": "CONTINUATION", "rank": rank,
        "old_state_match_error_physical": float(
            np.max(np.abs(old * represented - actual), initial=0)),
        "new_trace_defect_before_halfstep_scaled": float(
            np.max(np.abs(defect), initial=0)),
    }


__all__ = ("continue_accepted_mvi_profiles",)
