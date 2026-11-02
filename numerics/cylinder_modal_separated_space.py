"""Compactly constrained X/Y trial directions for large MODAL MVI grids.

For each scalar component and frozen coordinate, restrict the span of the
actual partner profiles by the homogeneous physical trace equations.
In particular, zeroing fixed nodal entries of an *unconstrained* profile
direction would enlarge the frozen-factor variation space and is forbidden.

This module only builds the existing separated variational test space; it
does not impose Fourier harmonics or solve a full-grid FDM eigenproblem.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def normalize_modal_partner(x, y, axis, fixed_mask=None):
    """QR in free coordinate rows, retaining exact homogeneous nodal traces.

    Full QR of nearly dependent profiles can put O(eps/condition) entries
    into an exactly zero endpoint row. The next constrained Y/X space then
    excludes that artificial direction. Complete the partner family inside
    the free rows instead. Nonuniform traces still use the usual exact
    intersection in fixed_trace_xy_basis; no nodal space is added.
    """
    from cylinder_variational_mvi import _normalize_partner
    if fixed_mask is None:
        return _normalize_partner(x, y, axis)
    if axis not in ("X", "Y"):
        raise ValueError("MODAL_MVI_INVALID_HALFSTEP_AXIS")
    x, y = x.copy(), y.copy()
    fixed = np.asarray(fixed_mask, bool).reshape(len(x), x.shape[2], y.shape[2])
    for c in range(len(x)):
        partner = y[c].T if axis == "X" else x[c].T
        blocked = np.all(fixed[c], axis=0 if axis == "X" else 1)
        # Only exact zero factor rows may be removed here. Nonzero rows can
        # cancel between rank terms and must go through the ordinary trace
        # intersection, not be clipped or subjected to a relaxed tolerance.
        blocked &= np.all(partner == 0., axis=1)
        if np.count_nonzero(~blocked) < partner.shape[1]:
            blocked[:] = False
        q_free, r = np.linalg.qr(partner[~blocked], mode="reduced")
        q = np.zeros_like(partner)
        q[~blocked] = q_free
        if axis == "X":
            y[c] = q.T
            x[c] = r @ x[c]
        else:
            x[c] = q.T
            y[c] = r @ y[c]
    return x, y


def _coefficient_null(frozen_profiles, fixed_entries):
    """Orthonormal coefficient nullspace of a frozen-partner trace block."""
    partner = np.asarray(frozen_profiles, dtype=float)
    mask = np.asarray(fixed_entries, dtype=bool)
    if (partner.ndim != 2 or mask.shape != (partner.shape[0],)
            or not np.isfinite(partner).all()):
        raise ValueError("MODAL_MVI_INVALID_FROZEN_PARTNER")
    rank = partner.shape[1]
    if not np.any(mask):
        return np.eye(rank)
    boundary = partner[mask, :]
    # All columns were normalized by the corresponding X/Y partner QR.
    # On a truly clamped endpoint their remaining entries can be 1e-15:
    # a rank test relative ONLY to boundary would treat that roundoff as
    # independent essential constraints and delete an admissible current
    # MVI field (Cammalleri TE1 ranks 2-4, frozen Y partner).
    #
    # Compare the boundary singular scale with the FULL frozen partner.
    # This is a numerical zero decision in the constraint assembly, not
    # a relaxation of mode orthogonality, eigenvalue, or K/M tolerances.
    partner_scale = float(np.linalg.norm(partner, ord=2))
    if partner_scale <= 0.:
        return np.eye(rank)
    q, r, _ = qr(boundary.T, mode="full", pivoting=True,
                 check_finite=False)
    diagonal = np.abs(np.diag(r))
    active = int(np.count_nonzero(
        diagonal > 1.e-12 * max(
            partner_scale, float(np.max(diagonal, initial=0.)))))
    return q[:, active:]


def fixed_trace_xy_basis(x, y, axis, fixed_mask, scales,
                         *, max_directions=1600):
    """Build the exact frozen X/Y variational intersection with nodal traces.

    Each group is confined to one component and one coordinate node. Small
    pivoted QR blocks are used only to select an independent basis in the
    original frozen-partner span. No full-grid nodal vectors are appended.
    The columns are Euclidean-orthonormal in dimensionless field coordinates,
    then multiplied by the original physical state scales.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("MODAL_MVI_INVALID_SEPARATED_FACTORS")
    ncomp, rank, nx = x.shape
    ns = y.shape[2]
    fixed = np.asarray(fixed_mask, dtype=bool)
    scale = np.asarray(scales, dtype=float)
    if (fixed.size != ncomp * nx * ns or scale.shape != (fixed.size,)
            or not np.isfinite(scale).all() or np.any(scale <= 0.)):
        raise ValueError("MODAL_MVI_INVALID_TRACE_OR_SCALES")
    fixed = fixed.reshape((ncomp, nx, ns))
    axis = str(axis).upper()
    if axis not in ("X", "Y"):
        raise ValueError("MODAL_MVI_INVALID_HALFSTEP_AXIS")
    directions = []
    for c in range(ncomp):
        count = nx if axis == "X" else ns
        partner = y[c].T if axis == "X" else x[c].T
        for index in range(count):
            blocked = (fixed[c, index, :] if axis == "X"
                       else fixed[c, :, index])
            coefficients = _coefficient_null(partner, blocked)
            if not coefficients.shape[1]:
                continue
            allowed = partner @ coefficients
            frame, triangular, _ = qr(
                allowed, mode="economic", pivoting=True,
                check_finite=False)
            diagonal = np.abs(np.diag(triangular))
            keep = int(np.count_nonzero(
                diagonal > 1.e-12 * np.max(diagonal, initial=0.)))
            for j in range(keep):
                vector = frame[:, j].copy()
                if np.max(np.abs(vector[blocked]), initial=0.) > 1.e-10:
                    raise ValueError("MODAL_MVI_FROZEN_PARTNER_TRACE_LOST")
                # Clear numerical roundoff only; no geometric projection.
                vector[blocked] = 0.
                directions.append((c, index, vector))
    if not directions:
        return np.empty((fixed.size, 0))
    if len(directions) > max_directions:
        raise ValueError(
            "MODAL_MVI_PROJECTED_DIRECTION_LIMIT: "
            f"{len(directions)} directions exceed {max_directions}; "
            "the full separated space was not replaced by Fourier sectors")
    result = np.zeros((fixed.size, len(directions)), dtype=float)
    row_offset = np.arange(nx, dtype=int) * ns
    for j, (c, idx, vector) in enumerate(directions):
        if axis == "X":
            start = (c * nx + idx) * ns
            result[start:start + ns, j] = vector
        else:
            rows = c * nx * ns + row_offset + idx
            result[rows, j] = vector
    if np.max(np.abs(result[fixed.ravel(), :]), initial=0.) > 1.e-12:
        raise ValueError("MODAL_MVI_INADMISSIBLE_CONSTRAINED_BASIS")
    return scale[:, None] * result


__all__ = ("fixed_trace_xy_basis", "normalize_modal_partner")

