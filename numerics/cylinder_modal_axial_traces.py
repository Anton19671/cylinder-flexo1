"""Exact, uniform axial trace spaces for large-grid MODAL MVI KL clamps.

Physical supports are taken from the SAME compiled variational boundary
equations as MKR/FDM. A KL end slope is a coupled linear row, not a fixed
neighboring nodal value. Only component-local, circumferentially uniform
homogeneous traces are supported. Other constraints FAIL CLOSED.

The tensor products below are precisely the constrained frozen-factor X/Y
variation subspaces of the classical variational iteration scheme.
No Fourier sector is imposed during the Y/X sweeps, and no full-grid
eigenvalue problem or MKR modes are used.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from cylinder_constraint_row import combined_constraint_entries
from cylinder_variational_boundary import compile_boundary_program


def _orthogonal_columns(matrix, *, scale=None):
    a = np.asarray(matrix, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("MODAL_MVI_AXIAL_INVALID_PROFILE")
    if not a.shape[1]:
        return np.empty((a.shape[0], 0))
    q, r, _ = qr(a, mode="economic", pivoting=True, check_finite=False)
    diag = np.abs(np.diag(r))
    reference = max(float(np.max(diag, initial=0.)),
                    float(np.linalg.norm(a, ord=2)) if scale is None else float(scale))
    count = int(np.count_nonzero(diag > 1.e-12 * reference))
    return q[:, :count].copy()


def uniform_axial_mechanical_traces(problem, mechanical):
    """Return per-component axial null bases, fixed-node mask, and real full C.

    Exactly the same boundary rows as the full variational solver. Uniform
    angular sectors are allowed only after independently checking every
    circumferential position, not inferred from the C-C case label.
    """
    f = problem.functional
    q = np.asarray(mechanical, float)
    if (q.ndim != 3 or not np.isfinite(q).all()
            or q.shape != (f.n_mechanical, *f.ops.shape)
            or f.electrical_on):
        raise ValueError("MODAL_MVI_AXIAL_TRACES_REQUIRES_MECHANICAL_ONLY")
    ncomp, nx, ns = q.shape
    edge = compile_boundary_program(
        f, q, None, spec=problem.spec,
        calculation_kind=problem.kind, check_charge_balance=False)
    if edge.gauge_required:
        raise ValueError("MODAL_MVI_AXIAL_GAUGE_UNSUPPORTED")
    grouped = [[[] for _ in range(ns)] for _ in range(ncomp)]
    fixed = np.zeros(q.size, dtype=bool)
    rows = []
    for row in edge.constraints:
        if row.field != "mechanical" or abs(float(row.prescribed)) > 1.e-13:
            raise ValueError("MODAL_MVI_AXIAL_NONHOMOGENEOUS_OR_ELECTRIC_TRACE")
        indexes, weights = combined_constraint_entries(
            row.indices, row.weights, q.size, offset=0)
        indexes = np.asarray(indexes, dtype=int)
        weights = np.asarray(weights, dtype=float)
        if (not len(indexes) or not np.isfinite(weights).all()
                or not np.isfinite(indexes).all()):
            raise ValueError("MODAL_MVI_AXIAL_INVALID_TRACE_ROW")
        comp = indexes // (nx * ns)
        angular = indexes % ns
        if (np.any(comp != comp[0]) or np.any(angular != angular[0])
                or comp[0] >= ncomp):
            raise ValueError("MODAL_MVI_AXIAL_NONUNIFORM_OR_COUPLED_COMPONENTS")
        norm = float(np.linalg.norm(weights))
        if not np.isfinite(norm) or norm <= 0.:
            raise ValueError("MODAL_MVI_AXIAL_ZERO_TRACE")
        c = int(comp[0])
        j = int(angular[0])
        axial = (indexes // ns) % nx
        profile = np.zeros(nx)
        np.add.at(profile, axial, weights / norm)
        size = float(np.linalg.norm(profile))
        if not np.isfinite(size) or size < 1.e-14:
            raise ValueError("MODAL_MVI_AXIAL_DEGENERATE_TRACE")
        profile /= size
        first = int(np.flatnonzero(np.abs(profile) > 1.e-14)[0])
        if profile[first] < 0.:
            profile = -profile
        grouped[c][j].append(profile)
        rows.append((indexes, weights / norm))
        if len(indexes) == 1:
            fixed[indexes[0]] = True
    if not rows:
        raise ValueError("MODAL_MVI_AXIAL_NO_ESSENTIAL_TRACES")
    constrained = []
    for c in range(ncomp):
        by_angle = [sorted(grp, key=lambda v: tuple(np.round(v, 12)))
                    for grp in grouped[c]]
        reference = by_angle[0]
        if any(len(other) != len(reference) or any(
                not np.allclose(a, b, rtol=1.e-10, atol=1.e-12)
                for a, b in zip(reference, other))
               for other in by_angle[1:]):
            raise ValueError("MODAL_MVI_AXIAL_NONUNIFORM_ANGULAR_TRACES")
        if reference:
            trace = np.stack(reference)
            frame, tri, _ = qr(trace.T, mode="full",
                               pivoting=True, check_finite=False)
            diagonal = np.abs(np.diag(tri))
            count = int(np.count_nonzero(
                diagonal > 1.e-11 * max(np.max(diagonal, initial=0.), 1.e-30)))
            basis = frame[:, count:].copy()
            if np.max(np.abs(trace @ basis), initial=0.) > 1.e-10:
                raise ValueError("MODAL_MVI_AXIAL_NULLSPACE_CERTIFICATE_FAILED")
        else:
            basis = np.eye(nx)
        if not basis.shape[1]:
            raise ValueError("MODAL_MVI_AXIAL_NO_MECHANICAL_DIRECTIONS")
        constrained.append(basis)
    # Lightweight full physical trace check of every MVI trial field.
    cm = np.zeros((len(rows), q.size))
    for j, (indexes, weights) in enumerate(rows):
        np.add.at(cm[j], indexes, weights)
    for c, b in enumerate(constrained):
        for j in range(ns):
            inserted = np.zeros((ncomp, nx, ns))
            inserted[c, :, j] = b[:, 0]
            if np.max(np.abs(cm @ inserted.ravel()), initial=0.) > 1.e-9:
                raise ValueError("MODAL_MVI_AXIAL_PHYSICAL_TRACE_MISMATCH")
    return constrained, fixed, cm


def _check_profiles(x, y, axial_bases, scales):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("MODAL_MVI_AXIAL_INVALID_XY")
    nc, _, nx = x.shape
    ns = y.shape[-1]
    scales = np.asarray(scales, float)
    if (len(axial_bases) != nc or scales.shape != (nc * nx * ns,)
            or np.any(scales <= 0.) or not np.isfinite(scales).all()):
        raise ValueError("MODAL_MVI_AXIAL_INVALID_SCALES_OR_BASES")
    for basis in axial_bases:
        if (basis.ndim != 2 or basis.shape[0] != nx
                or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]),
                                   atol=2.e-10, rtol=0.)):
            raise ValueError("MODAL_MVI_AXIAL_BASIS_NOT_ORTHONORMAL")
    return x, y, nc, nx, ns, scales


def _axial_partner_frame(partner, basis):
    """Exact span(partner) intersection with the physical axial nullspace."""
    outside = partner - basis @ (basis.T @ partner)
    # QR of constraint amplitudes, not an SVD or global modal problem.
    frame, tri, _ = qr(outside.T, mode="full",
                        pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(tri))
    norm = float(np.linalg.norm(partner, ord=2))
    rank = int(np.count_nonzero(
        diagonal > 1.e-12 * max(norm, np.max(diagonal, initial=0.))))
    allowed = _orthogonal_columns(partner @ frame[:, rank:], scale=norm)
    if (allowed.shape[1] and np.linalg.norm(
            allowed - basis @ (basis.T @ allowed)) > 1.e-10 * max(
                1., float(np.linalg.norm(allowed)))):
        raise ValueError("MODAL_MVI_AXIAL_Y_TRACE_INTERSECTION_FAILED")
    return allowed


def _assemble_columns(nc, nx, ns, axial_bases, angular_bases,
                      scales, *, max_directions):
    count = sum(axial_bases[c].shape[1] * angular_bases[c].shape[1]
                for c in range(nc))
    if count > max_directions:
        raise ValueError("MODAL_MVI_PROJECTED_DIRECTION_LIMIT")
    if not count:
        return np.empty((len(scales), 0))
    result = np.zeros((len(scales), count))
    cursor = 0
    for c in range(nc):
        for i in range(axial_bases[c].shape[1]):
            for j in range(angular_bases[c].shape[1]):
                vector = (axial_bases[c][:, i, None] *
                          angular_bases[c][:, j][None, :])
                result[c * nx * ns:(c + 1) * nx * ns, cursor] = vector.ravel()
                cursor += 1
    return scales[:, None] * result


def axial_trace_xy_basis(x, y, axis, axial_bases, scales,
                         *, max_directions=1600):
    """True frozen Y/X space intersected with all KL axial trace rows."""
    x, y, nc, nx, ns, scales = _check_profiles(
        x, y, axial_bases, scales)
    axis = str(axis).upper()
    if axis == "X":
        angular_bases = [_orthogonal_columns(y[c].T) for c in range(nc)]
        return _assemble_columns(nc, nx, ns, axial_bases,
                                 angular_bases, scales,
                                 max_directions=max_directions)
    if axis != "Y":
        raise ValueError("MODAL_MVI_INVALID_HALFSTEP_AXIS")
    rows = np.arange(nx) * ns
    count = nc * ns * x.shape[1]
    if count > max_directions:
        raise ValueError("MODAL_MVI_PROJECTED_DIRECTION_LIMIT")
    result = np.zeros((len(scales), count))
    cursor = 0
    for c in range(nc):
        frame = _axial_partner_frame(x[c].T, axial_bases[c])
        for angle in range(ns):
            for k in range(frame.shape[1]):
                result[c * nx * ns + rows + angle, cursor] = frame[:, k]
                cursor += 1
    return scales[:, None] * result[:, :cursor]


def axial_trace_harmonic_x_basis(y, harmonic, axial_bases, scales,
                                  *, max_directions=1600):
    """Initial harmonic X conditioning without changing the frozen-Y span."""
    from cylinder_modal_harmonic_spectrum import _harmonic_phase_basis
    yy = np.asarray(y, float)
    if yy.ndim != 3 or not np.isfinite(yy).all():
        raise ValueError("MODAL_MVI_AXIAL_INVALID_HARMONIC_PARTNER")
    nc, rank, ns = yy.shape
    nx = axial_bases[0].shape[0]
    dummy_x = np.zeros((nc, rank, nx))
    _, _, _, _, _, scales = _check_profiles(
        dummy_x, yy, axial_bases, scales)
    phase = _harmonic_phase_basis(ns, int(harmonic))
    angular = []
    for c in range(nc):
        partner = yy[c].T
        off = partner - phase.T @ (phase @ partner)
        q, r, _ = qr(off.T, mode="full", pivoting=True,
                      check_finite=False)
        d = np.abs(np.diag(r))
        norm = float(np.linalg.norm(partner, ord=2))
        count = int(np.count_nonzero(
            d > 1.e-12 * max(norm, np.max(d, initial=0.))))
        frame = _orthogonal_columns(partner @ q[:, count:], scale=norm)
        if np.linalg.norm(frame - phase.T @ (phase @ frame)) > (
                1.e-10 * max(1., np.linalg.norm(frame))):
            raise ValueError("MODAL_MVI_AXIAL_INITIAL_HARMONIC_ESCAPED_SPAN")
        angular.append(frame)
    return _assemble_columns(nc, nx, ns, axial_bases, angular, scales,
                             max_directions=max_directions)


__all__ = ("uniform_axial_mechanical_traces", "axial_trace_xy_basis",
           "axial_trace_harmonic_x_basis")
