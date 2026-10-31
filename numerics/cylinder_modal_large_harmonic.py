"""Exact homogeneous trace extraction for bounded large modal MVI harmonic scans."""
from __future__ import annotations

import numpy as np
from cylinder_variational_boundary import compile_boundary_program
from cylinder_constraint_row import combined_constraint_entries


def mechanical_fixed_trace_mask(problem, mechanical):
    """Accept single-DOF zero mechanical supports only; no penalty or nodal lift."""
    f = problem.functional
    if f.electrical_on:
        raise ValueError("MODAL_LARGE_HARMONIC_REQUIRES_MECHANICAL_ONLY")
    edge = compile_boundary_program(
        f, mechanical, None, spec=problem.spec,
        calculation_kind=problem.kind, check_charge_balance=False)
    if edge.gauge_required:
        raise ValueError("MODAL_LARGE_HARMONIC_UNSUPPORTED_ELECTRIC_GAUGE")
    mask = np.zeros(mechanical.size, dtype=bool)
    for row in edge.constraints:
        if row.field != "mechanical" or abs(float(row.prescribed)) > 1.e-13:
            raise ValueError("MODAL_LARGE_HARMONIC_UNSUPPORTED_PRESCRIBED_TRACE")
        indexes, weights = combined_constraint_entries(
            row.indices, row.weights, mechanical.size, offset=0)
        if len(indexes) != 1 or len(weights) != 1 or abs(float(weights[0])) < 1.e-13:
            raise ValueError("MODAL_LARGE_HARMONIC_UNSUPPORTED_COUPLED_TRACE")
        idx = int(indexes[0])
        if mask[idx]:
            raise ValueError("MODAL_LARGE_HARMONIC_DUPLICATE_TRACE")
        mask[idx] = True
    if not mask.any():
        raise ValueError("MODAL_LARGE_HARMONIC_WITHOUT_FIXED_TRACES")
    return mask


# Bounded Fourier-only modal MVI extension. These are representation/memory
# limits, not a physical assumption about equal axial/circumferential steps.
MAX_MODAL_MVI_AXIAL_NODES = 21
MAX_MODAL_MVI_CIRCUMFERENTIAL_NODES = 63
MAX_MODAL_MVI_SURFACE_NODES = 21 * 63
MAX_MODAL_MVI_MECHANICAL_DOFS = 15 * 21 * 63


def bounded_harmonic_modal_grid(nx, ns, n_components):
    """Whether a MACRO Fourier MVI modal grid is within safe storage limits."""
    return (all(type(value) is int for value in (nx, ns, n_components))
            and 9 <= nx <= MAX_MODAL_MVI_AXIAL_NODES
            and 9 <= ns <= MAX_MODAL_MVI_CIRCUMFERENTIAL_NODES
            and ns % 2 == 1 and 1 <= n_components <= 15
            and nx * ns <= MAX_MODAL_MVI_SURFACE_NODES
            and nx * ns * n_components <= MAX_MODAL_MVI_MECHANICAL_DOFS)


def projected_harmonic_y_basis(x, fixed_mask, scales, wave_number):
    """Exact intersection of frozen-X Y variations and homogeneous traces.

    Y variations are sums of X[c,r,x] times Fourier phases, with X frozen.
    If a frozen X factor violates an end trace, merely zeroing that factor
    at the end would add directions outside the actual Y variation space.
    Instead constrain its rank coefficients, F@a=0, on the fixed axial
    nodes and orthonormalize only X.T@null(F). No nodal/FDM enrichment.
    """
    from scipy.linalg import qr
    from cylinder_modal_harmonic_spectrum import _harmonic_phase_basis
    components, rank, nx = x.shape
    ns = fixed_mask.size // (components * nx)
    fixed = fixed_mask.reshape(components, nx, ns)
    if np.any(fixed != fixed[:, :, :1]):
        raise ValueError("MODAL_LARGE_HARMONIC_NONUNIFORM_ANGULAR_TRACES")
    phases = _harmonic_phase_basis(ns, wave_number)
    result = []
    for c in range(components):
        axial = x[c].T
        fixed_axial = fixed[c, :, 0]
        boundary = axial[fixed_axial, :].copy()
        # Ignore only roundoff of the frozen factors at a homogeneous trace.
        # Unit-norm QR partner completion can carry small nonzero entries.
        floor = 1.e-13 * max(1., float(np.linalg.norm(axial)))
        boundary[np.abs(boundary) <= floor] = 0.
        if boundary.size:
            q_constraints, r_constraints, _ = qr(
                boundary.T, mode="full", pivoting=True, check_finite=False)
            diag = np.abs(np.diag(r_constraints))
            constrained = int(np.count_nonzero(
                diag > 1.e-12 * np.max(diag, initial=0.)))
            admissible_factors = q_constraints[:, constrained:]
        else:
            admissible_factors = np.eye(rank)
        if not admissible_factors.shape[1]:
            continue
        allowed_axial = axial @ admissible_factors
        q, r, _ = qr(allowed_axial, mode="economic", pivoting=True,
                     check_finite=False)
        diag = np.abs(np.diag(r))
        keep = int(np.count_nonzero(
            diag > 1.e-12 * np.max(diag, initial=0.)))
        if not keep:
            continue
        # All output directions must remain in span(frozen X), and satisfy
        # the trace. Only roundoff may be cleared on a fixed axial node.
        if np.max(np.abs(q[fixed_axial, :keep]), initial=0.) > 1.e-10:
            raise ValueError("MODAL_LARGE_HARMONIC_Y_TRACE_SUBSPACE_INCONSISTENT")
        q[fixed_axial, :keep] = 0.
        for k in range(keep):
            for phase in phases:
                column = np.zeros((components, nx, ns))
                column[c] = q[:, k, None] * phase[None, :]
                result.append(column.ravel())
    if not result:
        return np.empty((scales.size, 0))
    return scales[:, None] * np.column_stack(result)


__all__ = ("mechanical_fixed_trace_mask", "bounded_harmonic_modal_grid",
           "projected_harmonic_y_basis",
           "MAX_MODAL_MVI_AXIAL_NODES", "MAX_MODAL_MVI_CIRCUMFERENTIAL_NODES",
           "MAX_MODAL_MVI_SURFACE_NODES", "MAX_MODAL_MVI_MECHANICAL_DOFS")
