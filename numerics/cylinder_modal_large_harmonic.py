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


__all__ = ("mechanical_fixed_trace_mask",)
