#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strong-form CUF/FDM incremental J2 adapter for TE1--TE4.

Supported constitutive/geometry pairings:
  LINEAR       -> small-strain 3-D J2,
  VON_KARMAN   -> small-strain 3-D J2 with CUF VK virtual work,
  GREEN_LAGRANGE -> multiplicative finite-strain 3-D J2, P=F S.

The global Jacobian differentiates the complete residual at one fixed accepted
history.  It is intentionally one-sided at the elastic/plastic corner.  A
returned trial history is never committed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import final_numeric_scaling as nscale
import final_progress as fprog
import final_residual_regions as rregions
import final_residual_scaling as rscale
import final_v3_fd as v3fd
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as gbc
import cylinder_cuf_stage7j_j2_material as jm
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as fdl
from final_j2_incremental import J2Parameters, J2State
from final_j2_finite_strain import FiniteJ2State


class CUFStrongFDMJ2Error(RuntimeError):
    pass


class CUFStrongFDMJ2SystemSingular(CUFStrongFDMJ2Error):
    pass


@dataclass
class CUFStrongFDMJ2Result:
    q: np.ndarray
    converged: bool
    status: str
    iterations: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    increment_inf: float
    history: list
    trial_state: J2State | FiniteJ2State
    material_state: jm.CUFJ2MaterialState
    resultants: object
    residual_inf_interior: float
    residual_inf_boundary: float
    residual_inf_load_front: float
    residual_convergence_inf: float
    residual_convergence_relative: float
    residual_inf_by_block: dict


def strong_collocation_residual_j2(
    q, ops, basis, bc, *, geometry, committed, parameters: J2Parameters,
    quadrature, generalized_load,
):
    q = fdl._validate_q(q, ops, basis)
    if bc.m != basis.m:
        raise ValueError("basis/BC m mismatch")
    bc.check_rigid_mode_requirement()
    p = np.asarray(generalized_load, float)
    if p.shape != q.shape or not np.isfinite(p).all():
        raise ValueError(f"generalized_load must be finite with shape {q.shape}")

    qx, qs = fdl.surface_first_derivatives(q, ops, basis)
    mode = str(geometry).upper()
    if mode in ("LINEAR", "VON_KARMAN"):
        jout = jm.resultants_small_strain(
            q, qx, qs, geometry=mode, R=ops.grid.radius,
            committed=committed, parameters=parameters,
            quadrature=quadrature, basis=basis,
        )
    elif mode == "GREEN_LAGRANGE":
        jout = jm.resultants_finite_strain(
            q, qx, qs, R=ops.grid.radius, committed=committed,
            parameters=parameters, quadrature=quadrature, basis=basis,
        )
    else:
        raise ValueError("J2 CUF geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")

    raw = g7.general_residuals(
        jout.resultants, Dx=ops.Dx, Ds=ops.Ds, generalized_load=p
    )
    assembled = gbc.assemble_strong_residual_with_end_bcs(
        raw, q, jout.resultants, bc, reject_fully_free=True
    )
    boundary = gbc.full_boundary_residuals(q, jout.resultants, bc)
    bvec = np.concatenate((
        np.asarray(boundary["left"], float).ravel(),
        np.asarray(boundary["right"], float).ravel(),
    ))
    interior = np.asarray(raw[..., 1:-1], float).ravel()
    return assembled.reshape(-1), raw, jout, bvec, interior


def finite_difference_jacobian(
    q, ops, basis, bc, *, geometry, committed, parameters: J2Parameters,
    quadrature, generalized_load, fd_step, base_residual=None,
):
    """One-sided derivative of the exact J2 residual at fixed committed history."""
    q = fdl._validate_q(q, ops, basis)
    x = fdl.flatten_q(q, ops, basis)
    r0 = (
        np.asarray(base_residual, float).reshape(-1)
        if base_residual is not None else
        strong_collocation_residual_j2(
            q, ops, basis, bc, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
            generalized_load=generalized_load,
        )[0]
    )
    if r0.shape != x.shape:
        raise CUFStrongFDMJ2Error("CUF J2 base residual size mismatch")
    jacobian = np.empty((x.size, x.size), float)
    scales = v3fd.cuf_characteristic_scale(q, ops, quadrature).reshape(-1)
    steps = v3fd.relative_steps(x, scales, fd_step)
    for col, step in enumerate(steps):
        xp = x.copy()
        xp[col] += step
        rp = strong_collocation_residual_j2(
            fdl.unflatten_q(xp, ops, basis), ops, basis, bc,
            geometry=geometry, committed=committed, parameters=parameters,
            quadrature=quadrature, generalized_load=generalized_load,
        )[0]
        jacobian[:, col] = (rp - r0) / step
    return jacobian, r0


def _measure_data(q, ops, basis, bc, quadrature, E0, generalized_load, residual):
    h = v3fd.thickness_from_quadrature(quadrature)
    measure_kw = dict(
        basis_m=basis.m, ns=ops.grid.ns, nx=ops.grid.nx,
        L=ops.grid.length_x, bc=bc, E0=E0, h=h,
    )
    residual_m = rscale.cuf_scaled_residual_for_measure(residual, **measure_kw)
    load_m = rscale.cuf_scaled_strong_load_for_measure(
        generalized_load, basis_m=basis.m, ns=ops.grid.ns,
        nx=ops.grid.nx, L=ops.grid.length_x,
    )
    return measure_kw, residual_m, load_m


def _result(q, converged, status, iterations, metrics, scale, increment, history, material, basis, ops):
    return CUFStrongFDMJ2Result(
        q=np.asarray(q, float), converged=bool(converged), status=str(status),
        iterations=int(iterations), residual_inf=float(metrics.residual_inf),
        residual_relative=float(metrics.residual_relative),
        residual_scale=float(scale), increment_inf=float(increment),
        history=list(history), trial_state=material.trial_state,
        material_state=material.material_state,
        resultants=material.resultants,
        residual_inf_interior=float(metrics.residual_inf_interior),
        residual_inf_boundary=float(metrics.residual_inf_boundary),
        residual_inf_load_front=float(metrics.residual_inf_load_front),
        residual_convergence_inf=float(metrics.residual_convergence_inf),
        residual_convergence_relative=float(metrics.residual_convergence_relative),
        residual_inf_by_block={},
    )


def solve_j2_strong_fdm(
    initial_q, ops, basis, bc, *, geometry, committed, parameters: J2Parameters,
    quadrature, generalized_load, settings, E0, load_front_surface_mask=None,
):
    """Solve one CUF J2 increment without committing material history."""
    q = fdl._validate_q(initial_q, ops, basis).copy()
    r0, _raw0, jout0, _b0, _i0 = strong_collocation_residual_j2(
        q, ops, basis, bc, geometry=geometry, committed=committed,
        parameters=parameters, quadrature=quadrature,
        generalized_load=generalized_load,
    )
    measure_kw, r0m, load_m = _measure_data(
        q, ops, basis, bc, quadrature, E0, generalized_load, r0
    )
    scale = rscale.residual_scale(load_m, r0m)
    limit = rscale.threshold(settings.residual_tolerance_relative, scale)
    masks = rregions.resolve_front_exclusion(
        rregions.cuf_masks(ops, basis, load_front_surface_mask), generalized_load
    )
    history = []
    last_increment = np.inf

    def metrics_for(residual):
        measured = rscale.cuf_scaled_residual_for_measure(residual, **measure_kw)
        return rregions.metrics(measured, masks, scale), measured

    for iteration in range(1, int(settings.max_iterations) + 1):
        if iteration == 1:
            residual, jout = r0, jout0
        else:
            residual, _raw, jout, _b, _i = strong_collocation_residual_j2(
                q, ops, basis, bc, geometry=geometry, committed=committed,
                parameters=parameters, quadrature=quadrature,
                generalized_load=generalized_load,
            )
        metrics, measured = metrics_for(residual)
        if rregions.converged(metrics, limit):
            out = _result(q, True, f"CONVERGED_CUF_STRONG_FDM_{str(geometry).upper()}_J2",
                          iteration - 1, metrics, scale, 0.0, history, jout, basis, ops)
            out.residual_inf_by_block = rscale.cuf_residual_inf_by_block(
                measured, basis_m=basis.m, ns=ops.grid.ns, nx=ops.grid.nx
            )
            return out

        jacobian, _ = finite_difference_jacobian(
            q, ops, basis, bc, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
            generalized_load=generalized_load, fd_step=settings.fd_step,
            base_residual=residual,
        )
        try:
            dq, _, _, _ = nscale.equilibrated_solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMJ2SystemSingular(
                f"SINGULAR_CUF_STRONG_FDM_{str(geometry).upper()}_J2_JACOBIAN / iteration={iteration}"
            ) from exc
        if not np.isfinite(dq).all():
            raise CUFStrongFDMJ2Error(
                f"NONFINITE_CUF_STRONG_FDM_{str(geometry).upper()}_J2_INCREMENT / iteration={iteration}"
            )
        last_increment = float(np.max(np.abs(dq)))
        q = fdl.unflatten_q(fdl.flatten_q(q, ops, basis) + dq, ops, basis)
        residual_new, _raw, jout_new, _b, _i = strong_collocation_residual_j2(
            q, ops, basis, bc, geometry=geometry, committed=committed,
            parameters=parameters, quadrature=quadrature,
            generalized_load=generalized_load,
        )
        metrics_new, measured_new = metrics_for(residual_new)
        row = {"iteration": int(iteration), "increment_inf": last_increment, "residual_scale": scale}
        row.update(rregions.as_dict(metrics_new))
        row["residual_inf_by_block"] = rscale.cuf_residual_inf_by_block(
            measured_new, basis_m=basis.m, ns=ops.grid.ns, nx=ops.grid.nx
        )
        history.append(row)
        w = np.asarray(q[2, 0], float)
        js, ix = np.unravel_index(int(np.argmax(np.abs(w))), w.shape)
        fprog.emit(
            "newton_iteration", stage=f"NEWTON_CUF_{str(geometry).upper()}_J2",
            iteration=int(iteration), residual_inf=float(metrics_new.residual_inf),
            residual_relative=float(metrics_new.residual_relative),
            residual_convergence_inf=float(metrics_new.residual_convergence_inf),
            residual_convergence_relative=float(metrics_new.residual_convergence_relative),
            residual_inf_interior=float(metrics_new.residual_inf_interior),
            residual_inf_boundary=float(metrics_new.residual_inf_boundary),
            residual_inf_load_front=float(metrics_new.residual_inf_load_front),
            increment_inf=last_increment, w_abs_max=float(np.max(np.abs(w))),
            w_at_abs_max=float(w[js, ix]), wmax_is=int(js), wmax_ix=int(ix),
        )
        if rregions.converged(metrics_new, limit) and last_increment <= settings.increment_tolerance:
            out = _result(q, True, f"CONVERGED_CUF_STRONG_FDM_{str(geometry).upper()}_J2",
                          iteration, metrics_new, scale, last_increment, history, jout_new, basis, ops)
            out.residual_inf_by_block = row["residual_inf_by_block"]
            return out

    residual_fin, _raw, jout_fin, _b, _i = strong_collocation_residual_j2(
        q, ops, basis, bc, geometry=geometry, committed=committed,
        parameters=parameters, quadrature=quadrature,
        generalized_load=generalized_load,
    )
    metrics_fin, measured_fin = metrics_for(residual_fin)
    out = _result(q, False, f"MAX_CUF_STRONG_FDM_{str(geometry).upper()}_J2_NEWTON_ITERATIONS",
                  int(settings.max_iterations), metrics_fin, scale, last_increment, history, jout_fin, basis, ops)
    out.residual_inf_by_block = rscale.cuf_residual_inf_by_block(
        measured_fin, basis_m=basis.m, ns=ops.grid.ns, nx=ops.grid.nx
    )
    return out


__all__ = [
    "CUFStrongFDMJ2Error", "CUFStrongFDMJ2SystemSingular",
    "CUFStrongFDMJ2Result", "strong_collocation_residual_j2",
    "finite_difference_jacobian", "solve_j2_strong_fdm",
]