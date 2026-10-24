#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KL strong-form FDM: von Karman kinematics with incremental J2 plasticity.

This module is deliberately independent from the Birger solver.  J2 is updated
at every thickness integration point by the small-strain plane-stress radial
return mapping.  The caller owns the accepted history: all residual evaluations
and all Jacobian columns start from the same committed state.

The global Jacobian is currently a directional derivative of this exact
non-mutating residual.  This is valid across the elastic/plastic switching
surface; it must not be confused with, or replaced by, a Birger secant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import final_numeric_scaling as nscale
import final_progress as fprog
import final_residual_regions as rregions
import final_residual_scaling as rscale
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage8j_j2_material as jm
import cylinder_kl_stage9b_strong_fdm_linear as fl
from final_j2_incremental import J2Parameters, J2State


class KLStrongFDMVKJ2Error(RuntimeError):
    pass


class KLStrongFDMVKJ2SystemSingular(KLStrongFDMVKJ2Error):
    pass


@dataclass
class KLStrongFDMVKJ2Result:
    U: np.ndarray
    converged: bool
    status: str
    iterations: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    increment_inf: float
    history: list
    trial_state: J2State
    material_state: jm.KLJ2MaterialState
    resultants: object
    residual_inf_interior: float
    residual_inf_boundary: float
    residual_inf_load_front: float
    residual_convergence_inf: float
    residual_convergence_relative: float


def strong_collocation_residual_vk_j2(
    U, ops, bc, *, committed: J2State, parameters: J2Parameters,
    quadrature, generalized_load,
):
    fl._validate_grid(ops)
    U = fl._validate_U(U, ops)
    bc.check_rigid_mode_requirement()
    p = np.asarray(generalized_load, float)
    expected = (3, ops.grid.ns, ops.grid.nx)
    if p.shape != expected or not np.isfinite(p).all():
        raise ValueError(f"generalized_load must be finite with shape {expected}")

    kin = kl.field_kinematics(U, ops, geometry="VON_KARMAN")
    jout = jm.resultants_from_kinematics(
        kin, committed=committed, parameters=parameters, quadrature=quadrature
    )
    nm = jout.resultants
    strong = kl.strong_residual_from_resultants(
        kin, nm, ops, geometry="VON_KARMAN", generalized_load=p
    )
    ql = kb.end_mechanical_quantities(
        kin, nm, ops, geometry="VON_KARMAN", side="LEFT"
    )
    qr = kb.end_mechanical_quantities(
        kin, nm, ops, geometry="VON_KARMAN", side="RIGHT"
    )

    ns, nx = ops.grid.ns, ops.grid.nx
    lu = fl._bc_component_residual(bc.left.u, ql.u, ql.T_u, ns)
    ru = fl._bc_component_residual(bc.right.u, qr.u, qr.T_u, ns)
    lv = fl._bc_component_residual(bc.left.v, ql.v, ql.T_v, ns)
    rv = fl._bc_component_residual(bc.right.v, qr.v, qr.T_v, ns)
    lw = fl._bc_component_residual(bc.left.w, ql.w, ql.T_w, ns)
    rw = fl._bc_component_residual(bc.right.w, qr.w, qr.T_w, ns)
    ls = fl._bc_component_residual(bc.left.slope, ql.theta_n, ql.T_theta, ns)
    rs = fl._bc_component_residual(bc.right.slope, qr.theta_n, qr.T_theta, ns)

    rows = []
    for js in range(ns):
        rows.append(lu[js]); rows.extend(strong[0, js, 1:nx-1]); rows.append(ru[js])
    for js in range(ns):
        rows.append(lv[js]); rows.extend(strong[1, js, 1:nx-1]); rows.append(rv[js])
    for js in range(ns):
        rows.append(lw[js]); rows.append(ls[js])
        rows.extend(strong[2, js, 2:nx-2])
        rows.append(rs[js]); rows.append(rw[js])
    residual = np.asarray(rows, float)
    if residual.shape != (3 * ns * nx,):
        raise KLStrongFDMVKJ2Error("VK J2 KL residual size mismatch")
    return residual, strong, jout, kin


def finite_difference_jacobian(
    U, ops, bc, *, committed: J2State, parameters: J2Parameters,
    quadrature, generalized_load, fd_step: float, base_residual=None,
):
    """One-sided derivative of the exact fixed-history return-map residual."""
    U = fl._validate_U(U, ops)
    x = fl.flatten_U(U, ops)
    residual0 = (
        np.asarray(base_residual, float).reshape(-1)
        if base_residual is not None else
        strong_collocation_residual_vk_j2(
            U, ops, bc, committed=committed, parameters=parameters,
            quadrature=quadrature, generalized_load=generalized_load,
        )[0]
    )
    if residual0.shape != x.shape:
        raise KLStrongFDMVKJ2Error("VK J2 base residual size mismatch")
    jacobian = np.empty((x.size, x.size), float)
    for col in range(x.size):
        step = float(fd_step) * max(1.0, abs(float(x[col])))
        xp = x.copy()
        xp[col] += step
        rp = strong_collocation_residual_vk_j2(
            fl.unflatten_U(xp, ops), ops, bc,
            committed=committed, parameters=parameters,
            quadrature=quadrature, generalized_load=generalized_load,
        )[0]
        jacobian[:, col] = (rp - residual0) / step
    return jacobian, residual0


def _result(U, converged, status, iterations, metrics, scale, increment, history, material):
    return KLStrongFDMVKJ2Result(
        U=np.asarray(U, float),
        converged=bool(converged),
        status=str(status),
        iterations=int(iterations),
        residual_inf=float(metrics.residual_inf),
        residual_relative=float(metrics.residual_relative),
        residual_scale=float(scale),
        increment_inf=float(increment),
        history=list(history),
        trial_state=material.trial_state,
        material_state=material.material_state,
        resultants=material.resultants,
        residual_inf_interior=float(metrics.residual_inf_interior),
        residual_inf_boundary=float(metrics.residual_inf_boundary),
        residual_inf_load_front=float(metrics.residual_inf_load_front),
        residual_convergence_inf=float(metrics.residual_convergence_inf),
        residual_convergence_relative=float(metrics.residual_convergence_relative),
    )


def solve_vk_j2_strong_fdm(
    initial_U, ops, bc, *, committed: J2State, parameters: J2Parameters,
    quadrature, generalized_load, settings, load_front_surface_mask=None,
):
    """Solve a single VK J2 increment without committing plastic history."""
    U = fl._validate_U(initial_U, ops).copy()
    R0, _strong0, jout0, _kin0 = strong_collocation_residual_vk_j2(
        U, ops, bc, committed=committed, parameters=parameters,
        quadrature=quadrature, generalized_load=generalized_load,
    )
    scale = rscale.residual_scale(generalized_load, R0)
    limit = rscale.threshold(settings.residual_tolerance_relative, scale)
    regions = rregions.resolve_front_exclusion(
        rregions.kl_masks(ops, load_front_surface_mask), generalized_load
    )
    history = []
    last_increment = np.inf
    last_jout = jout0

    for iteration in range(1, int(settings.max_iterations) + 1):
        if iteration == 1:
            residual, jout = R0, jout0
        else:
            residual, _strong, jout, _kin = strong_collocation_residual_vk_j2(
                U, ops, bc, committed=committed, parameters=parameters,
                quadrature=quadrature, generalized_load=generalized_load,
            )
        metrics = rregions.metrics(residual, regions, scale)
        last_jout = jout
        if rregions.converged(metrics, limit):
            return _result(
                U, True, "CONVERGED_STRONG_FDM_VK_J2", iteration - 1,
                metrics, scale, 0.0, history, jout
            )

        jacobian, _ = finite_difference_jacobian(
            U, ops, bc, committed=committed, parameters=parameters,
            quadrature=quadrature, generalized_load=generalized_load,
            fd_step=settings.fd_step, base_residual=residual,
        )
        try:
            dx, _, _, _ = nscale.equilibrated_solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise KLStrongFDMVKJ2SystemSingular(
                f"SINGULAR_STRONG_FDM_VK_J2_JACOBIAN / iteration={iteration}"
            ) from exc
        if not np.isfinite(dx).all():
            raise KLStrongFDMVKJ2Error(
                f"NONFINITE_STRONG_FDM_VK_J2_INCREMENT / iteration={iteration}"
            )

        last_increment = float(np.max(np.abs(dx)))
        U = fl.unflatten_U(fl.flatten_U(U, ops) + dx, ops)
        residual_new, _strong, jout_new, _kin = strong_collocation_residual_vk_j2(
            U, ops, bc, committed=committed, parameters=parameters,
            quadrature=quadrature, generalized_load=generalized_load,
        )
        metrics_new = rregions.metrics(residual_new, regions, scale)
        last_jout = jout_new
        row = {
            "iteration": int(iteration),
            "increment_inf": last_increment,
            "residual_scale": scale,
        }
        row.update(rregions.as_dict(metrics_new))
        history.append(row)
        ww = np.asarray(U[2], float)
        js, ix = np.unravel_index(int(np.argmax(np.abs(ww))), ww.shape)
        fprog.emit(
            "newton_iteration", iteration=int(iteration),
            residual_inf=float(metrics_new.residual_inf),
            residual_relative=float(metrics_new.residual_relative),
            residual_convergence_inf=float(metrics_new.residual_convergence_inf),
            residual_convergence_relative=float(metrics_new.residual_convergence_relative),
            residual_inf_interior=float(metrics_new.residual_inf_interior),
            residual_inf_boundary=float(metrics_new.residual_inf_boundary),
            residual_inf_load_front=float(metrics_new.residual_inf_load_front),
            increment_inf=last_increment, stage="NEWTON_VK_J2",
            w_abs_max=float(np.max(np.abs(ww))), w_at_abs_max=float(ww[js, ix]),
            wmax_is=int(js), wmax_ix=int(ix),
        )
        if (
            rregions.converged(metrics_new, limit)
            and last_increment <= settings.increment_tolerance
        ):
            return _result(
                U, True, "CONVERGED_STRONG_FDM_VK_J2", iteration,
                metrics_new, scale, last_increment, history, jout_new
            )

    residual_fin, _strong, jout_fin, _kin = strong_collocation_residual_vk_j2(
        U, ops, bc, committed=committed, parameters=parameters,
        quadrature=quadrature, generalized_load=generalized_load,
    )
    metrics_fin = rregions.metrics(residual_fin, regions, scale)
    return _result(
        U, False, "MAX_STRONG_FDM_VK_J2_NEWTON_ITERATIONS",
        int(settings.max_iterations), metrics_fin, scale, last_increment,
        history, jout_fin
    )


__all__ = [
    "KLStrongFDMVKJ2Error",
    "KLStrongFDMVKJ2SystemSingular",
    "KLStrongFDMVKJ2Result",
    "strong_collocation_residual_vk_j2",
    "finite_difference_jacobian",
    "solve_vk_j2_strong_fdm",
]