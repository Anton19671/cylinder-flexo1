#!/usr/bin/env python3
"""Projected matrix-free free-vibration eigenproblem on large CSR shell grids.

This is an opt-in FDM route for *uncoupled mechanical* seven-theory shells.
The physical operators are the unchanged variational tangent at the zero
unloaded state and the unchanged consistent weak mass action.  Dirichlet
traces are projected using the existing sparse trace projector, without a
dense null-space basis, fictitious mass or boundary penalties.

This solver is not a time integrator and never substitutes for the separated
Krysko-Kirichenko MVI modal method. The coupled massless-electric Schur
problem is left on the established dense route until separately implemented.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

from cylinder_matrixfree_fdm import SparseTraceProjector
from cylinder_variational_fdm import state_scales


def _projected_actions(prepared, density, projector, *, progress_callback=None):
    """Construct constrained K and M actions in consistently scaled coordinates.

    K is the centered directional derivative of the *same* full variational
    first variation as the small-grid reference. M is the physical mass_action.
    For v in ker(C), P K P and P M P define the exact discrete weak eigenproblem
    restricted to essential homogeneous traces; no artificial complement
    stiffness or mass is introduced.
    """
    functional = prepared.problem.functional
    q0, p0 = prepared.mechanical, prepared.electrical
    if p0 is not None:
        raise ValueError("MATRIX_FREE_MODAL_ELECTRIC_SCHUR_NOT_IMPLEMENTED: "
                         "use the dense modal reference for electric coupling")
    if density <= 0:
        raise ValueError("MODAL density must be positive")
    if np.max(np.abs(projector.target), initial=0.) > 1.e-12 or (
            np.max(np.abs(projector.lift), initial=0.) > 1.e-10):
        raise ValueError("MATRIX_FREE_MODAL_REQUIRES_HOMOGENEOUS_TRACES")
    n = q0.size
    scales = state_scales(functional, q0, None, prepared.settings)
    if n != projector.size:
        raise ValueError("MATRIX_FREE_MODAL_TRACE_DIMENSION_MISMATCH")
    area = float(np.sum(functional.W))
    stiffness_scale = float(functional.material.young_pa * functional.h * area)
    mass_scale = float(density * functional.h * area)
    if not (np.isfinite(stiffness_scale) and np.isfinite(mass_scale)
            and stiffness_scale > 0. and mass_scale > 0.):
        raise ValueError("MATRIX_FREE_MODAL_INVALID_OPERATOR_SCALES")
    step = prepared.settings.tangent_step_scaled
    if step <= 0.:
        raise ValueError("positive modal tangent step required")
    calls = {"K": 0, "M": 0}

    # In the LINEAR kinematics the tangent is exactly constant. Taking the
    # Gateaux derivative at zero requires one physical weak action, not two
    # perturbed nonlinear evaluations. Retain symmetric differences for the
    # nonlinear geometric branches and do not modify either formulation.
    is_linear = functional.geometry_model == "LINEAR"
    # Reuse the same local weak tangent used in the independent MVI route.
    # This is restricted to homogeneous LINEAR macro elasticity.
    local_tangent = None
    if is_linear and not functional.material.ell_m:
        from cylinder_modal_mvi import _LocalMechanicalTangent
        local_tangent = _LocalMechanicalTangent(
            prepared.problem, functional, step)
    elif is_linear:
        _, at_zero, _ = prepared.problem.evaluate_effective_residual(
            np.zeros_like(q0), None)
    def k_action(vector):
        direction = projector.project_null(np.asarray(vector, float))
        physical = (scales * direction).reshape(q0.shape)
        if local_tangent is not None:
            tangent_force = local_tangent.action(physical)
        else:
            _, right, _ = prepared.problem.evaluate_effective_residual(
                step * physical, None)
            if is_linear:
                tangent_force = (right - at_zero) / step
            else:
                _, left, _ = prepared.problem.evaluate_effective_residual(
                    -step * physical, None)
                tangent_force = (right - left) / (2. * step)
        action = projector.project_null(
            scales * tangent_force / stiffness_scale)
        calls["K"] += 1
        if progress_callback is not None:
            progress_callback("K", calls["K"])
        return action

    def m_action(vector):
        direction = projector.project_null(np.asarray(vector, float))
        physical = (scales * direction).reshape(q0.shape)
        action = functional.mass_action(
            physical, density_kg_m3=density,
            mechanical=q0, velocity=np.zeros_like(q0))
        calls["M"] += 1
        if progress_callback is not None:
            progress_callback("M", calls["M"])
        return projector.project_null(scales * action.ravel() / mass_scale)

    def as_operator(action):
        return LinearOperator(
            (n, n), dtype=float, matvec=action,
            matmat=lambda values: np.column_stack(
                [action(values[:, j]) for j in range(values.shape[1])]))

    return as_operator(k_action), as_operator(m_action), scales, stiffness_scale / mass_scale, mass_scale


def solve_modal_matrixfree(prepared, density, requested, *, modal_options,
                           progress_callback=None, detail_callback=None):
    """First physical modes via constrained generalized LOBPCG, fail-closed.

    The Krylov vectors, operator actions and preconditioned directions all
    belong to ker(C). The globally semidefinite projected operators are
    positive/regular on that subspace, which is the only admissible space.
    """
    if prepared.method != "FDM" or prepared.fdm_backend != "MATRIX_FREE_GMRES":
        raise ValueError("MATRIX_FREE_MODAL_REQUIRES_FDM_CSR")
    q0, p0 = prepared.mechanical, prepared.electrical
    if p0 is not None:
        raise ValueError("MATRIX_FREE_MODAL_ELECTRIC_SCHUR_NOT_IMPLEMENTED")
    options = modal_options or {}
    tolerance = float(options.get("eigen_tolerance", 1.e-6))
    iterations = options.get("max_iterations", 120)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("MODAL eigen_tolerance must be in (0,1)")
    if type(iterations) is not int or not 1 <= iterations <= 2000:
        raise ValueError("MODAL max_iterations must be an integer 1..2000")
    functional = prepared.problem.functional
    scales = state_scales(functional, q0, None, prepared.settings)
    projector = SparseTraceProjector.from_problem(
        prepared.problem, q0, None, scales, time_s=None,
        max_rows=prepared.settings.max_constraint_rows)
    n_free = q0.size - projector.count
    if requested > n_free or n_free < 5 * requested:
        raise ValueError("MATRIX_FREE_MODAL_INSUFFICIENT_FREE_DIRECTIONS")
    if detail_callback is not None:
        detail_callback(stage="MATRIX_FREE_MODAL_BEGIN", n_modes=requested,
                        mechanical_free_dofs=n_free)
    from cylinder_modal_direct_small import (
        can_use_modal_direct_small, solve_modal_direct_small)
    if can_use_modal_direct_small(projector):
        return solve_modal_direct_small(
            prepared, density, requested, projector,
            progress_callback=progress_callback, detail_callback=detail_callback)
    def progress(family, calls):
        if detail_callback is not None and (calls == 1 or calls % 25 == 0):
            detail_callback(stage="MATRIX_FREE_MODAL_ACTION",
                            operator=family, actions=calls)
    k, m, scales, conversion, mass_scale = _projected_actions(
        prepared, density, projector, progress_callback=progress)
    # Include additional Ritz vectors around the requested spectral cutoff.
    # With rotational doublets, a block ending exactly at the requested
    # eigenvalue can converge poorly. Oversampling keeps the physical
    # operator and boundary projection unchanged.
    block_size = min(max(requested + 4, 2 * requested), n_free // 5)
    rng = np.random.default_rng(20261030)
    seeds = np.column_stack([
        projector.project_null(rng.normal(size=q0.size))
        for _ in range(block_size)])
    seeds, _ = np.linalg.qr(seeds, mode="reduced")
    if np.min(np.linalg.norm(seeds, axis=0)) < .99:
        raise ValueError("MATRIX_FREE_MODAL_NO_INDEPENDENT_START")
    seeds = np.column_stack([
        projector.project_null(seeds[:, j]) for j in range(block_size)])
    # Sparse SPD block inverse acts only as an LOBPCG preconditioner.
    # It does not alter physical stiffness, inertia, or boundary conditions.
    from cylinder_modal_preconditioner import ModalSparseBlockPreconditioner
    area = float(np.sum(functional.W))
    stiffness_scale = float(functional.material.young_pa * functional.h * area)
    preconditioner = ModalSparseBlockPreconditioner(
        functional, scales, stiffness_scale, projector,
        max_surface_nodes=prepared.settings.max_surface_nodes,
        detail_callback=detail_callback)
    t = preconditioner.linear_operator()
    if detail_callback is not None:
        detail_callback(stage="EIGEN_BEGIN", n_modes=requested,
                        eigen_solver="LOBPCG_PROJECTED")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        values, vectors = lobpcg(
            k, seeds, B=m, M=t, tol=tolerance, maxiter=iterations,
            largest=False)
    if caught and detail_callback is not None:
        detail_callback(stage="MATRIX_FREE_MODAL_ITERATION_WARNINGS",
                        messages=[str(w.message) for w in caught])
    if not (np.isfinite(values).all() and np.isfinite(vectors).all()):
        raise ValueError("MATRIX_FREE_MODAL_NONFINITE_EIGENPAIR")
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    kv, mv = k @ vectors, m @ vectors
    relative = np.linalg.norm(kv - mv * values[None, :], axis=0) / np.maximum(
        np.linalg.norm(kv, axis=0) +
        np.abs(values) * np.linalg.norm(mv, axis=0), 1.e-14)
    if np.any(relative[:requested] > max(5.e-5, 20. * tolerance)):
        # An initial LOBPCG block can stop before the low modes are ready.
        # Continue from the computed admissible vectors, keeping the exact
        # physical K/M actions, projected boundary conditions and validation.
        # Do not accept an unverified eigenpair or change the shell model.
        continuation = min(max(2 * iterations, 200), 2000)
        if continuation > iterations:
            if detail_callback is not None:
                detail_callback(stage="MATRIX_FREE_MODAL_CONTINUATION",
                                initial_max_iterations=iterations,
                                additional_max_iterations=continuation)
            restart = np.column_stack([
                projector.project_null(vectors[:, j])
                for j in range(vectors.shape[1])])
            with warnings.catch_warnings(record=True) as retry_warnings:
                warnings.simplefilter("always", UserWarning)
                values, vectors = lobpcg(
                    k, restart, B=m, M=t, tol=tolerance,
                    maxiter=continuation, largest=False)
            if retry_warnings and detail_callback is not None:
                detail_callback(stage="MATRIX_FREE_MODAL_CONTINUATION_WARNINGS",
                                messages=[str(w.message) for w in retry_warnings])
            if not (np.isfinite(values).all() and np.isfinite(vectors).all()):
                raise ValueError("MATRIX_FREE_MODAL_NONFINITE_EIGENPAIR")
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
            kv, mv = k @ vectors, m @ vectors
            relative = np.linalg.norm(kv - mv * values[None, :], axis=0) / np.maximum(
                np.linalg.norm(kv, axis=0) +
                np.abs(values) * np.linalg.norm(mv, axis=0), 1.e-14)
        if np.any(relative[:requested] > max(5.e-5, 20. * tolerance)):
            raise ValueError(
                "MATRIX_FREE_MODAL_EIGENPAIRS_NOT_VERIFIED: "
                "increase modal.max_iterations or refine the preconditioner")
    # Keep mass actions consistent with the final eigenvectors, including
    # a continued LOBPCG solve. The pre-continuation M*v must not be reused.
    values, vectors = values[:requested], vectors[:, :requested]
    mv = m @ vectors
    if np.any(values < -1.e-8 * max(1., np.max(np.abs(values)))):
        raise ValueError("negative squared frequency: unstable modal linearization")
    if np.max(np.abs(projector.c @ vectors), initial=0.) > 1.e-7:
        raise ValueError("MATRIX_FREE_MODAL_BOUNDARY_CONSTRAINT_DRIFT")
    gram = vectors.T @ mv
    diagonal = np.sqrt(np.diag(gram))
    if np.any(diagonal <= 0.) or not np.isfinite(diagonal).all():
        raise ValueError("MATRIX_FREE_MODAL_MASS_IS_NOT_POSITIVE")
    vectors = vectors / diagonal[None, :]
    gram = gram / diagonal[:, None] / diagonal[None, :]
    if np.max(np.abs(gram - np.eye(requested))) > 1.e-4:
        raise ValueError("MATRIX_FREE_MODAL_MODES_NOT_MASS_ORTHOGONAL")
    physical_values = np.maximum(values, 0.) * conversion
    omega = np.sqrt(physical_values)
    frequency = omega / (2. * np.pi)
    fields = np.stack([
        (scales * projector.project_null(vectors[:, i]) /
         np.sqrt(mass_scale)).reshape(q0.shape)
        for i in range(requested)])
    for i in range(requested):
        if detail_callback is not None:
            detail_callback(stage="FDM_MODE_DONE", mode=i + 1,
                            n_modes=requested, frequency_hz=float(frequency[i]),
                            omega_rad_s=float(omega[i]),
                            modal_amplitude_max=float(np.max(abs(fields[i]))))
        if progress_callback is not None:
            progress_callback(i + 1, requested)
    return {
        "theory": functional.theory, "method": "FDM",
        "fdm_backend": "MATRIX_FREE_GMRES",
        "modal_eigensolver": "LOBPCG_PROJECTED",
        "density_kg_m3": float(density),
        "frequencies_hz": frequency,
        "omega_rad_s": omega,
        "omega_squared_rad2_s2": physical_values,
        "mechanical_modes": fields, "electrical_modes": None,
        "n_mechanical_free_dofs": n_free, "n_electric_free_dofs": 0,
    }


__all__ = ("_projected_actions", "solve_modal_matrixfree")
