"""Consistent constrained initial acceleration for the reference Newmark route.

ZERO initial displacement and velocity do not imply zero initial acceleration.
At t0 solve the exact mechanical mass equation on admissible acceleration
variations. Electrical coefficients remain quasistatic; they have no mass.
No fixed MVI rank, material law, or spatial solver is modified here.
"""
from __future__ import annotations

import numpy as np
from cylinder_boundary_spatial import affine_trace_space
from cylinder_dynamic_trace import mechanical_trace_matrix
from cylinder_variational_fdm import state_scales, solve_checked_reduced_tangent


def consistent_initial_acceleration(problem, mechanical, electrical, *,
                                    initial_residual, initial_edge, settings,
                                    density_kg_m3, backend="DENSE_REFERENCE",
                                    full_assembly=None, sparse_projector=None):
    """Solve M(q0) a0 = -R_static(q0, phi0), C_mechanical a0 = 0.

    initial_residual must contain all effective applied forces and boundary
    work at t0 (including any velocity-dependent terms evaluated at v0).
    Current reference CASEs require zero initial velocity. The provided mass
    action is linear in the unknown acceleration at that fixed state.
    """
    if getattr(problem, "kind", None) != "DYNAMIC":
        raise ValueError("INITIAL_ACCELERATION_REQUIRES_DYNAMIC_PROBLEM")
    f = problem.functional
    q = np.asarray(mechanical, dtype=float)
    p = None if electrical is None else np.asarray(electrical, dtype=float)
    n = q.size
    r = np.asarray(initial_residual, dtype=float)
    if (q.ndim != 3 or q.shape != (f.n_mechanical, *f.ops.shape)
            or r.shape != (n + (0 if p is None else p.size),)
            or not np.isfinite(q).all() or not np.isfinite(r).all()
            or (p is not None and not np.isfinite(p).all())):
        raise ValueError("INITIAL_ACCELERATION_INVALID_STATE_OR_FORCE")
    density = float(density_kg_m3)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("INITIAL_ACCELERATION_INVALID_DENSITY")
    scales = state_scales(f, q, p, settings)
    scale_m = scales[:n]
    zeros = np.zeros_like(q)
    c, _ = mechanical_trace_matrix(getattr(initial_edge, "constraints", ()), n)

    # The mechanical mass cannot balance an electric Gauss-law equation.
    if p is not None:
        electric_only = np.zeros_like(r)
        electric_only[n:] = r[n:]
        if backend == "DENSE_REFERENCE":
            if full_assembly is None:
                raise ValueError("INITIAL_ACCELERATION_DENSE_ASSEMBLY_REQUIRED")
            electric_test = full_assembly.space.residual(electric_only)
            electric_tol = 1.e-10
        elif backend == "MATRIX_FREE_GMRES":
            if sparse_projector is None:
                raise ValueError("INITIAL_ACCELERATION_SPARSE_PROJECTOR_REQUIRED")
            electric_test = sparse_projector.project_null(scales * electric_only)
            electric_tol = 1.e-10 * max(1., float(np.max(scales)))
        else:
            raise ValueError("INITIAL_ACCELERATION_UNKNOWN_BACKEND")
        if np.max(np.abs(electric_test), initial=0) > electric_tol:
            raise ValueError(
                "INITIAL_ELECTRIC_STATE_NOT_EQUILIBRATED_AT_T0: "
                "mechanical acceleration cannot balance electric charge; "
                "supply an initially equilibrated potential")

    def physical_mass_action(acc):
        # The reference CASE prescribes v0=0; KL convective inertia vanishes.
        result = f.mass_action(acc.reshape(q.shape), density_kg_m3=density,
                               mechanical=q, velocity=zeros)
        return np.asarray(result, dtype=float).reshape(-1)

    if backend == "DENSE_REFERENCE":
        space = affine_trace_space(c * scale_m[None, :],
                                   np.zeros(len(c)), np.zeros(n))
        b = space.basis
        if not b.shape[1]:
            return zeros.copy()
        force = b.T @ (scale_m * r[:n])
        if not np.any(force):
            return zeros.copy()
        mass = np.column_stack([
            b.T @ (scale_m * physical_mass_action(scale_m * direction))
            for direction in b.T
        ])
        try:
            reduced, _ = solve_checked_reduced_tangent(mass, force)
        except np.linalg.LinAlgError as exc:
            raise ValueError("INITIAL_MECHANICAL_MASS_MATRIX_SINGULAR") from exc
        acceleration = (scale_m * (b @ reduced)).reshape(q.shape)
        defect = b.T @ (scale_m *
                        (r[:n] + physical_mass_action(acceleration.ravel())))
        allowed = 1.e-9 * max(float(np.max(np.abs(force))),
                              np.finfo(float).tiny)
        if np.max(np.abs(defect), initial=0) > allowed:
            raise ValueError("INITIAL_MECHANICAL_MASS_BALANCE_FAILED")
    elif backend == "MATRIX_FREE_GMRES":
        if sparse_projector is None:
            raise ValueError("INITIAL_ACCELERATION_SPARSE_PROJECTOR_REQUIRED")
        from scipy.sparse.linalg import LinearOperator, cg

        def project(v):
            full = np.zeros_like(r)
            full[:n] = v
            return sparse_projector.project_null(full)[:n]

        rhs = -project(scale_m * r[:n])
        if not np.any(rhs):
            return zeros.copy()

        def mass_matvec(v):
            admissible = project(v)
            return project(scale_m * physical_mass_action(scale_m * admissible))

        mass_norm = float(np.max(np.abs(mass_matvec(np.ones(n))), initial=0))
        shift = max(mass_norm, np.finfo(float).tiny)

        def matvec(v):
            projected = project(v)
            return mass_matvec(projected) + shift * (v - projected)

        a, info = cg(LinearOperator((n, n), matvec=matvec, dtype=float),
                     rhs, rtol=1.e-10, atol=0., maxiter=min(4*n, 1200))
        if info != 0 or not np.isfinite(a).all():
            raise ValueError("INITIAL_MECHANICAL_MASS_CG_NOT_CONVERGED")
        acceleration = (scale_m * project(a)).reshape(q.shape)
        force = project(scale_m * (r[:n] +
                                  physical_mass_action(acceleration.ravel())))
        allowed = 1.e-8 * max(float(np.max(np.abs(rhs))),
                              np.finfo(float).tiny)
        if np.max(np.abs(force), initial=0) > allowed:
            raise ValueError("INITIAL_MECHANICAL_MASS_BALANCE_FAILED")
    else:
        raise ValueError("INITIAL_ACCELERATION_UNKNOWN_BACKEND")

    # Newmark must begin with acceleration consistent with its support traces.
    trace_defect = c @ acceleration.ravel()
    if (not np.isfinite(acceleration).all()
            or np.max(np.abs(trace_defect), initial=0) >
            1.e-10 * max(1., float(np.max(np.abs(acceleration))))):
        raise ValueError("INITIAL_MECHANICAL_ACCELERATION_TRACE_FAILED")
    return acceleration


__all__ = ("consistent_initial_acceleration",)
