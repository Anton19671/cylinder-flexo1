"""Independent full-grid MKR/FDM modal eigensolver branch.

The legacy GUI label MKR denotes the full variational FDM discretization,
not an alternative Ritz basis.  This module never calls a fixed-rank MVI
solver or consumes MVI mode fields.  The physical functional is deliberately
the same for both methods; only the numerical approximation differs.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve


BRANCH_ID = "MKR_FULL_SPACE"


def _dense_full_space(prepared, density, requested, *,
                      progress_callback=None, detail_callback=None):
    """Solve the assembled full generalized eigenproblem, without MVI."""
    # This is the existing full-space assembly, now invoked only by MKR.
    # Imported here to avoid circular imports while retaining unchanged K/M.
    from cylinder_modal_reference import _linear_matrices

    k, m, bm, be, q_shape, p_shape = _linear_matrices(
        prepared, density, progress_callback=progress_callback,
        detail_callback=detail_callback)
    n_m = bm.shape[1]
    if requested > n_m:
        raise ValueError("requested modes exceed free mechanical DOFs")
    k_mech = k[:n_m, :n_m]
    electric_recovery = None
    if be.shape[1]:
        k_qp = k[:n_m, n_m:]
        k_pq = k[n_m:, :n_m]
        k_pp = k[n_m:, n_m:]
        # Electrostatic coefficients have zero inertia: condense Kpp.
        electric_recovery = -solve(k_pp, k_pq, assume_a="sym", check_finite=True)
        k_mech = k_mech + k_qp @ electric_recovery
        k_mech = (k_mech + k_mech.T) * 0.5
    if detail_callback:
        detail_callback(stage="EIGEN_BEGIN", n_modes=requested)
    eigenvalues, eigvec = eigh(
        k_mech, m, subset_by_index=(0, requested - 1),
        check_finite=True, driver="gvx")
    if not np.isfinite(eigenvalues).all():
        raise ValueError("nonfinite MODAL eigenvalues")
    negative_limit = -1.e-8 * max(float(np.max(np.abs(eigenvalues))), 1.)
    if np.any(eigenvalues < negative_limit):
        raise ValueError(
            "negative squared frequency: unstable or incorrect modal linearization")
    omega = np.sqrt(np.maximum(eigenvalues, 0.))
    frequencies_hz = omega / (2. * np.pi)
    q_modes = np.stack([(bm @ eigvec[:, j]).reshape(q_shape)
                        for j in range(requested)])
    if detail_callback:
        for j in range(requested):
            detail_callback(
                stage="FDM_MODE_DONE", mode=j + 1, n_modes=requested,
                frequency_hz=float(frequencies_hz[j]),
                omega_rad_s=float(omega[j]),
                modal_amplitude_max=float(np.max(abs(q_modes[j]), initial=0.)))
    p_modes = (None if electric_recovery is None else np.stack([
        (be @ (electric_recovery @ eigvec[:, j])).reshape(p_shape)
        for j in range(requested)]))
    return {
        "theory": prepared.problem.functional.theory,
        "method": "FDM",
        "density_kg_m3": density,
        "frequencies_hz": frequencies_hz,
        "omega_rad_s": omega,
        "omega_squared_rad2_s2": eigenvalues,
        "mechanical_modes": q_modes,
        "electrical_modes": p_modes,
        "n_mechanical_free_dofs": n_m,
        "n_electric_free_dofs": be.shape[1],
    }


def solve_modal_mkr(prepared, density, requested, *, modal_options=None,
                    progress_callback=None, detail_callback=None):
    """Exclusive full-space branch.  Never borrow an MVI result."""
    if getattr(prepared, "method", None) != "FDM":
        raise ValueError("MODAL_MKR_REQUIRES_FDM_PREPARED_METHOD")
    backend = getattr(prepared, "fdm_backend", None)
    if backend == "MATRIX_FREE_GMRES":
        from cylinder_modal_matrixfree import solve_modal_matrixfree
        result = solve_modal_matrixfree(
            prepared, density, requested, modal_options=modal_options,
            progress_callback=progress_callback, detail_callback=detail_callback)
    elif backend == "DENSE_REFERENCE":
        result = _dense_full_space(
            prepared, density, requested, progress_callback=progress_callback,
            detail_callback=detail_callback)
    else:
        raise ValueError("MODAL_MKR_UNKNOWN_FDM_BACKEND: " + str(backend))
    if (not isinstance(result, dict) or result.get("method") != "FDM"
            or "rank" in result or "mvi_sweeps" in result
            or "mvi_modal_algorithm" in result):
        raise ValueError("MODAL_MKR_RESULT_METHOD_CROSSCONTAMINATION")
    result["modal_solver_branch"] = BRANCH_ID
    return result


__all__ = ("BRANCH_ID", "solve_modal_mkr")
