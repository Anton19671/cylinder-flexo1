#!/usr/bin/env python3
"""Small-grid linear free-vibration reference for all seven shell theories.

This is a distinct MODAL analysis, NOT a transient Newmark calculation.
K is the derivative of the existing variational static force at the
undeformed, unloaded state. M is its existing consistent inertial action.
Both are pulled back through the actual model-dependent essential traces.

The electric field is quasi-static, so its massless coefficients are
eliminated by a Schur complement rather than assigned fictitious inertia.
No shell kinematics, stiffness laws or dynamic integrators are replaced.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve, cholesky

from cylinder_boundary_spatial import affine_trace_space
from cylinder_variational_fdm import state_scales
from cylinder_reference_case import BACKEND, prepare_reference_case, _integer, _real


THEORIES = ("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")


def _modal_spec(case, *, base_dir=None):
    """Strict, read-only preflight using the same physics and BC as STATIC."""
    if not isinstance(case, dict):
        raise ValueError("MODAL CASE must be an object")
    if str(case.get("solver_backend", "")).upper() != BACKEND:
        raise ValueError("MODAL requires VARIATIONAL_REFERENCE")
    if str(case.get("calculation_kind", "")).upper() != "MODAL":
        raise ValueError("MODAL requires calculation_kind=MODAL")
    if str(case.get("mode", "")).upper() != "LINEAR_ELASTIC":
        raise ValueError("MODAL linearization requires mode=LINEAR_ELASTIC")
    if str(case.get("method", "")).upper() != "FDM":
        raise ValueError("MODAL uses the full variational FDM space, not MVI ranks")
    if str(case.get("fdm_backend", "DENSE_REFERENCE")).upper() != "DENSE_REFERENCE":
        raise ValueError("MODAL is a small-grid dense reference, not matrix-free")
    if case.get("dynamics") is not None or case.get("rank") is not None:
        raise ValueError("MODAL has neither time integration nor MVI rank")
    if case.get("reference_mvi") is not None:
        raise ValueError("MODAL must not contain reference_mvi")
    if case.get("load_definition") is not None or case.get("reference_loads") is not None:
        raise ValueError("MODAL free vibrations require an unloaded CASE")
    raw = case.get("modal", {})
    if not isinstance(raw, dict) or set(raw) - {"density_kg_m3", "n_modes"}:
        raise ValueError("modal accepts only density_kg_m3 and n_modes")
    density = _real(raw.get("density_kg_m3"), "modal.density_kg_m3", True)
    n_modes = _integer(raw.get("n_modes"), "modal.n_modes", 1)
    if n_modes > 32:
        raise ValueError("small-grid reference supports at most 32 modes")
    scale = str(case.get("scale_model", "")).upper()
    if scale == "MICRO":
        micro = case.get("micro", {})
        if "rho_kg_m3" in micro and not np.isclose(
                density, _real(micro["rho_kg_m3"], "micro.rho_kg_m3", True),
                rtol=1.e-12, atol=0.):
            raise ValueError("modal density must match micro.rho_kg_m3")
        if float(micro.get("eta_s", 0.)) != 0.:
            raise ValueError("MODAL is undamped: micro.eta_s must be zero")
    theory = str(case.get("theory", "")).upper()
    if theory == "CUF":
        theory = "CUF_" + str(case.get("cuf_model", "")).upper()
    if theory not in THEORIES:
        raise ValueError("MODAL requires KL, FSDT, REDDY or CUF TE1..TE4")
    n_mechanical = (3 if theory == "KL" else 5 if theory in ("FSDT", "REDDY")
                    else 3 * (int(theory[-1]) + 1))
    surrogate = copy.deepcopy(case)
    surrogate["calculation_kind"] = "STATIC"
    surrogate["mode"] = "NONLINEAR_ELASTIC"
    surrogate["reference_loads"] = {
        "mechanical": [0.0] * n_mechanical, "load_factors": [0.0]}
    surrogate.pop("modal", None)
    prepared = prepare_reference_case(surrogate, base_dir=base_dir)
    return prepared, density, n_modes


def prepare_modal_case(case, *, base_dir=None):
    """Validate the complete MODAL CASE without producing output files."""
    prepared, density, requested = _modal_spec(case, base_dir=base_dir)
    f = prepared.problem.functional
    q, p = prepared.mechanical, prepared.electrical
    zero = np.zeros(q.size + (0 if p is None else p.size))
    first = prepared.problem.assemble(q, p)
    if np.max(np.abs(first.space.origin), initial=0.) > 1.e-10:
        raise ValueError("MODAL supports only homogeneous essential BC")
    if first.space.basis.shape[1] > prepared.settings.max_free_dofs:
        raise ValueError("MODAL exceeds reference_newton.max_free_dofs")
    scales = state_scales(f, q, p, prepared.settings)
    mechanical_basis, electric_basis = _modal_bases(
        first.space.matrix, first.space.target, scales, q.size)
    if mechanical_basis.shape[1] < requested:
        raise ValueError("modal.n_modes exceeds unconstrained mechanical DOFs")
    return {"theory": f.theory, "method": "FDM",
            "calculation_kind": "MODAL", "density_kg_m3": density,
            "n_modes": requested, "mechanical_free_dofs": int(mechanical_basis.shape[1]),
            "electric_free_dofs": int(electric_basis.shape[1]),
            "reference_only": True}


def _modal_bases(c, target, scales, mechanical_size):
    """Separate mechanical and massless electric constraint null spaces."""
    nm = mechanical_size
    nq = len(scales)
    c = np.asarray(c, float)
    target = np.asarray(target, float)
    cm, ce = c[:, :nm], c[:, nm:]
    is_m = np.any(cm != 0., axis=1)
    is_e = np.any(ce != 0., axis=1)
    if np.any(is_m & is_e):
        raise ValueError("MODAL cannot eliminate mixed mechanical/electric essential traces")
    if np.any(~(is_m | is_e)) or np.max(np.abs(target), initial=0.) > 1.e-12:
        raise ValueError("MODAL needs homogeneous, nonzero constraint rows")
    bm = affine_trace_space(
        cm[is_m] * scales[None, :nm], target[is_m],
        np.zeros(nm)).basis
    if nq == nm:
        be = np.empty((0, 0), float)
    else:
        be = affine_trace_space(
            ce[is_e] * scales[None, nm:], target[is_e],
            np.zeros(nq - nm)).basis
    return bm * scales[:nm, None], be * scales[nm:, None]


def _linear_matrices(prepared, density, *, progress_callback=None):
    problem = prepared.problem
    f = problem.functional
    q0, p0 = prepared.mechanical, prepared.electrical
    nm = q0.size
    n = nm + (0 if p0 is None else p0.size)
    scales = state_scales(f, q0, p0, prepared.settings)
    first = problem.assemble(q0, p0)
    bm, be = _modal_bases(first.space.matrix, first.space.target, scales, nm)
    n_m, n_e = bm.shape[1], be.shape[1]
    if not n_m:
        raise ValueError("all mechanical DOFs are fixed")
    directions = np.zeros((n, n_m + n_e), float)
    directions[:nm, :n_m] = bm
    if n_e:
        directions[nm:, n_m:] = be
    if directions.shape[1] > prepared.settings.max_free_dofs:
        raise ValueError("MODAL dense free-DOF safety limit exceeded")
    step = prepared.settings.tangent_step_scaled
    if step <= 0:
        raise ValueError("positive linearization step required")

    def residual(state):
        mechanical = state[:nm].reshape(q0.shape)
        electrical = None if p0 is None else state[nm:].reshape(p0.shape)
        _, full, _ = problem.evaluate_effective_residual(mechanical, electrical)
        return full

    # Central difference of the exact *discrete first variation*. All seven
    # theory-specific kinematic kernels remain exclusively in CylinderFunctional.
    k = np.empty((n_m + n_e, n_m + n_e), float)
    for j in range(k.shape[1]):
        direction = directions[:, j]
        derivative = (residual(step * direction)
                      - residual(-step * direction)) / (2. * step)
        k[:, j] = directions.T @ derivative
        if progress_callback is not None:
            progress_callback(j + 1, k.shape[1])
    if not np.isfinite(k).all():
        raise ValueError("nonfinite MODAL tangent")
    asymmetry = float(np.linalg.norm(k - k.T, ord=np.inf))
    magnitude = max(float(np.linalg.norm(k, ord=np.inf)), 1.e-30)
    if asymmetry > 1.e-5 * magnitude:
        raise ValueError("MODAL tangent is not variationally symmetric")
    k = (k + k.T) * 0.5

    # Mass from the existing KL exact inertial action, director inertia
    # for FSDT/Reddy, or through-thickness TE mass moments for CUF.
    m = np.empty((n_m, n_m), float)
    for j in range(n_m):
        acceleration = bm[:, j].reshape(q0.shape)
        mass_column = f.mass_action(
            acceleration, density_kg_m3=density,
            mechanical=q0, velocity=np.zeros_like(q0))
        m[:, j] = bm.T @ mass_column.ravel()
    if not np.isfinite(m).all():
        raise ValueError("nonfinite MODAL mass")
    mass_asymmetry = float(np.linalg.norm(m - m.T, ord=np.inf))
    if mass_asymmetry > 1.e-9 * max(float(np.linalg.norm(m, ord=np.inf)), 1.e-30):
        raise ValueError("MODAL mass is not symmetric")
    m = (m + m.T) * 0.5
    cholesky(m, lower=True, check_finite=True)
    return k, m, bm, be, q0.shape, None if p0 is None else p0.shape


def solve_modal_case(case, *, base_dir=None, progress_callback=None):
    """Compute consistent small-grid modes in Hz, with massless electric DOFs condensed."""
    prepared, density, requested = _modal_spec(case, base_dir=base_dir)
    k, m, bm, be, q_shape, p_shape = _linear_matrices(
        prepared, density, progress_callback=progress_callback)
    n_m = bm.shape[1]
    if requested > n_m:
        raise ValueError("requested modes exceed free mechanical DOFs")
    k_mech = k[:n_m, :n_m]
    electric_recovery = None
    if be.shape[1]:
        k_qp = k[:n_m, n_m:]
        k_pq = k[n_m:, :n_m]
        k_pp = k[n_m:, n_m:]
        # Electric potential has NO inertia. Its stationary equation is
        # K_pq q + K_pp phi = 0. Never add a fictitious electric mass.
        electric_recovery = -solve(k_pp, k_pq, assume_a="sym", check_finite=True)
        k_mech = k_mech + k_qp @ electric_recovery
        k_mech = (k_mech + k_mech.T) * 0.5
    eigenvalues, eigvec = eigh(
        k_mech, m, subset_by_index=(0, requested - 1),
        check_finite=True, driver="gvx")
    if not np.isfinite(eigenvalues).all():
        raise ValueError("nonfinite MODAL eigenvalues")
    negative_limit = -1.e-8 * max(float(np.max(np.abs(eigenvalues))), 1.)
    if np.any(eigenvalues < negative_limit):
        raise ValueError("negative squared frequency: unstable or incorrect modal linearization")
    omega = np.sqrt(np.maximum(eigenvalues, 0.))
    frequencies_hz = omega / (2. * np.pi)
    q_modes = np.stack([(bm @ eigvec[:, j]).reshape(q_shape)
                        for j in range(requested)])
    p_modes = (None if electric_recovery is None else np.stack([
        (be @ (electric_recovery @ eigvec[:, j])).reshape(p_shape)
        for j in range(requested)]))
    return {"theory": prepared.problem.functional.theory,
            "density_kg_m3": density, "frequencies_hz": frequencies_hz,
            "omega_rad_s": omega, "omega_squared_rad2_s2": eigenvalues,
            "mechanical_modes": q_modes, "electrical_modes": p_modes,
            "n_mechanical_free_dofs": n_m,
            "n_electric_free_dofs": be.shape[1]}


def run_modal_case(case, case_path):
    """Write a standalone, reproducible MODAL output; never run transient dynamics."""
    path = Path(case_path).expanduser().resolve()
    prepare_modal_case(case, base_dir=path.parent)
    out = Path(case["output_dir"]).expanduser()
    if not out.is_absolute():
        out = path.parent / out
    out = out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("OUTPUT_DIR_NOT_EMPTY: MODAL results must be separate")

    def report(current, total):
        if current == 1 or current == total or current % max(1, total // 10) == 0:
            print(f"MODAL tangent column {current}/{total}", flush=True)

    result = solve_modal_case(case, base_dir=path.parent, progress_callback=report)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frequencies_hz": result["frequencies_hz"],
        "omega_rad_s": result["omega_rad_s"],
        "omega_squared_rad2_s2": result["omega_squared_rad2_s2"],
        "mechanical_modes": result["mechanical_modes"],
        "x_m": np.linspace(0., float(case["geometry"]["L"]),
                           result["mechanical_modes"].shape[2]),
        "s_m": np.arange(result["mechanical_modes"].shape[3]) *
               (2. * np.pi * float(case["geometry"]["R"]) /
                result["mechanical_modes"].shape[3]),
    }
    if result["electrical_modes"] is not None:
        arrays["electrical_modes"] = result["electrical_modes"]
    tmp = out / "REFERENCE_MODAL_MODES.npz.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, out / "REFERENCE_MODAL_MODES.npz")
    summary = {
        "solver_backend": BACKEND, "calculation_kind": "MODAL",
        "method": "FDM", "mode": "LINEAR_ELASTIC", "theory": result["theory"],
        "scale_model": case["scale_model"], "material_model": "HOOKE",
        "status": "MODAL_COMPLETED", "converged": True,
        "accepted_steps": int(len(result["frequencies_hz"])),
        "n_modes": int(len(result["frequencies_hz"])),
        "density_kg_m3": float(result["density_kg_m3"]),
        "frequencies_hz": [float(v) for v in result["frequencies_hz"]],
        "omega_rad_s": [float(v) for v in result["omega_rad_s"]],
        "mechanical_free_dofs": int(result["n_mechanical_free_dofs"]),
        "electric_free_dofs": int(result["n_electric_free_dofs"]),
        "modes_file": "REFERENCE_MODAL_MODES.npz",
        "output_dir": str(out), "reference_only": True,
        "mesh_convergence_verified": False,
        "benchmark_validation_verified": False,
        "linearization": "undeformed_unloaded_zero_state",
        "mass": "consistent_theory_specific_weak_inertia",
        "electric_inertia": False,
        "normalization": "reduced_mechanical_mass_orthonormal",
        "units": {"frequency": "Hz", "omega": "rad/s", "geometry": "m",
                  "density": "kg/m3", "material": "Pa"},
    }
    (out / "REFERENCE_CASE.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "FINAL_RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


__all__ = ("THEORIES", "prepare_modal_case", "solve_modal_case", "run_modal_case")
