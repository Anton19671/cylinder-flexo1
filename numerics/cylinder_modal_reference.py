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
import time
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
    # For an unloaded, undeformed modal analysis the tangent is the same
    # first variation for LINEAR, VON_KARMAN and GREEN_LAGRANGE.
    # Their nonlinear terms only contribute above quadratic order in the
    # stationary energy. Preserve the requested kinematics in the CASE.
    geometry_model = str(case.get("geometry_model", "")).upper()
    if geometry_model not in ("LINEAR", "VON_KARMAN", "GREEN_LAGRANGE"):
        raise ValueError("MODAL geometry_model must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")
    method = str(case.get("method", "")).upper()
    if method not in ("FDM", "MVI"):
        raise ValueError("MODAL method must be FDM or MVI")
    if method == "MVI":
        rank = _integer(case.get("rank"), "MODAL MVI rank", 1)
        if rank > 8:
            raise ValueError("MODAL MVI rank must be 1..8")
        options = case.get("reference_mvi", {})
        if not isinstance(options, dict) or set(options) - {"max_sweeps"}:
            raise ValueError("MODAL MVI reference_mvi accepts only max_sweeps")
    elif case.get("rank") is not None or case.get("reference_mvi") is not None:
        raise ValueError("FDM MODAL cannot contain MVI rank or settings")
    backend = str(case.get("fdm_backend", "DENSE_REFERENCE")).upper()
    if method == "MVI" and backend != "DENSE_REFERENCE":
        raise ValueError("MODAL MVI uses the independent dense separated solver")
    if method == "FDM" and backend not in ("DENSE_REFERENCE", "MATRIX_FREE_GMRES"):
        raise ValueError("MODAL FDM backend must be DENSE_REFERENCE or MATRIX_FREE_GMRES")
    if backend == "MATRIX_FREE_GMRES":
        grid = case.get("fdm_grid", {})
        if grid.get("operator_scheme") not in ("SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR"):
            raise ValueError("MATRIX_FREE_MODAL_REQUIRES_SBP21_PERIODIC_ODD_CSR")
        if str(case.get("scale_model", "")).upper() != "MACRO":
            raise ValueError("MATRIX_FREE_MODAL_ELECTRIC_SCHUR_NOT_IMPLEMENTED: MACRO only")
    if case.get("dynamics") is not None:
        raise ValueError("MODAL has no time integration")
    if case.get("load_definition") is not None or case.get("reference_loads") is not None:
        raise ValueError("MODAL free vibrations require an unloaded CASE")
    raw = case.get("modal", {})
    if not isinstance(raw, dict) or set(raw) - {"density_kg_m3", "n_modes", "eigen_tolerance", "max_iterations"}:
        raise ValueError("modal accepts density_kg_m3, n_modes, eigen_tolerance, max_iterations")
    density = _real(raw.get("density_kg_m3"), "modal.density_kg_m3", True)
    n_modes = _integer(raw.get("n_modes"), "modal.n_modes", 1)
    if n_modes > 32:
        raise ValueError("MODAL supports at most 32 requested modes")
    if "eigen_tolerance" in raw:
        tol = _real(raw["eigen_tolerance"], "modal.eigen_tolerance", True)
        if tol >= 1.:
            raise ValueError("modal.eigen_tolerance must be less than 1")
    if "max_iterations" in raw:
        limit = _integer(raw["max_iterations"], "modal.max_iterations", 1)
        if limit > 2000:
            raise ValueError("modal.max_iterations must not exceed 2000")
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
    surrogate["mode"] = ("LINEAR_ELASTIC" if geometry_model=="LINEAR"
                         else "NONLINEAR_ELASTIC")
    # Use the selected independent kinematic branch when assembling K0.
    # No conversion to GREEN_LAGRANGE is permitted.
    surrogate["geometry_model"] = geometry_model
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
    if prepared.fdm_backend == "MATRIX_FREE_GMRES":
        from cylinder_matrixfree_fdm import SparseTraceProjector
        scales = state_scales(f, q, p, prepared.settings)
        projector = SparseTraceProjector.from_problem(
            prepared.problem, q, p, scales, time_s=None,
            max_rows=prepared.settings.max_constraint_rows)
        if np.max(np.abs(projector.lift), initial=0.) > 1.e-10:
            raise ValueError("MODAL requires homogeneous essential boundary conditions")
        available = q.size - projector.count
        if requested > available or available < 5 * requested:
            raise ValueError("MODAL requested modes exceed matrix-free trial space")
        return {"theory": f.theory, "method": "FDM",
                "calculation_kind": "MODAL", "density_kg_m3": density,
                "n_modes": requested, "mechanical_free_dofs": available,
                "electric_free_dofs": 0, "reference_only": True,
                "fdm_backend": "MATRIX_FREE_GMRES"}
    first = prepared.problem.assemble(q, p)
    if np.max(np.abs(first.space.origin), initial=0.) > 1.e-10:
        raise ValueError("MODAL supports only homogeneous essential BC")
    if (prepared.method == "FDM" and prepared.fdm_backend == "DENSE_REFERENCE"
            and first.space.basis.shape[1] > prepared.settings.max_free_dofs):
        raise ValueError("MODAL FDM exceeds reference_newton.max_free_dofs")
    scales = state_scales(f, q, p, prepared.settings)
    mechanical_basis, electric_basis = _modal_bases(
        first.space.matrix, first.space.target, scales, q.size)
    if mechanical_basis.shape[1] < requested:
        raise ValueError("modal.n_modes exceeds unconstrained mechanical DOFs")
    if prepared.method == "MVI":
        from cylinder_modal_mvi import MAX_PROJECTED_DIRECTIONS
        rank = prepared.mvi_settings.rank
        component_count = q.shape[0] + (0 if p is None else p.shape[0])
        direction_count = component_count * rank * max(q.shape[1:])
        if direction_count > MAX_PROJECTED_DIRECTIONS:
            raise ValueError("MODAL MVI projected direction limit exceeded")
        if rank > min(q.shape[1:]):
            raise ValueError("MODAL MVI rank exceeds grid dimensions")
    return {"theory": f.theory, "method": prepared.method,
            **({"rank": prepared.mvi_settings.rank} if prepared.method == "MVI" else {}),
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


def _linear_matrices(prepared, density, *, progress_callback=None, detail_callback=None):
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
        if detail_callback:detail_callback(stage='K_COLUMN_START',column=j+1,total_columns=k.shape[1])
        direction = directions[:, j]
        derivative = (residual(step * direction)
                      - residual(-step * direction)) / (2. * step)
        k[:, j] = directions.T @ derivative
        if detail_callback:detail_callback(stage='K_COLUMN_DONE',column=j+1,total_columns=k.shape[1])
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
        if detail_callback:detail_callback(stage='M_COLUMN_START',column=j+1,total_columns=n_m)
        acceleration = bm[:, j].reshape(q0.shape)
        mass_column = f.mass_action(
            acceleration, density_kg_m3=density,
            mechanical=q0, velocity=np.zeros_like(q0))
        m[:, j] = bm.T @ mass_column.ravel()
        if detail_callback:detail_callback(stage='M_COLUMN_DONE',column=j+1,total_columns=n_m)
    if not np.isfinite(m).all():
        raise ValueError("nonfinite MODAL mass")
    mass_asymmetry = float(np.linalg.norm(m - m.T, ord=np.inf))
    if mass_asymmetry > 1.e-9 * max(float(np.linalg.norm(m, ord=np.inf)), 1.e-30):
        raise ValueError("MODAL mass is not symmetric")
    m = (m + m.T) * 0.5
    cholesky(m, lower=True, check_finite=True)
    return k, m, bm, be, q0.shape, None if p0 is None else p0.shape


def solve_modal_case(case, *, base_dir=None, progress_callback=None, detail_callback=None):
    """Compute consistent small-grid modes in Hz, with massless electric DOFs condensed."""
    prepared, density, requested = _modal_spec(case, base_dir=base_dir)
    if prepared.method == "MVI":
        from cylinder_modal_mvi import solve_modal_mvi
        return solve_modal_mvi(prepared, density, requested,
                               progress_callback=progress_callback,detail_callback=detail_callback)
    if prepared.fdm_backend == "MATRIX_FREE_GMRES":
        from cylinder_modal_matrixfree import solve_modal_matrixfree
        return solve_modal_matrixfree(
            prepared, density, requested, modal_options=case.get("modal"),
            progress_callback=progress_callback, detail_callback=detail_callback)
    k, m, bm, be, q_shape, p_shape = _linear_matrices(
        prepared, density, progress_callback=progress_callback,detail_callback=detail_callback)
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
    if detail_callback:detail_callback(stage='EIGEN_BEGIN',n_modes=requested)
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
    if detail_callback:
        for j in range(requested):
            detail_callback(stage='FDM_MODE_DONE',mode=j+1,n_modes=requested,
                frequency_hz=float(frequencies_hz[j]),omega_rad_s=float(omega[j]),
                modal_amplitude_max=float(np.max(abs(q_modes[j]),initial=0.)))
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
    started=time.monotonic()
    print('MODAL PREPARE: checking geometry and boundary conditions',flush=True)
    prepare_modal_case(case, base_dir=path.parent)
    print(f'MODAL PREPARE DONE elapsed_s={time.monotonic()-started:.3f}',flush=True)
    out = Path(case["output_dir"]).expanduser()
    if not out.is_absolute():
        out = path.parent / out
    out = out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("OUTPUT_DIR_NOT_EMPTY: MODAL results must be separate")

    def report(current, total):
        print(f"MODAL STEP {current}/{total} elapsed_s={time.monotonic()-started:.3f}",flush=True)

    def details(**event):
        # Real solver stages, including all MVI X/Y substeps and K/M columns.
        event["elapsed_s"]=round(time.monotonic()-started,3)
        print("MODAL_EVENT "+json.dumps(event,ensure_ascii=False,allow_nan=False),flush=True)

    print("MODAL ASSEMBLY START",flush=True)
    result = solve_modal_case(case,base_dir=path.parent,
                              progress_callback=report,detail_callback=details)
    print(f"MODAL SOLVE DONE elapsed_s={time.monotonic()-started:.3f}",flush=True)
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
        "method": result.get("method", "FDM"), "mode": "LINEAR_ELASTIC", "theory": result["theory"],
        "fdm_backend": result.get("fdm_backend", "DENSE_REFERENCE"),
        **({"rank": result["rank"], "mvi_sweeps": result["mvi_sweeps"],
            "mvi_modal_algorithm": result["mvi_modal_algorithm"]}
           if "rank" in result else {}),
        "scale_model": case["scale_model"], "material_model": "HOOKE",
        "geometry_model": str(case["geometry_model"]).upper(),
        "status": "MODAL_COMPLETED", "converged": True,
        "accepted_steps": int(len(result["frequencies_hz"])),
        "n_modes": int(len(result["frequencies_hz"])),
        "density_kg_m3": float(result["density_kg_m3"]),
        "frequencies_hz": [float(v) for v in result["frequencies_hz"]],
        "omega_rad_s": [float(v) for v in result["omega_rad_s"]],
        "mechanical_free_dofs": int(result["n_mechanical_free_dofs"]),
        "electric_free_dofs": int(result["n_electric_free_dofs"]),
        "modes_file": "REFERENCE_MODAL_MODES.npz",
        "excel_file": "00_МОДАЛЬНЫЙ_АНАЛИЗ_ЧАСТОТЫ_ФОРМЫ.xlsx",
        "output_dir": str(out), "reference_only": True,
        "mesh_convergence_verified": False,
        "benchmark_validation_verified": False,
        "linearization": "undeformed_unloaded_zero_state",
        "linearization_source": str(case["geometry_model"]).upper()+"_TANGENT_AT_ZERO",
        "kinematics_linear_limit": True,
        "mass": "consistent_theory_specific_weak_inertia",
        "electric_inertia": False,
        "normalization": "reduced_mechanical_mass_orthonormal",
        "units": {"frequency": "Hz", "omega": "rad/s", "geometry": "m",
                  "density": "kg/m3", "material": "Pa"},
    }
    # Human-readable tables for frequencies, parameters and all modal fields.
    from cylinder_excel_tables import LabeledWorkbook, write_labeled_table
    with LabeledWorkbook(str(out / "00_МОДАЛЬНЫЙ_АНАЛИЗ_ЧАСТОТЫ_ФОРМЫ.xlsx")) as wb:
        write_labeled_table(wb,"FREQUENCIES",
            ("mode","frequency_hz","omega_rad_s","omega_squared_rad2_s2"),
            ((i+1,float(result["frequencies_hz"][i]),
              float(result["omega_rad_s"][i]),
              float(result["omega_squared_rad2_s2"][i]))
             for i in range(len(result["frequencies_hz"]))))
        geo=case["geometry"];mat=case["material"]
        grid=case.get("mvi_grid" if case["method"]=="MVI" else "fdm_grid",
                      case.get("grid",{}))
        mech_bc=case["boundary_conditions"]["variational"].get("mechanical",{})
        parameters=[("theory",result["theory"]),("density_kg_m3",result["density_kg_m3"]),
            ("L_m",geo["L"]),("R_m",geo["R"]),("h_m",geo["h"]),
            ("E0_Pa",mat["E0"]),("nu0",mat["nu0"]),
            ("nx",grid["nx"]),("ns",grid["ns"]),
            ("n_gauss",case["quadrature"]["n_gauss"]),
            ("n_modes",len(result["frequencies_hz"])),
            ("method",case["method"]),("rank",case.get("rank","")),
            ("support_left",mech_bc.get("left",{}).get("support","FREE")),
            ("support_right",mech_bc.get("right",{}).get("support","FREE"))]
        write_labeled_table(wb,"PARAMETERS",("code","value"),parameters)
        order=result["theory"]
        n=result["mechanical_modes"].shape[1]
        if order=="KL":components=("u_m","v_m","w_m")
        elif order in ("FSDT","REDDY"):
            components=("u_m","v_m","w_m","theta_x_rad","theta_s_rad")
        else:
            degree=int(order[-1])+1
            components=tuple(f"{axis}{i}_m" for axis in ("u","v","w") for i in range(degree))
        if len(components)!=n:raise ValueError("MODAL exported component order mismatch")
        nx,ns=result["mechanical_modes"].shape[2:]
        length=float(geo["L"]);radius=float(geo["R"])
        def mechanical_rows():
            for k in range(len(result["frequencies_hz"])):
                for ix in range(nx):
                    for js in range(ns):
                        yield (k+1,ix,js,length*ix/(nx-1),
                               2*np.pi*radius*js/ns,2*np.pi*js/ns,
                               *(float(result["mechanical_modes"][k,c,ix,js])
                                 for c in range(n)))
        write_labeled_table(wb,"MODE_FIELDS",
            ("mode","ix","js","x_m","s_m","theta_rad",*components),mechanical_rows())
        if result["electrical_modes"] is not None:
            npot=result["electrical_modes"].shape[1]
            def electric_rows():
                for k in range(len(result["frequencies_hz"])):
                    for ix in range(nx):
                        for js in range(ns):
                            yield (k+1,ix,js,*map(float,result["electrical_modes"][k,:,ix,js]))
            write_labeled_table(wb,"ELECTRIC_MODES",
                ("mode","ix","js",*(f"phi{i}_v" for i in range(npot))),electric_rows())
    (out / "REFERENCE_CASE.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "FINAL_RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


__all__ = ("THEORIES", "prepare_modal_case", "solve_modal_case", "run_modal_case")
