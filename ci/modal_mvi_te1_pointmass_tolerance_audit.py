#!/usr/bin/env python3
"""Physical CUF TE1 nonseparable modal audit: single added radial point mass.

The underlying shell equations/operators are unchanged. The added kinetic
energy is 0.5 * POINT_MASS_KG * w0(x_i,theta_j)^2. In the full-space FDM
branch, it is added explicitly to its separately assembled M matrix. In MVI,
both the physical mass action and the exact frozen-factor projected mass are
augmented by the same rank-one term. No eigenvector passes between solvers.
"""
from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from cylinder_paths import activate
activate()
from cylinder_modal_reference import _effective_modal_case, _modal_spec
from cylinder_modal_mvi import _SeparatedModalProblem, solve_modal_mvi
from cylinder_modal_matrixfree import _projected_actions
from cylinder_matrixfree_fdm import SparseTraceProjector

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"
OUT = ROOT / "modal_te1_pointmass_tolerance_audit.json"
MVI_CASE = BASE / "mvi_13x13/RAWAT2016_CUF_TE1_13x13_MODAL_MVI1.json"
FDM_CASE = BASE / "fdm_13x13/RAWAT2016_CUF_TE1_13x13_MODAL_FDM.json"
NX = NS = 9
N_MODES = 2
POINT_MASS_KG = 500.0
TOLS = (1e-5, 1e-7, 1e-9)
RANKS = (1, 2)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_case(method, rank=1):
    case = copy.deepcopy(read(MVI_CASE if method == "MVI" else FDM_CASE))
    key = "mvi_grid" if method == "MVI" else "fdm_grid"
    case[key] = {
        "nx": NX, "ns": NS,
        "operator_scheme": "SBP42_FOURIER_ODD" + ("_CSR" if method == "FDM" else "")
    }
    case["modal"]["n_modes"] = N_MODES
    case["modal"]["density_kg_m3"] = 7850.0
    if method == "MVI":
        case["rank"] = rank
        case["reference_mvi"] = {"max_sweeps": 70}
        case["modal"].pop("max_iterations", None)
    return case


def point_index(qshape):
    # TE1 mechanical component layout is u0,u1,v0,v1,w0,w1.
    if qshape != (6, NX, NS):
        raise ValueError(f"Unexpected physical CUF TE1 field ordering: {qshape}")
    return int(np.ravel_multi_index((4, NX // 2, NS // 3), qshape))


def full_space_fdm(case, point, mass_kg):
    """Assemble a complete eigenproblem using FDM-owned sparse BC projector.

    It does NOT call the MVI projector or use MVI trial fields. The only
    shared element is the unchanged physics of the cylindrical shell.
    """
    prepared, density, requested = _modal_spec(case)
    q0 = prepared.mechanical
    scales0 = __import__("cylinder_variational_fdm").state_scales(
        prepared.problem.functional, q0, None, prepared.settings)
    projector = SparseTraceProjector.from_problem(
        prepared.problem, q0, None, scales0,
        time_s=None, max_rows=prepared.settings.max_constraint_rows)
    c = projector.c
    if ((c.getnnz(axis=1) != 1).any() or
            (c.getnnz(axis=0) > 1).any() or
            np.max(abs(projector.target), initial=0.) > 1e-12):
        raise ValueError("POINTMASS_FDM_REQUIRES_INDEPENDENT_SIMPLE_CLAMPED_TRACE")
    free = np.flatnonzero(c.getnnz(axis=0) == 0)
    if point not in free:
        raise ValueError("Point mass placed on a fixed degree of freedom")
    k_action, m_action, scales, conversion, mass_scale = _projected_actions(
        prepared, density, projector)
    if not np.allclose(scales, scales0, rtol=1e-12, atol=0.):
        raise ValueError("Unexpected FDM scale mismatch")
    n = len(free)
    k = np.empty((n, n))
    m0 = np.empty((n, n))
    trial = np.zeros(q0.size)
    for column, j in enumerate(free):
        trial[j] = 1.0
        k[:, column] = (k_action @ trial)[free]
        m0[:, column] = (m_action @ trial)[free]
        trial[j] = 0.0
    for tag, matrix in (("K", k), ("M", m0)):
        asymmetry = (np.linalg.norm(matrix - matrix.T, ord=np.inf) /
                     max(np.linalg.norm(matrix, ord=np.inf), 1e-30))
        if asymmetry > 1e-7:
            raise ValueError(f"Physical FDM {tag} asymmetry: {asymmetry}")
        matrix[:] = 0.5 * (matrix + matrix.T)
    m = m0.copy()
    local = int(np.flatnonzero(free == point)[0])
    point_term = float(mass_kg * scales[point] ** 2 / mass_scale)
    m[local, local] += point_term

    def solve(matrix):
        vals, vecs = eigh(k, matrix, subset_by_index=(0, requested - 1),
                          driver="gvx", check_finite=True)
        if not np.isfinite(vals).all() or np.any(vals < 0.):
            raise ValueError("Nonpositive or nonfinite full-space eigenvalues")
        freq = np.sqrt(vals * conversion) / (2.0 * np.pi)
        fields = np.zeros((requested,) + q0.shape)
        for j in range(requested):
            scaled_mode = np.zeros(q0.size)
            scaled_mode[free] = vecs[:, j]
            fields[j] = (scales * scaled_mode /
                         np.sqrt(mass_scale)).reshape(q0.shape)
        return freq, fields

    base_freq, _ = solve(m0)
    pert_freq, modes = solve(m)
    return {
        "unperturbed_hz": base_freq.tolist(),
        "perturbed_hz": pert_freq.tolist(),
        "unperturbed_pair_gap_hz": float(base_freq[1] - base_freq[0]),
        "perturbed_pair_gap_hz": float(pert_freq[1] - pert_freq[0]),
        "nonseparable_angular_energy_fraction": [
            angular_mixing(field) for field in modes
        ],
        "free_dofs": int(n),
        "point_mass_scaled_diagonal": point_term,
        "solver": "FULL_SPACE_FDM_OWN_SPARSE_TRACE_AND_DENSE_EIGH",
        "operator_scheme": _effective_modal_case(case)["fdm_grid"]["operator_scheme"]
    }, pert_freq, modes


def angular_mixing(field):
    # Fourier sectors are orthogonal on the uniform periodic grid.
    power = np.sum(np.abs(np.fft.rfft(field, axis=-1)) ** 2, axis=(0, 1))
    total = float(np.sum(power))
    return 0.0 if not total else float(1 - np.max(power) / total)


def solve_mvi(case, point, mass_kg, tolerance, full_freq, full_modes):
    prepared, density, requested = _modal_spec(case)
    rank = case["rank"]
    op = _SeparatedModalProblem(prepared, density, rank)
    # Physical kinetic-energy perturbation for each arbitrary mechanical field.
    original_mass = op.mass
    original_projection = op.cached_cuf_projection

    def lumped_mass(direction):
        v = np.asarray(direction, float).ravel()
        physical = np.array(original_mass(v), copy=True)
        physical[point] += mass_kg * v[point]
        return physical

    def lumped_projected_matrices(bm):
        output = original_projection(bm)
        if output is None:
            # Original MVI assembly calls the modified mass action columnwise.
            return None
        k, m = output
        r = bm[point, :]
        return k, m + mass_kg * np.outer(r, r)

    op.mass = lumped_mass
    op.cached_cuf_projection = lumped_projected_matrices

    current_attempt = {}
    histories = {}
    def trace(**event):
        stage = event.get("stage")
        mode = event.get("mode")
        if stage == "MVI_CANDIDATE_START":
            current_attempt[mode] = event["attempt"]
        if stage in ("MVI_HALFSTEP_DONE", "MVI_SWEEP_DONE"):
            key = f"{mode}:{current_attempt.get(mode, 0)}"
            histories.setdefault(key, []).append({
                "stage": stage,
                "sweep": event.get("sweep"),
                "axis": event.get("axis"),
                "frequency_hz": event.get("frequency_hz"),
                "relative_lambda_change": event.get("frequency_squared_change_relative"),
                "relative_shape_change": event.get("shape_change_relative")
            })

    start = time.monotonic()
    result = solve_modal_mvi(
        prepared, density, requested, _cached_operator=op,
        eigen_tolerance=tolerance, mode_tolerance=tolerance,
        detail_callback=trace)
    elapsed = time.monotonic() - start
    f = np.asarray(result["frequencies_hz"], dtype=float)
    modes = np.asarray(result["mechanical_modes"], dtype=float)
    if f.shape != (N_MODES,) or modes.shape[0] != N_MODES:
        raise ValueError("MVI did not return the requested two modes")
    if (result.get("method") != "MVI" or
            result.get("mvi_modal_algorithm") !=
                    "fixed_rank_constrained_variational_YX_stationarity" or
            result.get("harmonic_sector_restriction") is not False):
        raise ValueError("MVI route was not unrestricted Y/X")
    def mass_mac(u, v):
        u, v = u.ravel(), v.ravel()
        cross = float(u @ op.mass(v))
        norm_u = float(u @ op.mass(u))
        norm_v = float(v @ op.mass(v))
        return abs(cross) / np.sqrt(norm_u * norm_v)

    selected_histories = []
    for j, record in enumerate(result["mvi_multistart_audit"]):
        original_mode = result["mode_discovery_indices"][j]
        selected_histories.append(
            histories.get(f"{original_mode}:{record['selected_attempt']}", []))
    return {
        "status": "SOLVED",
        "rank": rank,
        "eigen_tolerance": tolerance,
        "mode_tolerance": tolerance,
        "frequencies_hz": f.tolist(),
        "absolute_fdm_differences_hz": np.abs(f - full_freq).tolist(),
        "relative_fdm_differences": (
            np.abs(f - full_freq) / full_freq).tolist(),
        "pair_gap_hz": float(f[1] - f[0]),
        "pair_gap_difference_hz": float((f[1] - f[0]) -
                                        (full_freq[1] - full_freq[0])),
        "direct_mass_mac": [mass_mac(modes[j], full_modes[j])
                            for j in range(N_MODES)],
        "nonseparable_angular_energy_fraction": [
            angular_mixing(field) for field in modes
        ],
        "mvi_sweeps": result["mvi_sweeps"],
        "accepted_candidate_histories": selected_histories,
        "global_spectrum_completeness_verified": result.get(
            "global_spectrum_completeness_verified"),
        "wall_seconds": round(elapsed, 3)
    }


def main():
    start = time.monotonic()
    report = {
        "problem": "CUF TE1 C-C 9x9 SBP42, added local point mass on w0",
        "point_mass_kg": POINT_MASS_KG,
        "point_component": "w0",
        "point_grid_index": [NX // 2, NS // 3],
        "n_modes": N_MODES, "physical_operator_changed": False,
        "perturbation": "additive physical point kinetic energy, applied separately to FDM/MVI",
        "fdm_bc": "SparseTraceProjector from independent FDM case",
        "mvi_bc": "affine_trace_space from MVI case",
        "reference_is_independent_physical_validation": False,
        "runs": {}, "completed": False
    }
    try:
        fdm_case = make_case("FDM")
        mvi_case = make_case("MVI", 1)
        eff_fdm = _effective_modal_case(fdm_case)
        eff_mvi = _effective_modal_case(mvi_case)
        for key in ("theory", "cuf_model", "geometry", "material",
                    "quadrature", "boundary_conditions"):
            if eff_fdm.get(key) != eff_mvi.get(key):
                raise ValueError(f"Mismatch in physical input: {key}")
        if eff_fdm["fdm_grid"]["operator_scheme"] != (
                eff_mvi["mvi_grid"]["operator_scheme"] + "_CSR"):
            raise ValueError("Different physical discretization for pointmass test")
        point = point_index((6, NX, NS))
        fdm, frequencies, fields = full_space_fdm(
            fdm_case, point, POINT_MASS_KG)
        report["fdm"] = fdm
        print("[POINTMASS] FDM " + json.dumps(fdm), flush=True)
        for rank in RANKS:
            tolerances = TOLS if rank == 1 else (1e-7,)
            for tol in tolerances:
                name = f"MVI_{rank}_tol_{tol:.0e}"
                print(f"[POINTMASS] START {name}", flush=True)
                case = make_case("MVI", rank)
                t = time.monotonic()
                try:
                    result = solve_mvi(case, point, POINT_MASS_KG,
                                       tol, frequencies, fields)
                except Exception as error:
                    result = {
                        "status": "ERROR", "rank": rank,
                        "eigen_tolerance": tol,
                        "mode_tolerance": tol,
                        "error": repr(error),
                        "traceback": traceback.format_exc(limit=6),
                        "wall_seconds": round(time.monotonic() - t, 3)
                    }
                report["runs"][name] = result
                print("[POINTMASS] RESULT " + name + " " +
                      json.dumps({k: v for k, v in result.items()
                                  if k != "accepted_candidate_histories" and
                                  k != "traceback"}), flush=True)
                report["elapsed_s"] = round(time.monotonic() - start, 3)
                OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["completed"] = all(
            r.get("status") == "SOLVED" for r in report["runs"].values())
        return 0 if report["completed"] else 1
    except Exception as error:
        report["fatal_error"] = repr(error)
        report["traceback"] = traceback.format_exc(limit=10)
        print("[POINTMASS] ERROR " + repr(error), flush=True)
        return 1
    finally:
        report["elapsed_s"] = round(time.monotonic() - start, 3)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[POINTMASS] SUMMARY " + json.dumps({
            "completed": report["completed"],
            "elapsed_s": report["elapsed_s"],
            "statuses": {k: r["status"] for k, r in report["runs"].items()}
        }), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
