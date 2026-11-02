#!/usr/bin/env python3
"""Cross-grid CUF TE1 point-mass modal comparison: full FDM vs true MVI-4.

Each grid (9, 13, 17) uses the same physical shell, C-C supports, SBP42
operators, 500 kg point mass at x=L/2 and theta=0, and first two modes.
FDM assembles a complete matrix with its own sparse trace projector; MVI
uses rank-four separated component fields and unrestricted Y/X iterations.
The published 9x9 rank/tolerance study remains unchanged.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
import modal_mvi_te1_pointmass_tolerance_audit as audit

ROOT = Path(__file__).resolve().parents[1]
RANK = 4
ADDED_MASS_KG = 500.0
TOLERANCE = 1e-7
GRIDS = (9, 13, 17)


def run_grid(n):
    if n not in GRIDS:
        raise ValueError("GRID_STUDY_ONLY_9_13_17")
    audit.NX = audit.NS = n
    point = audit.point_index((6, n, n))
    fdm_case = audit.make_case("FDM")
    mvi_case = audit.make_case("MVI", RANK)
    e_fdm = audit._effective_modal_case(fdm_case)
    e_mvi = audit._effective_modal_case(mvi_case)
    for key in ("theory", "cuf_model", "geometry", "material",
                "quadrature", "boundary_conditions"):
        if e_fdm.get(key) != e_mvi.get(key):
            raise ValueError(f"GRID_MVI_FDM_DIFFERENT_PHYSICAL_CASE_{key}")
    if (e_fdm["fdm_grid"]["operator_scheme"] !=
            e_mvi["mvi_grid"]["operator_scheme"] + "_CSR"):
        raise ValueError("GRID_MVI_FDM_DIFFERENT_SBP_OPERATOR")
    report = {
        "grid": [n, n],
        "model": "CUF_TE1",
        "boundary_conditions": "C-C axially and periodic circumferentially",
        "point_component": "w0",
        "point_grid_index": [n // 2, 0],
        "theta_rad": 0.0,
        "point_mass_kg": ADDED_MASS_KG,
        "shell_mass_kg": float(2 * np.pi * 1 * 2 * .05 * 7850),
        "mvi_rank": RANK,
        "mvi_rank_scope": "per component",
        "mvi_eigen_tolerance": TOLERANCE,
        "mvi_mode_tolerance": TOLERANCE,
        "n_modes": 2,
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "comparison_scope": "same discrete physical model and different numerical algorithms",
        "grid_independent_3d_reference": False,
        "complete": False,
    }
    try:
        t = time.perf_counter()
        ref, fdm_freq, fdm_modes = audit.full_space_fdm(
            fdm_case, point, ADDED_MASS_KG)
        report["fdm"] = ref
        report["fdm_hz"] = np.asarray(fdm_freq, float).tolist()
        report["fdm_unperturbed_hz"] = ref["unperturbed_hz"]
        report["fdm_measured_total_s"] = float(time.perf_counter() - t)
        print("[GRID] FDM " + json.dumps({
            "grid": n, "freq_hz": report["fdm_hz"],
            "fdm_total_s": ref["fdm_perturbed_total_wall_seconds"],
            "free_dofs": ref["free_dofs"]}), flush=True)

        mvi = audit.solve_mvi(mvi_case, point, ADDED_MASS_KG,
                              TOLERANCE, fdm_freq, fdm_modes)
        if mvi["status"] != "SOLVED":
            raise ValueError("MVI_GRID_NO_VALID_RESULT")
        report["mvi"] = mvi
        report["mvi_hz"] = mvi["frequencies_hz"]
        report["delta_hz_signed"] = mvi["signed_frequency_differences_hz"]
        report["delta_hz_absolute"] = mvi["absolute_fdm_differences_hz"]
        report["max_abs_freq_difference_hz"] = float(
            max(mvi["absolute_fdm_differences_hz"]))
        report["fmvi_per_fdm"] = (
            np.asarray(mvi["frequencies_hz"], float) /
            np.asarray(fdm_freq, float)).tolist()
        report["mvi_wall_seconds"] = mvi["wall_seconds"]
        report["fdm_wall_seconds"] = ref["fdm_perturbed_total_wall_seconds"]
        report["mvi_to_fdm_time_ratio"] = (
            report["mvi_wall_seconds"] / report["fdm_wall_seconds"])
        report["complete"] = True
        print("[GRID] RESULT " + json.dumps({
            "grid": n,
            "fdm_hz": report["fdm_hz"],
            "mvi_hz": report["mvi_hz"],
            "delta_hz_signed": report["delta_hz_signed"],
            "mvi_sweeps": mvi["mvi_sweeps"],
            "mvi_s": report["mvi_wall_seconds"],
            "fdm_s": report["fdm_wall_seconds"],
            "time_ratio": report["mvi_to_fdm_time_ratio"],
            "mvi_mode_mac": mvi["direct_mass_mac"]
        }), flush=True)
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc(limit=8)
        print("[GRID] ERROR " + json.dumps({
            "grid": n, "error": repr(exc)}), flush=True)
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python ci/modal_te1_pointmass_grid_study.py 9|13|17")
    try:
        n = int(sys.argv[1])
    except ValueError as exc:
        raise SystemExit("grid must be an integer") from exc
    report = run_grid(n)
    out = ROOT / f"modal_te1_pointmass_grid_{n}x{n}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("[GRID] WRITTEN " + str(out), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
