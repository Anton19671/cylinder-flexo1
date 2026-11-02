#!/usr/bin/env python3
"""External CUF 2023 Table 10 C-C R=1m shell benchmark, first 8 frequencies.

Source: Liang, Liu, Li, Wang, Materials 2023, 16(9), 3345,
DOI 10.3390/ma16093345, Table 10 CC R=1.
Omega = omega * L * sqrt(rho / (E * (2 + 2*nu))),
omega = 2*pi*f. Published figures are rounded to four decimals.

This is independent 3D elasticity literature, unlike the internal FDM/MVI
comparison. The test explicitly changes the old Rawat geometry/material and
never substitutes the eight published values for calculated frequencies.
"""
from __future__ import annotations

import csv
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import (
    _effective_modal_case, _modal_spec, solve_modal_case)
from modal_mvi_rawat_te1_cc_audit import CONFIG, read_case, harmonic_numbers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "config/verification/carrera2023/Liang_Liu_Li_Wang_2023_Table10_CC_R1.csv"
DEST = ROOT / "benchmark_carrera2023_cc_r1"
THEORIES = ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
DENSITY = 7850.
ELASTIC_MODULUS = 206e9
POISSON = .3
RADIUS_M = 1.
LENGTH_M = 2.
THICKNESS_M = .1
METHODS = ("FDM", "MVI_1", "MVI_2", "MVI_3", "MVI_4",
           "MVI_5", "MVI_6", "MVI_7", "MVI_8")
N_MODES = 8
GRID = 13
MAX_SWEEPS = 40


def reference():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8 or [int(r["mode"]) for r in rows] != list(range(1, 9)):
        raise ValueError("CARRERA2023_LITERATURE_TABLE_INVALID_EIGHT_MODES")
    for row in rows:
        expected = {
            "bc": "CC", "R_m": RADIUS_M, "L_m": LENGTH_M,
            "h_m": THICKNESS_M, "E_pa": ELASTIC_MODULUS,
            "nu": POISSON, "rho_kg_m3": DENSITY,
            "reference_doi": "10.3390/ma16093345"}
        for k, v in expected.items():
            item = row[k]
            if (item != v if isinstance(v, str) else float(item) != v):
                raise ValueError(f"CARRERA2023_REFERENCE_CASE_MISMATCH_{k}")
    return np.asarray([float(row["omega_dimensionless"]) for row in rows])


def set_case(theory, method, n=GRID):
    if theory not in THEORIES or method not in METHODS or n < 9 or n % 2 != 1:
        raise ValueError("CARRERA2023_INVALID_CASE")
    suffix = "FDM" if method == "FDM" else "MVI" + method.rsplit("_", 1)[-1]
    dir_name = "fdm_13x13" if method == "FDM" else "mvi_13x13"
    path = CONFIG / dir_name / (
        f"RAWAT2016_{theory}_13x13_MODAL_{suffix}.json")
    case = copy.deepcopy(read_case(path))
    case.update(geometry={"L": LENGTH_M, "R": RADIUS_M, "h": THICKNESS_M},
                material={"E0": ELASTIC_MODULUS, "nu0": POISSON},
                output_dir=f"OUTPUT_CARRERA2023_{theory}_{method}_{n}x{n}")
    case["modal"]["density_kg_m3"] = DENSITY
    case["modal"]["n_modes"] = N_MODES
    grid_field = "fdm_grid" if method == "FDM" else "mvi_grid"
    case[grid_field] = {
        "nx": n, "ns": n,
        "operator_scheme": ("SBP42_FOURIER_ODD_CSR" if method == "FDM"
                            else "SBP42_FOURIER_ODD")}
    if method != "FDM":
        case["reference_mvi"]["max_sweeps"] = MAX_SWEEPS
        case["modal"]["eigen_tolerance"] = 1e-7
        case["modal"]["mode_tolerance"] = 1e-5
    return case


def one(theory, method, n):
    path = DEST / theory / method / f"{n}x{n}"
    path.mkdir(parents=True, exist_ok=True)
    result_file = path / "result.json"
    t0 = time.perf_counter()
    data = {"status": "ERROR", "theory": theory, "method": method,
            "grid": [n, n], "modes_requested": N_MODES,
            "published_reference": {
                "source": "Liang et al., Materials 2023, 16(9), 3345, Table 10 CC R=1m",
                "doi": "10.3390/ma16093345",
                "normalization": "Omega=2*pi*f*L*sqrt(rho/(E*(2+2*nu)))",
                "published_precision": "four decimal places",
                "normalized_omega": reference().tolist(),
                "independent_3D_comparison": True,
            },
            "shell": {
                "L_m": LENGTH_M, "R_m": RADIUS_M, "h_m": THICKNESS_M,
                "E_pa": ELASTIC_MODULUS, "nu": POISSON,
                "rho_kg_m3": DENSITY, "support": "C-C",
                "thickness_order": int(theory[-1])
            },
            "literature_mode_matching": "FIRST_EIGHT_SORTED_ONLY",
            "physical_model_equivalent_to_article": "PENDING_KINEMATIC_AND_TRACE_CHECK",
            "notes": [
                "Full 3D literature calculation and this finite-order CUF shell model are not identical theories.",
                "A numerical difference is not automatically a numerical-algorithm defect.",
                "CC in the code constrains all generalized displacement components on both ends.",
                "No published Carrera frequency is used to construct the eigenproblem.",
            ]}
    try:
        case = set_case(theory, method, n)
        prepared, density, nreq = _modal_spec(case)
        if nreq != N_MODES or not math.isclose(density, DENSITY):
            raise ValueError("CARRERA2023_WRONG_MODAL_CASE")
        if prepared.mechanical.shape != (3 * (int(theory[-1]) + 1), n, n):
            raise ValueError("CARRERA2023_TE_COMPONENTS_INCORRECT")
        print(f"[CARRERA2023] START {theory} {method} {n}x{n}", flush=True)
        result = solve_modal_case(case)
        f = np.asarray(result["frequencies_hz"], float)
        q = np.asarray(result["mechanical_modes"], float)
        if (f.shape != (N_MODES,) or q.shape[0] != N_MODES or
                np.any(f <= 0) or not np.isfinite(f).all() or
                not np.isfinite(q).all() or
                np.any(np.diff(f) < -1e-7)):
            raise ValueError("CARRERA2023_INCOMPLETE_OR_UNORDERED_SPECTRUM")
        branch = result.get("modal_solver_branch")
        if method == "FDM" and branch != "MKR_FULL_SPACE":
            raise ValueError("CARRERA2023_WRONG_FDM_BRANCH")
        if method != "FDM" and (branch != "MVI_FIXED_RANK_YX" or
               result.get("classical_unrestricted_mvi") is not True or
               result.get("harmonic_sector_restriction") is not False):
            raise ValueError("CARRERA2023_WRONG_MVI_BRANCH")
        norm_ref = reference()
        coefficient = 2 * math.pi * LENGTH_M * math.sqrt(
            DENSITY / (ELASTIC_MODULUS * (2 + 2 * POISSON)))
        norm = f * coefficient
        relative = (norm / norm_ref - 1) * 100
        np.savez_compressed(path / "modes.npz",
                            mechanical_modes=q,
                            frequencies_hz=f,
                            normalized_omega=norm)
        data.update(
            status="COMPUTED_COMPARE_ONLY",
            frequencies_hz=f.tolist(),
            normalized_omega=norm.tolist(),
            published_frequencies_hz=(norm_ref / coefficient).tolist(),
            omega_difference=(norm - norm_ref).tolist(),
            frequency_percent_difference=relative.tolist(),
            maximum_absolute_percent_difference=float(np.max(np.abs(relative))),
            frequency_rounding_halfstep_hz=float(.00005 / coefficient),
            circumferential_harmonic_n=harmonic_numbers(q),
            global_spectrum_completeness_verified=result.get(
                "global_spectrum_completeness_verified"),
            mvi_sweeps=result.get("mvi_sweeps"),
            full_discrete_reference_used_as_trial=False,
            computed_mode_file="modes.npz",
            solver_branch=branch)
        print("[CARRERA2023] RESULT " + json.dumps({
            "theory": theory, "method": method, "grid": n,
            "freq_hz": data["frequencies_hz"],
            "omega": data["normalized_omega"],
            "diff_pct": data["frequency_percent_difference"],
            "mvi_sweeps": data["mvi_sweeps"]}), flush=True)
        return 0
    except Exception as exc:
        data["error"] = repr(exc)
        data["traceback"] = traceback.format_exc(limit=8)
        print("[CARRERA2023] ERROR " + json.dumps({
            "theory": theory, "method": method, "error": repr(exc)
        }), flush=True)
        return 1
    finally:
        data["wall_seconds"] = round(time.perf_counter() - t0, 4)
        result_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: script.py CUF_TE1..TE4 FDM|MVI_1..8 [odd-grid-n]")
    n = int(sys.argv[3]) if len(sys.argv) == 4 else GRID
    raise SystemExit(one(sys.argv[1], sys.argv[2], n))
