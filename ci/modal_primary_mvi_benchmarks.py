#!/usr/bin/env python3
"""MVI-first modal benchmark: Rawat/Cammalleri extra grids.

Runs the existing physical variational solver unchanged.  MKR/FDM is a
separate control, not a substitute for any MVI rank.  A missing or failed
spectrum is never replaced by a reference frequency.  The published
axial index m=1 is *not* certified from frequency proximity.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import (
    _check_unrestricted_modal_mvi_bounds, solve_modal_case,
)
from modal_mvi_rawat_te1_cc_audit import harmonic_numbers

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"
STUDIES = {
    "rawat_te1_17x17": ("CUF", "TE1", 17, 17, "rawat"),
    "rawat_te4_17x17": ("CUF", "TE4", 17, 17, "rawat"),
    "rawat_te4_21x21": ("CUF", "TE4", 21, 21, "rawat"),
    "cammalleri_kl_13x21": ("KL", None, 13, 21, "cammalleri"),
    "cammalleri_kl_17x31": ("KL", None, 17, 31, "cammalleri"),
    "cammalleri_kl_21x41": ("KL", None, 21, 41, "cammalleri"),
}
GROUPS = {
    "mkr": ["MKR_FDM"],
    "low": ["MVI_1", "MVI_2", "MVI_3", "MVI_4"],
    "high": ["MVI_5", "MVI_6", "MVI_7", "MVI_8"],
}
OUT = ROOT / "benchmark_modal_primary"
REFERENCE = {
    "rawat": {"doi": "10.16943/ptinsa/2016/48426",
              "source": "Table 2, p. 354, C-C; axial m=1",
              "n_hz": {"2": 342.0, "3": 269.0, "4": 277.0, "5": 352.0}},
    "cammalleri": {"doi": "10.3390/designs7060122",
              "source": "Table 6, PDF p. 11, C-C; axial m=1",
              "n_hz": {"5": 605.0, "6": 555.0, "7": 611.0},
              "ansys_hz": {"5": 581.0, "6": 538.0, "7": 600.0}},
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_case(study, method):
    theory, cuf, nx, ns, paper = STUDIES[study]
    name = ("CUF_" + cuf) if cuf else theory
    if method == "MKR_FDM":
        if paper == "rawat" and name == "CUF_TE4" and nx == 21:
            path = BASE / "mvi_21x21/RAWAT2016_CUF_TE4_21x21_MODAL_FDM_SBP42.json"
        elif paper == "rawat":
            path = BASE / f"RAWAT2016_{name}_17x17_MODAL_FDM.json"
        else:
            path = BASE / "fdm_13x13/RAWAT2016_KL_13x13_MODAL_FDM.json"
        case = read(path)
        case.pop("rank", None)
        case.pop("reference_mvi", None)
        case.pop("mvi_grid", None)
        case["method"] = "FDM"
        case["fdm_backend"] = "MATRIX_FREE_GMRES"
        case["fdm_grid"] = {
            "nx": nx, "ns": ns, "operator_scheme": "SBP42_FOURIER_ODD_CSR"}
        case["reference_newton"] = {
            "tangent_step_scaled": 2e-6,
            "max_total_dofs": 100000,
            "max_surface_nodes": 8192,
            "max_constraint_rows": 20000,
        }
        case["modal"]["eigen_tolerance"] = 1e-6
        case["modal"]["max_iterations"] = 400
    else:
        rank = int(method.rsplit("_", 1)[-1])
        if paper == "rawat" and name == "CUF_TE4" and nx == 21:
            path = BASE / "mvi_21x21/RAWAT2016_CUF_TE4_21x21_MODAL_MVI1_SBP42.json"
        elif paper == "rawat" and name == "CUF_TE4" and nx == 17:
            path = BASE / "mvi_17x17/RAWAT2016_CUF_TE4_17x17_MODAL_MVI1_SBP42.json"
        elif paper == "rawat":
            path = BASE / f"mvi_13x13/RAWAT2016_{name}_13x13_MODAL_MVI1.json"
        else:
            path = BASE / "mvi_13x13/RAWAT2016_KL_13x13_MODAL_MVI1.json"
        case = read(path)
        case.pop("fdm_grid", None)
        case["method"] = "MVI"
        case["fdm_backend"] = "DENSE_REFERENCE"
        case["mvi_grid"] = {
            "nx": nx, "ns": ns, "operator_scheme": "SBP42_FOURIER_ODD"}
        case["rank"] = rank
        case["reference_mvi"] = {"max_sweeps": 30}
        case["modal"].pop("eigen_tolerance", None)
        case["modal"].pop("max_iterations", None)
    if paper == "cammalleri":
        case["geometry"] = {"L": 0.305, "R": 0.076, "h": 0.000254}
        case["material"] = {"E0": 207e9, "nu0": 0.3}
        density = 7833
    else:
        case["geometry"] = {"L": 2.0, "R": 1.0, "h": 0.05}
        case["material"] = {"E0": 210e9, "nu0": 0.3}
        density = 7850
    case["theory"] = theory
    if cuf:
        case["cuf_model"] = cuf
    else:
        case.pop("cuf_model", None)
    case["calculation_kind"] = "MODAL"
    case["mode"] = "LINEAR_ELASTIC"
    case["geometry_model"] = "LINEAR"
    case["material_model"] = "HOOKE"
    case["scale_model"] = "MACRO"
    case["modal"]["density_kg_m3"] = density
    case["modal"]["n_modes"] = 12
    case["output_dir"] = f"OUTPUT_BENCH_{study}_{method}"
    case["boundary_conditions"]["variational"]["mechanical"] = {
        "left": {"support": "CLAMPED", "higher": "NATURAL"},
        "right": {"support": "CLAMPED", "higher": "NATURAL"},
    }
    return case


def run_group(study, group):
    if study not in STUDIES or group not in GROUPS:
        raise ValueError("Unknown benchmark study or group")
    theory, cuf, nx, ns, paper = STUDIES[study]
    folder = OUT / study / group
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "study": study, "grid": [nx, ns],
        "theory": "CUF_" + cuf if cuf else theory,
        "boundary": "C-C",
        "kinematics": "LINEAR",
        "mvi_is_primary": True,
        "mkr_is_control_only": True,
        "literature": REFERENCE[paper],
        "axial_m1_independently_verified": False,
        "runs": {},
    }
    for method in GROUPS[group]:
        case = make_case(study, method)
        (folder / (method + "_CASE.json")).write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record = {"status": "ERROR", "method": method, "grid": [nx, ns],
                  "requested_modes": case["modal"]["n_modes"]}
        start = time.monotonic()
        try:
            if method != "MKR_FDM":
                components = 3 if theory == "KL" else 3 * (int(cuf[-1]) + 1)
                _check_unrestricted_modal_mvi_bounds(nx, ns, components,
                                                     case["rank"])
            print(f"[MVI-PRIMARY] START {study} {method}", flush=True)
            solved = solve_modal_case(case)
            freq = np.asarray(solved["frequencies_hz"], float)
            fields = np.asarray(solved["mechanical_modes"], float)
            if (freq.shape != (12,) or fields.shape[0] != 12 or
                    fields.shape[-2:] != (nx, ns) or
                    not np.isfinite(freq).all() or np.any(freq <= 0) or
                    not np.isfinite(fields).all()):
                raise ValueError("Incomplete or nonfinite modal spectrum")
            nums = solved.get("circumferential_wavenumbers")
            if nums is None or len(nums) != len(freq):
                nums = harmonic_numbers(fields)
            nums = np.asarray(nums, int)
            np.savez_compressed(folder / (method + "_MODES.npz"),
                                frequencies_hz=freq,
                                circumferential_wavenumbers=nums,
                                mechanical_modes=fields)
            record.update(status="COMPUTED_AXIAL_M_UNVERIFIED",
                          frequencies_hz=freq.tolist(),
                          circumferential_wavenumbers=nums.tolist(),
                          global_spectrum_completeness_verified=solved.get(
                              "global_spectrum_completeness_verified"),
                          modes_file=method + "_MODES.npz")
        except Exception as exc:
            err = str(exc)
            record.update(status=("UNSUPPORTED_LIMIT" if
                          err.startswith("MODAL_MVI_UNRESTRICTED_") else "ERROR"),
                          error=repr(exc))
            print(f"[MVI-PRIMARY] {record['status']} {study} {method}: {exc}",
                  flush=True)
        finally:
            record["wall_seconds"] = round(time.monotonic() - start, 3)
            summary["runs"][method] = record
            (folder / (method + "_RESULT.json")).write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            (folder / "group_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print("[MVI-PRIMARY] RESULT " + json.dumps(
                {"study": study, "method": method, **record},
                ensure_ascii=False), flush=True)
    failed = any(r["status"] == "ERROR" for r in summary["runs"].values())
    return 1 if failed else 0


def consolidate(directory):
    root = Path(directory)
    rows = []
    statuses = []
    for study, (theory, cuf, nx, ns, paper) in STUDIES.items():
        available = {}
        for group, methods in GROUPS.items():
            for method in methods:
                path = root / study / group / (method + "_RESULT.json")
                if path.exists():
                    available[method] = read(path)
                else:
                    available[method] = {"status": "NOT_RETURNED"}
                statuses.append({
                    "study": study, "method": method,
                    "status": available[method]["status"],
                    "message": available[method].get("error", ""),
                })
        for mode in range(12):
            row = {"study": study, "axial_m1_verified": False,
                   "mode_number": mode + 1}
            for method in ("MVI_1", "MVI_2", "MVI_3", "MVI_4",
                           "MVI_5", "MVI_6", "MVI_7", "MVI_8", "MKR_FDM"):
                rec = available[method]
                if rec.get("status") == "COMPUTED_AXIAL_M_UNVERIFIED":
                    row[method + "_Hz"] = rec["frequencies_hz"][mode]
                    row[method + "_circumferential_n"] = (
                        rec["circumferential_wavenumbers"][mode])
                else:
                    row[method + "_Hz"] = ""
                    row[method + "_circumferential_n"] = ""
            rows.append(row)
    dest = root / "consolidated"
    dest.mkdir(parents=True, exist_ok=True)
    for filename, data in (("full_12_mode_frequency_matrix.csv", rows),
                           ("run_status_matrix.csv", statuses)):
        with (dest / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (dest / "run_status_matrix.json").write_text(
        json.dumps(statuses, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print("[MVI-PRIMARY] CONSOLIDATED " + json.dumps(
        {"case_count": len(STUDIES), "rows": len(rows),
         "statuses": statuses}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "consolidate":
        raise SystemExit(consolidate(sys.argv[2]))
    if len(sys.argv) != 3:
        raise SystemExit("Usage: script.py STUDY GROUP | consolidate DIRECTORY")
    raise SystemExit(run_group(sys.argv[1], sys.argv[2]))
