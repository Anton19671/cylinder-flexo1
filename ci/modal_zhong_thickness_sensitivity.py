#!/usr/bin/env python3
"""Zhong/Rawat thickness-sensitivity study: fixed geometry and material.

This is a numerical *model-sensitivity* experiment, not external Fluegge
validation. The original Rawat h/R=0.05 remains one of three cases.
CUF TE1 and TE4 are solved separately with their physical FDM modal
operators, unchanged K/M physics, supports, and solver tolerance.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/rawat2016/modal_linear"
THICKNESS_RATIOS = (0.025, 0.05, 0.10)
HARMONICS = (1, 2, 3, 4, 5, 6, 7, 8)


def run(theory, grid):
    if theory not in ("CUF_TE1", "CUF_TE4") or grid not in (17, 21, 25):
        raise ValueError("THICKNESS_STUDY_INVALID_CASE")
    path = BASE / f"RAWAT2016_{theory}_17x17_MODAL_FDM.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    if source["method"] != "FDM" or source["cuf_model"] != theory[-3:]:
        raise ValueError("THICKNESS_STUDY_NOT_A_PHYSICAL_FDM_CUF_CASE")
    report = {
        "study": "ZHONG_THICKNESS_H_OVER_R",
        "theory": theory,
        "grid": [grid, grid],
        "thickness_ratios": list(THICKNESS_RATIOS),
        "material": source["material"],
        "geometry_L_m": source["geometry"]["L"],
        "geometry_R_m": source["geometry"]["R"],
        "operator_scheme": "SBP42_FOURIER_ODD_CSR",
        "mechanical_boundary_conditions": source["boundary_conditions"]["variational"]["mechanical"],
        "status": "INCOMPLETE",
        "rawat_h_over_R": 0.05,
        "independent_fluegge_frequencies_available": False,
        "independent_ANSYS_modes_available": False,
        "axial_m1_verified": False,
        "runs": [],
    }
    output = ROOT / f"zhong_thickness_{theory}_{grid}x{grid}.json"
    for ratio in THICKNESS_RATIOS:
        case = copy.deepcopy(source)
        case["geometry"]["h"] = float(ratio * case["geometry"]["R"])
        case["fdm_grid"] = {
            "nx": grid, "ns": grid, "operator_scheme": "SBP42_FOURIER_ODD_CSR"
        }
        case["modal"]["n_modes"] = 12
        case["output_dir"] = f"OUTPUT_ZHONG_THICKNESS_{theory}_{grid}_{ratio}"
        row = {"h_over_R": ratio, "status": "ERROR"}
        started = time.monotonic()
        try:
            solved = solve_modal_case(case)
            families = solved.get("harmonic_family_scan")
            if not isinstance(families, list):
                raise ValueError("THICKNESS_NO_PHYSICAL_FOURIER_SECTOR_SCAN")
            rows = {}
            for n in HARMONICS:
                sector = [item for item in families
                          if int(item["n"]) == n
                          and int(item["axial_mode_index"]) == 1]
                if len(sector) != 1:
                    raise ValueError(f"THICKNESS_MISSING_OR_DUPLICATE_N_{n}")
                rows[str(n)] = {
                    "frequency_hz": float(sector[0]["frequency_hz"]),
                    "axial_index_verified": bool(
                        sector[0].get("axial_mode_index_verified", False)),
                }
            row.update({
                "status": "COMPUTED_AXIAL_BRANCH_NOT_INDEPENDENTLY_IDENTIFIED",
                "by_n": rows,
                "spectrum_completeness_verified": bool(
                    solved.get("global_spectrum_completeness_verified", False)),
                "first_twelve_frequencies_hz": [
                    float(x) for x in solved["frequencies_hz"]],
            })
            np.savez_compressed(
                ROOT / f"zhong_thickness_{theory}_{grid}_{ratio:.3f}_modes.npz",
                frequencies_hz=np.asarray(solved["frequencies_hz"], float),
                modes=np.asarray(solved["mechanical_modes"], float))
        except Exception as error:
            row["error"] = repr(error)
        row["elapsed_s"] = round(time.monotonic() - started, 3)
        report["runs"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("[ZHONG-THICKNESS] " + json.dumps({
            "theory": theory, "grid": grid, **row}, ensure_ascii=False), flush=True)
    good = all(r["status"] == "COMPUTED_AXIAL_BRANCH_NOT_INDEPENDENTLY_IDENTIFIED"
               for r in report["runs"])
    if good:
        report["relative_to_rawat_h0p05_percent"] = {
            str(n): {
                f"{r['h_over_R']:.3f}": 100. * (
                    r["by_n"][str(n)]["frequency_hz"] /
                    report["runs"][1]["by_n"][str(n)]["frequency_hz"] - 1.)
                for r in report["runs"]
            }
            for n in HARMONICS
        }
    report["status"] = "COMPUTED_DIAGNOSTIC_ONLY" if good else "INCOMPLETE"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if good else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("theory", choices=("CUF_TE1", "CUF_TE4"))
    parser.add_argument("--grid", type=int, choices=(17, 21, 25), default=21)
    args = parser.parse_args()
    raise SystemExit(run(args.theory, args.grid))
