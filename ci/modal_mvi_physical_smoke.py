#!/usr/bin/env python3
"""Small REAL shell modal smoke: MKR/FDM versus unrestricted fixed-rank MVI.

This is not a literature benchmark or proof of full-spectrum completeness.
No physical matrices/BC/kinematics are modified for the comparison.
"""
from __future__ import annotations

import copy
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case, _component_separation_ranks


HERE = Path(__file__).resolve().parents[1]
REPORT = HERE / "modal_mvi_physical_smoke_result.json"


def reference_case():
    base = json.loads(
        (HERE / "config" / "CASE_REFERENCE_MODAL_SEVEN_THEORIES.json")
        .read_text(encoding="utf-8"))
    base["theory"] = "KL"
    base.pop("cuf_model", None)
    base["geometry_model"] = "LINEAR"
    base["quadrature"]["n_gauss"] = 4
    base["modal"] = {
        "density_kg_m3": 7850.0,
        "n_modes": 1,
        "eigen_tolerance": 1.0e-7,
        "mode_tolerance": 1.0e-5}
    base["reference_newton"]["max_free_dofs"] = 512
    base["reference_newton"]["tangent_step_scaled"] = 2.0e-6
    base["mvi_grid"] = {
        "nx": 9, "ns": 9, "operator_scheme": "SBP42_FOURIER_ODD"}
    base.pop("fdm_grid", None)
    base["method"] = "MVI"
    base["rank"] = 1
    base["fdm_backend"] = "DENSE_REFERENCE"
    base["reference_mvi"] = {"max_sweeps": 25}
    return base


def main():
    base = reference_case()
    report = {
        "scope": "KL_LINEAR_MACRO_CC_SBP42_9x9_unloaded",
        "physical_model": {
            "geometry": base["geometry"],
            "material": base["material"],
            "density_kg_m3": base["modal"]["density_kg_m3"],
            "quadrature": base["quadrature"],
            "bc": base["boundary_conditions"],
            "grid": base["mvi_grid"]},
        "runs": {},
        "frequency_comparison": {},
        "all_three_completed": False}
    try:
        for name, method, rank in (
                ("MVI_1", "MVI", 1),
                ("MVI_2", "MVI", 2),
                ("MKR_FDM", "FDM", None)):
            case = copy.deepcopy(base)
            case["method"] = method
            if method == "FDM":
                case["fdm_grid"] = case.pop("mvi_grid")
                case.pop("rank", None)
                case.pop("reference_mvi", None)
                case["modal"].pop("mode_tolerance", None)
            else:
                case["rank"] = rank
            started = time.monotonic()
            print(f"[PHYSICAL MODAL] BEGIN {name}", flush=True)
            try:
                result = solve_modal_case(case)
                frequencies = np.asarray(result["frequencies_hz"], dtype=float)
                mechanical_modes = np.asarray(result["mechanical_modes"], dtype=float)
                if (frequencies.shape != (1,) or not np.isfinite(frequencies).all()
                        or np.any(frequencies <= 0) or
                        not np.isfinite(mechanical_modes).all()):
                    raise ValueError("PHYSICAL_MODAL_NONFINITE_OR_NONPOSITIVE_OUTPUT")
                data = {
                    "status": "SOLVED",
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "frequency_hz": float(frequencies[0]),
                    "actual_per_component_shape_ranks":
                        _component_separation_ranks(mechanical_modes)[0],
                    "mvi_final_xy_trial_dimensions":
                        result.get("mvi_final_xy_trial_dimensions"),
                    "global_mode_ranks": result.get("global_mode_ranks"),
                    "mvi_sweeps": result.get("mvi_sweeps"),
                    "global_spectrum_completeness_verified":
                        bool(result.get("global_spectrum_completeness_verified", False)),
                    "method": result.get("method", method),
                    "rank": result.get("rank", None)}
                report["runs"][name] = data
                print("[PHYSICAL MODAL] RESULT " + name + " " +
                      json.dumps(data, ensure_ascii=False, allow_nan=False),
                      flush=True)
            except Exception as exc:
                report["runs"][name] = {
                    "status": "FAILED",
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8)}
                print(f"[PHYSICAL MODAL] FAILED {name}: {exc}", flush=True)
                continue

        mk = report["runs"].get("MKR_FDM", {})
        if mk.get("status") == "SOLVED":
            fref = mk["frequency_hz"]
            for name in ("MVI_1", "MVI_2"):
                trial = report["runs"].get(name, {})
                if trial.get("status") == "SOLVED":
                    report["frequency_comparison"][name] = {
                        "relative_difference_percent":
                            100.0 * abs(trial["frequency_hz"] - fref) / fref}
        m1 = report["runs"].get("MVI_1", {})
        m2 = report["runs"].get("MVI_2", {})
        if m1.get("status") == m2.get("status") == "SOLVED":
            report["frequency_comparison"]["rank_1_vs_2"] = {
                "relative_difference_percent":
                    100.0 * abs(m1["frequency_hz"] - m2["frequency_hz"]) /
                    m2["frequency_hz"]}
            s1 = m1.get("mvi_final_xy_trial_dimensions") or [{}]
            s2 = m2.get("mvi_final_xy_trial_dimensions") or [{}]
            if (not s1[0].get("X") or not s2[0].get("X") or
                    s2[0]["X"]["mechanical_directions"] <=
                    s1[0]["X"]["mechanical_directions"]):
                raise ValueError(
                    "MVI_RANK_2_TRIAL_DIMENSION_NOT_GREATER_THAN_RANK_1")
        report["all_three_completed"] = all(
            report["runs"].get(n, {}).get("status") == "SOLVED"
            for n in ("MVI_1", "MVI_2", "MKR_FDM"))
    except Exception as exc:
        report["harness_error"] = traceback.format_exc(limit=10)
        print(f"[PHYSICAL MODAL] HARNESS FAILURE {exc}", flush=True)
    finally:
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        print(f"[PHYSICAL MODAL] REPORT {REPORT}", flush=True)
        print(f"[PHYSICAL MODAL] ALL_COMPLETED={report['all_three_completed']}",
              flush=True)
    return 0 if report["all_three_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
