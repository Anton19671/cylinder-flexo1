#!/usr/bin/env python3
"""KL C-C pressure benchmark: preflight by default; local solve only with --run.

The primary matrix is 8 independent MVI ranks + 1 FDM/MKR on the exact same
grid/operator, plus FDM and MVI1 at half pressure to check linearity.
No result is fabricated and no failure is replaced by another method.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "config/verification/kl_cc_uniform_static"


def inputs():
    primary = [DIR / "KL_CC_9x9_100KPA_MKR_FDM.json"]
    primary += [DIR / f"KL_CC_9x9_100KPA_MVI{n}.json" for n in range(1, 9)]
    controls = [DIR / "KL_CC_9x9_50KPA_MKR_FDM.json",
                DIR / "KL_CC_9x9_50KPA_MVI1.json"]
    return primary + controls


def case_data(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    return case, (path.parent / case["output_dir"]).resolve()


def preflight():
    from cylinder_paths import activate
    activate()
    from cylinder_reference_case import prepare_reference_case
    for path in inputs():
        case, output = case_data(path)
        prepared = prepare_reference_case(case, base_dir=path.parent)
        assert prepared.calculation_kind == "STATIC"
        assert prepared.problem.functional.theory == "KL"
        assert prepared.electrical is None
        assert case["load_definition"]["kinematics"] == "DEAD_REFERENCE"
        assert case["load_definition"]["face"] == "MIDDLE"
        assert case["load_definition"]["direction"] == "RADIAL_INWARD"
        assert case["geometry_model"] == "LINEAR"
        assert case["fdm_grid" if case["method"] == "FDM" else "mvi_grid"] == {
            "nx": 9, "ns": 9, "operator_scheme": "SBP42_FOURIER_ODD"}
        assert not output == DIR
        print("PREFLIGHT_OK", path.name, "method", prepared.method)
    print("Prepared", len(inputs()), "independent CASEs; no PDE solve started.")


def run_local():
    for path in inputs():
        case, output = case_data(path)
        if output.exists() and any(output.iterdir()):
            print("SKIP_EXISTING", path.name, output)
            continue
        print("RUN", path.name, flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / "RUN_FINAL_CASE.py"),
                               str(path)], cwd=ROOT, check=False)
        if proc.returncode != 0:
            print("RUN_FAILED", path.name, "exit", proc.returncode, flush=True)


def load_row(path):
    case, output = case_data(path)
    row = {"case": path.stem, "pressure_pa": case["load_definition"]["q"],
           "method": case["method"], "rank": case.get("rank", ""),
           "status": "NOT_RUN", "accepted": False,
           "mid_w_mean_m": "", "max_abs_w_m": "",
           "theta_spread_mid_w_m": "", "clamp_abs_max_m": "",
           "field_delta_from_fdm_percent": ""}
    state = None
    summary_path = output / "FINAL_RUN_SUMMARY.json"
    if not summary_path.is_file():
        return row, state
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row["status"] = summary.get("status", "UNKNOWN")
        steps = summary.get("steps", [])
        if not summary.get("converged") or len(steps) != 1 or not steps[0].get("converged"):
            return row, state
        filename = steps[0]["state_file"]
        if not filename.startswith("REFERENCE_STEP_"):
            raise ValueError("unaccepted or unverifiable state file")
        with np.load(output / filename, allow_pickle=False) as result:
            state = np.array(result["mechanical"], dtype=float, copy=True)
        if state.shape != (3, 9, 9) or not np.isfinite(state).all():
            raise ValueError("wrong/nonfinite displacement array")
        w = state[2]
        row.update(accepted=True, status=steps[0]["status"],
                   mid_w_mean_m=float(np.mean(w[4])),
                   max_abs_w_m=float(np.max(np.abs(w))),
                   theta_spread_mid_w_m=float(np.ptp(w[4])),
                   clamp_abs_max_m=float(np.max(np.abs(state[:, [0, -1], :]))))
    except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
        row.update(status="INVALID_RESULT:" + str(exc), accepted=False)
        state = None
    return row, state


def report():
    data = [load_row(path) for path in inputs()]
    fdm = next(((row, state) for row, state in data
                if row["method"] == "FDM" and row["pressure_pa"] == 100000
                and state is not None), None)
    for row, state in data:
        if (fdm and state is not None and row["pressure_pa"] == 100000
                and row["method"] == "MVI"):
            denominator = np.linalg.norm(fdm[1])
            if denominator > 0:
                row["field_delta_from_fdm_percent"] = float(
                    100.0 * np.linalg.norm(state - fdm[1]) / denominator)
    rows = [row for row, _ in data]
    path = DIR / "KL_CC_STATIC_VERIFICATION.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scaling = {}
    for method, rank in (("FDM", ""), ("MVI", 1)):
        candidates = {r["pressure_pa"]: r for r in rows
                      if r["method"] == method and r["rank"] == rank and r["accepted"]}
        if 100000 in candidates and 50000 in candidates:
            full, half = candidates[100000]["mid_w_mean_m"], candidates[50000]["mid_w_mean_m"]
            scaling[method] = (None if abs(half) == 0 else float(full / half))
    meta = {"benchmark": "KL C-C uniform inward midsurface dead load",
            "grid": [9, 9], "pressure_pa": 100000.0,
            "accepted": sum(bool(r["accepted"]) for r in rows),
            "requested": len(rows), "halving_pressure_w_ratio_expected": 2.0,
            "measured_w_ratio": scaling,
            "mesh_convergence_verified": False,
            "independent_3d_validation_verified": False,
            "note": "FDM is only an on-grid numerical control, not an exact reference."}
    (DIR / "KL_CC_STATIC_VERIFICATION.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Report:", path, "accepted", meta["accepted"], "/", meta["requested"])
    for row in rows:
        print(row["case"], row["status"])
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="opt-in physical solves on this computer")
    parser.add_argument("--report", action="store_true", help="read existing results; never solve")
    args = parser.parse_args()
    if args.run or not args.report:
        preflight()
    if args.run:
        run_local()
    if args.report or args.run:
        report()


if __name__ == "__main__":
    main()
