#!/usr/bin/env python3
"""Inspect nine independent KL C-C Newmark trajectories; never solve a PDE.

FDM is an on-grid numerical control, not a 3D or analytic benchmark.
Unaccepted or missing step states cannot enter the comparison.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "config" / "verification" / "kl_cc_dynamic"
OUT = CASES / "output"
LABELS = ["MKR_FDM"] + [f"MVI{i}" for i in range(1, 9)]
STEM = "KL_CC_9x9_HARMONIC25HZ_"
FIELDS = ("mechanical", "velocity", "acceleration")


def read_case(label):
    base = OUT / (STEM + label)
    report = base / "FINAL_RUN_SUMMARY.json"
    if not report.is_file():
        return None, [], "MISSING_SUMMARY"
    try:
        s = json.loads(report.read_text(encoding="utf-8"))
        if s.get("calculation_kind") != "DYNAMIC" or s.get("theory") != "KL":
            return s, [], "WRONG_PHYSICS"
        if s.get("method") != ("FDM" if label == "MKR_FDM" else "MVI"):
            return s, [], "METHOD_MISMATCH"
        steps = s.get("steps", [])
        if not s.get("converged") or s.get("accepted_steps") != 2 or len(steps) != 2:
            return s, [], "NOT_ALL_STEPS_ACCEPTED"
        states = []
        for i, item in enumerate(steps, 1):
            name = item.get("state_file", "")
            if name != f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz":
                return s, [], "UNACCEPTED_STATE"
            if item.get("converged") is not True:
                return s, [], "UNACCEPTED_STEP"
            with np.load(base / name, allow_pickle=False) as z:
                data = {field: np.array(z[field], dtype=float) for field in FIELDS}
                data["time_s"] = float(z["time_s"])
                if label != "MKR_FDM":
                    data["x_profiles"] = np.array(z["x_profiles"], dtype=float)
                    data["y_profiles"] = np.array(z["y_profiles"], dtype=float)
                    data["state_scales"] = np.array(z["state_scales"], dtype=float)
            if any(data[f].shape != (3, 9, 9) or not np.isfinite(data[f]).all()
                   for f in FIELDS):
                return s, [], "WRONG_SHAPE_OR_NONFINITE"
            if abs(data["time_s"] - 0.001 * i) > 1e-12:
                return s, [], "TIME_GRID_MISMATCH"
            if label != "MKR_FDM":
                reconstructed = np.einsum("crx,crs->cxs",
                                          data["x_profiles"], data["y_profiles"])
                reconstructed = data["state_scales"] * reconstructed
                if not np.allclose(reconstructed, data["mechanical"], atol=1e-11,
                                   rtol=1e-9):
                    return s, [], "MVI_RECONSTRUCTION_FAILED"
            states.append(data)
        return s, states, "ACCEPTED"
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        return None, [], "INVALID_STATE:" + str(exc)


def main():
    data = {label: read_case(label) for label in LABELS}
    control = data["MKR_FDM"][1]
    rows = []
    for label in LABELS:
        summary, states, status = data[label]
        for i in (0, 1):
            field = states[i] if i < len(states) else None
            row = dict(case=label, time_s=0.001*(i+1), status=status,
                       accepted_steps=summary.get("accepted_steps", 0) if summary else 0,
                       w_midspan_mean_m="", max_abs_w_m="",
                       max_abs_velocity_m_per_s="", max_abs_acceleration_m_per_s2="",
                       max_clamped_displacement_m="",
                       displacement_difference_percent="",
                       velocity_difference_percent="",
                       acceleration_difference_percent="")
            if field is not None:
                q = field["mechanical"]
                row.update(w_midspan_mean_m=float(np.mean(q[2, 4])),
                           max_abs_w_m=float(np.max(np.abs(q[2]))),
                           max_abs_velocity_m_per_s=float(np.max(np.abs(field["velocity"]))),
                           max_abs_acceleration_m_per_s2=float(np.max(np.abs(field["acceleration"]))),
                           max_clamped_displacement_m=float(np.max(np.abs(q[:, [0, -1]]))))
                if control:
                    for name, key in ((FIELDS[0], "displacement"),
                                      (FIELDS[1], "velocity"),
                                      (FIELDS[2], "acceleration")):
                        ref, got = control[i][name], field[name]
                        den = float(np.linalg.norm(ref))
                        row[key + "_difference_percent"] = (
                            float(100.0 * np.linalg.norm(got - ref) / den)
                            if den > 0.0 else (0.0 if np.linalg.norm(got) == 0.0 else "UNDEFINED"))
            rows.append(row)
    csv_path = CASES / "KL_CC_NEWMARK_COMPARE.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = all(data[label][2] == "ACCEPTED" for label in LABELS)
    manifest = {"requested_methods": 9, "accepted_methods": sum(
        data[l][2] == "ACCEPTED" for l in LABELS),
        "requested_steps_per_method": 2,
        "all_accepted": complete,
        "method_status": {l: data[l][2] for l in LABELS},
        "independent_validation_verified": False,
        "time_step_convergence_verified": False,
        "mesh_convergence_verified": False}
    (CASES / "KL_CC_NEWMARK_COMPARE.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for label in LABELS:
        print(label, data[label][2], flush=True)
    print("ACCEPTED", manifest["accepted_methods"], "/", 9)
    if not complete:
        raise SystemExit("KL_CC_DYNAMIC_INCOMPLETE: one or more methods were not accepted")
    for row in rows:
        if row["max_clamped_displacement_m"] > 1e-9:
            raise SystemExit("KL_CC_DYNAMIC_BC_FAILED: " + row["case"])
    print("ALL_NINE_DYNAMIC_METHODS_ACCEPTED_2_STEPS")


if __name__ == "__main__":
    main()
