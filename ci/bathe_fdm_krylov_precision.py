#!/usr/bin/env python3
"""Physical Bathe FDM Krylov-accuracy diagnosis against archived MVI1/FDM.

Only the FDM linear Krylov relative tolerance changes. This does not
calculate a substitute MVI field, alter physics, or loosen acceptance.
"""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

from generate_linear_bathe_case import make_case
from compare_bathe_linear_all_theories import PHYSICAL_KEYS, nmechanical

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "config/verification/bathe_fdm_precision"
ARCHIVE = ROOT / "bathe_precision_archived"
STEM = Path("config/verification/linear_bathe_all_theories")
THEORIES = ("FSDT", "REDDY", "CUF_TE2")
STAGES = (("GAMMA", "BATHE_SUBSTEP_0001_GAMMA.npz",
           (2 - math.sqrt(2)) * .002),
          ("END", "REFERENCE_DYNAMIC_STEP_0001.npz", .002))
FIELDS = ("mechanical", "velocity", "acceleration")


def parameters():
    theory = os.environ["THEORY"]
    if theory not in THEORIES:
        raise ValueError("ONLY_THE_ORIGINAL_THREE_DIAGNOSTIC_THEORIES")
    raw = os.environ["KRYLOV_RTOL"]
    rtol = float(raw)
    if rtol not in (.001, .0001):
        raise ValueError("PRECISION_VARIANT_MUST_BE_DECLARED")
    return theory, rtol, f"{theory}_FDM_TIGHT_{raw.replace('.', 'p')}"


def generate():
    theory, rtol, name = parameters()
    case = make_case(theory, "FDM", None, 1, .002, name)
    case["reference_newton"]["krylov_rtol"] = rtol
    HERE.mkdir(parents=True, exist_ok=True)
    path = HERE / (name + ".json")
    if path.exists():
        raise ValueError("DIAGNOSTIC_CASE_OVERWRITE_REFUSED")
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    print("BATHE_FDM_TIGHT_REAL_CASE", path, rtol, flush=True)


def load_case(theory, kind, name):
    if kind == "TIGHT":
        folder = HERE
    else:
        folder = ARCHIVE / f"bathe-linear-{theory}-{kind}" / STEM
    case_file = folder / (name + ".json")
    output = folder / "output" / name
    case = json.loads(case_file.read_text(encoding="utf-8"))
    summary = json.loads((output / "FINAL_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    if (summary["theory"] != theory or summary["integrator"] != "BATHE"
            or summary["accepted_steps"] != 1 or summary["requested_steps"] != 1
            or summary["status"] != "CONVERGED"
            or summary["method"] != ("MVI" if kind == "MVI" else "FDM")):
        raise ValueError("BAD_OR_FAILED_INDEPENDENT_PHYSICAL_CASE:" + kind)
    return case, output


def read_stage(folder, stage, name, expected, theory, *, is_mvi=False):
    with np.load(folder / name, allow_pickle=False) as z:
        t = float(z["time_s"])
        if (abs(t - expected) > 1.e-12
                or abs(float(z["load_factor"]) - math.sin(50 * math.pi * t)) > 1.e-12):
            raise ValueError("BAD_BATHE_PHYSICAL_TIME_OR_LOAD:" + stage)
        fields = {k: np.array(z[k], dtype=float) for k in FIELDS}
        if any(v.shape != (nmechanical(theory), 9, 9)
                   or not np.isfinite(v).all() for v in fields.values()):
            raise ValueError("BAD_BATHE_PHYSICAL_FIELD:" + stage)
        if is_mvi:
            reconstructed = (np.einsum("crx,crs->cxs",
                                     z["x_profiles"], z["y_profiles"])
                             * z["state_scales"])
            if np.max(abs(reconstructed - fields["mechanical"])) > 1.e-11:
                raise ValueError("BAD_FIXED_RANK_MVI_RECONSTRUCTION:" + stage)
    return fields


def report():
    theory, rtol, name = parameters()
    cases = {
        "FDM": load_case(theory, "FDM", theory + "_FDM"),
        "MVI": load_case(theory, "MVI", theory + "_MVI1"),
        "TIGHT": load_case(theory, "TIGHT", name),
    }
    baseline = cases["FDM"][0]
    for kind, (case, folder) in cases.items():
        for key in PHYSICAL_KEYS:
            if case[key] != baseline[key]:
                raise ValueError("PHYSICAL_PASSPORT_DIFFERENCE:" + kind + ":" + key)
        with np.load(folder / "REFERENCE_DYNAMIC_INITIAL.npz",
                     allow_pickle=False) as initial:
            if abs(float(initial["time_s"])) > 1.e-12:
                raise ValueError("INITIAL_TIME_NOT_ZERO:" + kind)
    if cases["TIGHT"][0]["fdm_grid"] != baseline["fdm_grid"]:
        raise ValueError("SPATIAL_OPERATOR_CHANGED_IN_PRECISION_TEST")
    exact_settings = dict(baseline["reference_newton"])
    exact_settings["krylov_rtol"] = rtol
    if cases["TIGHT"][0]["reference_newton"] != exact_settings:
        raise ValueError("NON_KRYLOV_PARAMETER_CHANGED")
    for field in FIELDS:
        originals = []
        for _, folder in cases.values():
            with np.load(folder / "REFERENCE_DYNAMIC_INITIAL.npz",
                         allow_pickle=False) as initial:
                originals.append(np.array(initial[field]))
        if not all(np.array_equal(originals[0], v) for v in originals[1:]):
            raise ValueError("DIFFERENT_INITIAL_PHYSICAL_FIELDS:" + field)
    rows = []
    for stage, file, expected_time in STAGES:
        states = {kind: read_stage(folder, stage, file, expected_time, theory,
                                   is_mvi=(kind == "MVI"))
                  for kind, (_, folder) in cases.items()}
        for field in FIELDS:
            reference = states["FDM"][field]
            mvi = states["MVI"][field]
            tight = states["TIGHT"][field]
            global_norm = float(np.linalg.norm(reference))
            for component in range(nmechanical(theory)):
                ref = reference[component]
                trial = mvi[component]
                improved = tight[component]
                scale = float(np.linalg.norm(ref))
                significant = bool(scale > 1.e-8 * global_norm)
                original_error = float(np.linalg.norm(trial - ref))
                new_error = float(np.linalg.norm(trial - improved))
                fdm_shift = float(np.linalg.norm(improved - ref))
                pointwise = abs(trial - improved)
                index = np.unravel_index(int(np.argmax(pointwise)), pointwise.shape)
                row = {
                    "theory": theory, "stage": stage, "field": field,
                    "component": component, "krylov_rtol": rtol,
                    "significant_component": significant,
                    "reference_peak": float(np.max(abs(ref))),
                    "original_mvi_fdm_norm_difference": original_error,
                    "tight_mvi_fdm_norm_difference": new_error,
                    "tight_fdm_shift_norm_difference": fdm_shift,
                    "original_mvi_fdm_percent": (
                        100 * original_error / scale if significant else None),
                    "tight_mvi_fdm_percent": (
                        100 * new_error / scale if significant else None),
                    "tight_fdm_shift_percent": (
                        100 * fdm_shift / scale if significant else None),
                    "maximum_absolute_mvi_tight_difference": float(pointwise[index]),
                    "node_x": int(index[0]), "node_s": int(index[1]),
                }
                rows.append(row)
                print("BATHE_PRECISION_COMPONENT", row, flush=True)
    out = ROOT / "bathe_precision_report"
    out.mkdir(parents=True, exist_ok=True)
    target = out / (name + "_component_fields")
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    target.with_suffix(".json").write_text(
        json.dumps({
            "status": "PHYSICAL_FIELDS_COMPARED_NOT_DECLARED_EQUAL",
            "source_run": 41645787449, "theory": theory, "krylov_rtol": rtol,
            "changed_physical_equations": False,
            "changed_numerical_parameter": "reference_newton.krylov_rtol",
            "component_comparisons": len(rows), "data": rows,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("BATHE_KRYLOV_DIAGNOSTIC_COMPLETE", theory, rtol, len(rows), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("generate", "report"):
        raise SystemExit("usage: bathe_fdm_krylov_precision.py generate|report")
    (generate if sys.argv[1] == "generate" else report)()
