#!/usr/bin/env python3
"""Strict nodewise MVI-1 vs independent FDM/MKR audit of archived Bathe states.

Analysis only: does not execute any plate/shell solver. Rank is never changed,
physical fields are never synthesized, and no failed component is averaged away.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

import compare_bathe_linear_all_theories as base

# Fixed diagnostic acceptance limits, chosen before evaluating the archives.
# For EVERY node, |MVI-FDM| <= atol + rtol*max(|MVI|,|FDM|).
# A local absolute floor avoids meaningless relative percentages at zero.
TOLERANCES = {
    "mechanical": (1e-12, 1e-4),
    "velocity": (1e-9, 1e-4),
    "acceleration": (1e-6, 1e-4),
}
REPORT_STEM = "BATHE_SEVEN_THEORIES_STRICT_COMPONENT_FIELDS"
SOURCE_RUN = base.SOURCE_RUN


def component_names(theory):
    if theory == "KL":
        return ("u", "v", "w")
    if theory in ("FSDT", "REDDY"):
        return ("u", "v", "w", "rotation_x", "rotation_s")
    p = int(theory[-1])
    return tuple(f"{axis}_{k}" for axis in ("U", "V", "W")
                 for k in range(p + 1))


def initial_fields(theory, method):
    name = f"{theory}_{method}{'1' if method == 'MVI' else ''}"
    path = base.BASE / "output" / name / "REFERENCE_DYNAMIC_INITIAL.npz"
    if not path.is_file():
        raise ValueError(f"MISSING_INDEPENDENT_INITIAL_STATE:{name}")
    with np.load(path, allow_pickle=False) as data:
        fields = {key: np.array(data[key], dtype=float)
                  for key in TOLERANCES}
        t = float(data["time_s"])
    expected = (base.nmechanical(theory), 9, 9)
    if any(a.shape != expected or not np.isfinite(a).all()
           for a in fields.values()) or abs(t) > 1e-12:
        raise ValueError(f"INVALID_PHYSICAL_INITIAL_STATE:{name}")
    return fields


def write_reports(rows, theory_summary, failures):
    destination = base.BASE
    destination.mkdir(parents=True, exist_ok=True)
    csv_file = destination / (REPORT_STEM + ".csv")
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_file = destination / (REPORT_STEM + ".json")
    json_file.write_text(json.dumps({
        "source_run_id": SOURCE_RUN,
        "compared": "seven theories, two physical Bathe substeps, every DOF component and node",
        "status": "MISMATCH" if failures else "PASS",
        "total_component_field_comparisons": len(rows),
        "failed_component_field_comparisons": len(failures),
        "nodewise_acceptance_rule": "|MVI-FDM| <= atol + rtol*max(|MVI|,|FDM|)",
        "tolerances": {k: {"atol": v[0], "rtol": v[1]}
                       for k, v in TOLERANCES.items()},
        "theory_stage_summary": theory_summary,
        "failed_components": failures,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = [
        "# Bathe MVI-1 versus independent FDM (MKR): full-field audit",
        "",
        f"Source GitHub Actions run: {SOURCE_RUN}. Only saved physical states were read.",
        "No shell calculations were run by this comparison.",
        "",
        "Nodewise gate for every component: abs(MVI-FDM) <= atol + rtol*max(abs(MVI),abs(FDM)).",
        "Small nearly-zero components are recorded separately; they are not hidden by whole-field norms.",
        "",
        "| Theory | Stage | Failed component fields | Maximum acceleration difference |",
        "| --- | --- | ---: | ---: |",
    ]
    for s in theory_summary:
        markdown.append(
            f"| {s['theory']} | {s['stage']} | {s['failed_component_fields']} "
            f"| {s['max_abs_acceleration']:.9g} |")
    markdown += [
        "",
        "A mismatch is a numerical difference, not by itself proof of an equation bug.",
        "MVI rank 1 and full-nodal Krylov FDM are distinct approximations.",
        "",
    ]
    (destination / (REPORT_STEM + ".md")).write_text(
        "\n".join(markdown) + "\n", encoding="utf-8")
    print("BATHE_COMPONENT_REPORT_FILES", csv_file, json_file, flush=True)


def main():
    rows = []
    theory_summary = []
    failures = []
    for theory in base.THEORIES:
        cases = {method: base.read_case(theory, method)
                 for method in ("FDM", "MVI")}
        for key in base.PHYSICAL_KEYS:
            if cases["MVI"][0][key] != cases["FDM"][0][key]:
                raise ValueError(f"PHYSICAL_PASSPORT_MISMATCH:{theory}:{key}")
        if cases["MVI"][1]["fdm_backend"] != "DENSE_REFERENCE" or (
            cases["FDM"][1]["fdm_backend"] != "MATRIX_FREE_GMRES"):
            raise ValueError(f"INDEPENDENT_METHOD_BACKENDS_CHANGED:{theory}")
        initial_mvi = initial_fields(theory, "MVI")
        initial_fdm = initial_fields(theory, "FDM")
        for field in TOLERANCES:
            if not np.array_equal(initial_mvi[field], initial_fdm[field]):
                raise ValueError(f"INITIAL_STATE_DIFFERENCE:{theory}:{field}")
        names = component_names(theory)
        if len(names) != base.nmechanical(theory):
            raise ValueError(f"COMPONENT_LAYOUT_ERROR:{theory}")
        for stage in ("GAMMA", "END"):
            tm, mvi = cases["MVI"][2][stage]
            tf, fdm = cases["FDM"][2][stage]
            if abs(tm - tf) > 1e-12:
                raise ValueError(f"PHYSICAL_SUBSTEP_TIME_DIFFERENCE:{theory}:{stage}")
            failed_here = 0
            max_acceleration_difference = 0.
            for field, (atol, rtol) in TOLERANCES.items():
                qm = mvi[field]
                qf = fdm[field]
                for c, name in enumerate(names):
                    delta = np.abs(qm[c] - qf[c])
                    tolerance = atol + rtol * np.maximum(
                        np.abs(qm[c]), np.abs(qf[c]))
                    violations = delta > tolerance
                    location = np.unravel_index(int(np.argmax(delta)), delta.shape)
                    worst = np.unravel_index(
                        int(np.argmax(delta / tolerance)), delta.shape)
                    max_abs = float(delta[location])
                    scale = float(max(np.max(np.abs(qm[c])),
                                      np.max(np.abs(qf[c]))))
                    count = int(np.count_nonzero(violations))
                    row = {
                        "theory": theory, "stage": stage, "time_s": float(tm),
                        "field": field, "component_index": c,
                        "component": name, "grid_nx": qm.shape[1],
                        "grid_ns": qm.shape[2],
                        "maximum_absolute_difference": max_abs,
                        "component_peak_abs": scale,
                        "component_peak_relative_difference_percent": (
                            100. * max_abs / scale if scale else 0.),
                        "max_difference_x_index": int(location[0]),
                        "max_difference_s_index": int(location[1]),
                        "mvi_at_max_difference": float(qm[c][location]),
                        "fdm_at_max_difference": float(qf[c][location]),
                        "failed_nodes": count,
                        "most_violated_x_index": int(worst[0]),
                        "most_violated_s_index": int(worst[1]),
                        "worst_ratio_to_allowed_difference": float(
                            (delta / tolerance)[worst]),
                        "absolute_tolerance": atol,
                        "relative_tolerance": rtol,
                        "verdict": "MISMATCH" if count else "PASS",
                    }
                    rows.append(row)
                    if count:
                        failed_here += 1
                        failures.append(row)
                    if field == "acceleration":
                        max_acceleration_difference = max(
                            max_acceleration_difference, max_abs)
            theory_summary.append({
                "theory": theory, "stage": stage,
                "time_s": float(tm),
                "component_fields_compared": len(names) * len(TOLERANCES),
                "failed_component_fields": failed_here,
                "max_abs_acceleration": max_acceleration_difference,
            })
            print("BATHE_STRICT_THEORY_STAGE", theory, stage,
                  "failed_component_fields", failed_here,
                  "max_abs_acceleration", max_acceleration_difference,
                  flush=True)
    write_reports(rows, theory_summary, failures)

    # Preserve the existing independent whole-field gate as an additional,
    # not replacement, check. The detailed reports are written before it runs.
    old_error = None
    try:
        base.main()
    except Exception as exc:
        old_error = str(exc)
        print("BATHE_ORIGINAL_WHOLE_FIELD_GATE_FAILED", old_error, flush=True)
    if failures or old_error is not None:
        raise ValueError(
            f"BATHE_SEVEN_THEORY_FULL_FIELD_MISMATCH: "
            f"{len(failures)} component fields failed. "
            f"Original gate: {old_error or 'PASS'}.")
    print("BATHE_SEVEN_THEORIES_ALL_COMPONENTS_AND_NODES_AGREE", flush=True)


if __name__ == "__main__":
    main()
