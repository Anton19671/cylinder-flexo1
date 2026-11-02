#!/usr/bin/env python3
"""Certify the complete 7 theory x 8 rank modal matrix from REAL CI artifacts.

No results are inferred from green jobs, no expected modal values are invented,
and no eigenproblem runs here. A PASS requires FDM and MVI spectra plus the
mass-weighted principal subspace overlaps to have already passed their physical
comparison. Missing, malformed or stale audit artifacts are FAIL, not PASS.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

THEORIES = ("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
RANKS = tuple(range(1, 9))
MODES = 8
MINIMUM_MAC = 0.999


def verify_case(model, rank, audit):
    if (not isinstance(audit, dict) or audit.get("model") != model
            or audit.get("rank") != rank or audit.get("grid") != [13, 13]
            or audit.get("boundary") != "C-C"
            or audit.get("completed") is not True):
        raise ValueError("case identity or completed flag invalid")
    runs = audit.get("runs", {})
    comparison = audit.get("comparison")
    if not isinstance(runs, dict) or not isinstance(comparison, dict):
        raise ValueError("missing independent FDM/MVI physical comparison")
    def frequencies(name):
        record = runs.get(name)
        seq = record.get("frequencies_hz") if isinstance(record, dict) else None
        if not isinstance(seq, list) or len(seq) != MODES:
            raise ValueError(name + " missing eight frequencies")
        values = [float(f) for f in seq]
        if (not all(math.isfinite(f) and f > 0. for f in values)
                or any(values[j] < values[j-1] - 1.e-7
                       for j in range(1, MODES))):
            raise ValueError(name + " has nonfinite, nonpositive or unsorted spectrum")
        return values
    reference = frequencies("FDM")
    trial = frequencies("MVI_" + str(rank))
    if (comparison.get("status") != "PASS"
            or comparison.get("frequency_verified") is not True
            or comparison.get("all_mode_subspaces_verified") is not True
            or comparison.get("issues") != []):
        raise ValueError("spectral or mass-MAC certification failed")
    rel = comparison.get("relative_frequency_errors")
    if not isinstance(rel, list) or len(rel) != MODES:
        raise ValueError("missing eight frequency comparison values")
    rtol = float(comparison.get("frequency_rtol", 0))
    atol = float(comparison.get("frequency_atol_hz", 0))
    if (not 0 < rtol <= 2.e-5 or not 0 <= atol <= 1.e-4):
        raise ValueError("comparison tolerance is too permissive")
    for j, (expected, actual) in enumerate(zip(reference, trial)):
        if (abs(expected - actual) > atol + rtol * expected
                or not math.isfinite(float(rel[j]))):
            raise ValueError("frequency mismatch at mode " + str(j + 1))
    groups = comparison.get("mode_subspaces")
    if not isinstance(groups, list) or not groups:
        raise ValueError("no certified degenerate modal subspaces")
    visited = []
    minima = []
    for group in groups:
        if not isinstance(group, dict) or group.get("shape_verified") is not True:
            raise ValueError("uncertified modal subspace")
        indices = group.get("modes")
        if (not isinstance(indices, list) or not indices
                or any(type(k) is not int for k in indices)):
            raise ValueError("malformed modal indices")
        visited.extend(indices)
        values = group.get("principal_mass_mac")
        if not isinstance(values, list) or len(values) != len(indices):
            raise ValueError("missing principal MAC of a mode group")
        mac = [float(value) for value in values]
        minimum = float(group.get("minimum_mass_mac"))
        if (not all(math.isfinite(x) and MINIMUM_MAC <= x <= 1.000001
                    for x in mac) or not math.isfinite(minimum)
                or minimum < MINIMUM_MAC
                or abs(min(mac)-minimum) > 1.e-7):
            raise ValueError("mass-weighted modal subspaces do not match")
        minima.append(minimum)
    if sorted(visited) != list(range(1, MODES + 1)):
        raise ValueError("missing or repeated modal index")
    return {
        "status": "PASS",
        "max_relative_frequency_error": max(float(v) for v in rel),
        "minimum_principal_mass_mac": min(minima),
        "frequency_modes_checked": MODES,
        "cause": "",
    }


def summarize(directory):
    directory = Path(directory)
    rows = []
    for theory in THEORIES:
        for rank in RANKS:
            name = "modal_" + theory + "_rank" + str(rank) + "_audit.json"
            path = directory / name
            row = {"theory": theory, "rank": rank, "grid": "13x13",
                   "support": "C-C", "status": "MISSING",
                   "max_relative_frequency_error": "",
                   "minimum_principal_mass_mac": "", "cause": ""}
            if not path.is_file():
                row["cause"] = "No GitHub artifact: " + name
            else:
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                    row.update(verify_case(theory, rank, report))
                except (ValueError, TypeError, OverflowError, KeyError,
                        OSError, json.JSONDecodeError) as exc:
                    row["status"] = "FAIL"
                    row["cause"] = str(exc)
            rows.append(row)
    if len(rows) != 56:
        raise AssertionError("MODAL_56_CASES_NOT_ENUMERATED")
    return rows


def write_report(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n_passed = sum(row["status"] == "PASS" for row in rows)
    verdict = "PASS" if n_passed == 56 else "INCOMPLETE_OR_FAILED"
    report = {"status": verdict, "passed": n_passed, "total": 56,
              "head_sha": os.environ.get("GITHUB_SHA", ""),
              "scope": "C-C SBP42 13x13, first eight MVI vs independent FDM",
              "rows": rows}
    (output / "matrix-56.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "matrix-56.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 56 modal cases: KL / FSDT / REDDY / CUF TE1-TE4, MVI-1..8",
        "",
        "**Result:** " + verdict + " (" + str(n_passed) + "/56)",
        "",
        "| Theory | " + " | ".join("MVI-" + str(k) for k in RANKS) + " |",
        "|---|" + "|".join("---" for _ in RANKS) + "|",
    ]
    for theory in THEORIES:
        selected = [r for r in rows if r["theory"] == theory]
        lines.append("| " + theory + " | " +
                     " | ".join(row["status"] for row in selected) + " |")
    lines.extend([
        "", "PASS requires first-eight spectral agreement and physical",
        "mass-weighted principal MAC >= 0.999 for each modal subspace.",
        "This is NOT an independent 3D FEM or nonseparable shell benchmark.", "",
    ])
    (output / "matrix-56.md").write_text("\n".join(lines), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
                "a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    print("[MODAL-56] " + verdict + " " + str(n_passed) + "/56", flush=True)
    for row in rows:
        print("[MODAL-56] " + row["theory"] + " MVI-" + str(row["rank"]) +
              " " + row["status"] + " " + str(row["cause"]), flush=True)
    return verdict


def main():
    root = Path(os.environ.get("MODAL_MATRIX_ARTIFACT_DIR", "."))
    verdict = write_report(summarize(root), "modal-matrix-56-report")
    if verdict != "PASS":
        raise SystemExit("MODAL_MVI_56_CASES_INCOMPLETE_OR_MISMATCH")


if __name__ == "__main__":
    main()
