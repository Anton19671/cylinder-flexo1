#!/usr/bin/env python3
"""Rawat CUF TE1 C-C 13x13 modal branch audit.

This is a regression observation for the independent Rawat configuration.
It uses the normal full FDM and unrestricted rank-one MVI paths; no form
from either solver is supplied to the other.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from cylinder_paths import activate
activate()
from cylinder_modal_reference import _effective_modal_case, solve_modal_case

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "verification" / "rawat2016" / "modal_linear"
FDM_CONFIG = CONFIG / "fdm_13x13" / "RAWAT2016_CUF_TE1_13x13_MODAL_FDM.json"
MVI_CONFIG = CONFIG / "mvi_13x13" / "RAWAT2016_CUF_TE1_13x13_MODAL_MVI1.json"
OUT = ROOT / "modal_mvi_rawat_te1_cc_audit_result.json"
N_MODES = 10
TARGET_BRANCHES_HZ = (342.0, 352.0)


def read_case(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_comparable_cases(fdm, mvi):
    """Refuse a comparison of different physical problems or discrete operators."""
    fdm, mvi = _effective_modal_case(fdm), _effective_modal_case(mvi)
    for key in ("theory", "cuf_model", "geometry_model", "geometry",
                "material", "quadrature", "boundary_conditions"):
        if fdm.get(key) != mvi.get(key):
            raise ValueError(f"RAWAT_CASE_MISMATCH: {key}")
    if fdm["modal"]["density_kg_m3"] != mvi["modal"]["density_kg_m3"]:
        raise ValueError("RAWAT_CASE_MISMATCH: density")
    fg, mg = fdm["fdm_grid"], mvi["mvi_grid"]
    if (fg["nx"], fg["ns"], fg["operator_scheme"]) != (
            mg["nx"], mg["ns"], mg["operator_scheme"] + "_CSR"):
        raise ValueError("RAWAT_CASE_MISMATCH: effective grids or operators")
    return {"fdm": fg["operator_scheme"], "mvi": mg["operator_scheme"]}


def harmonic_numbers(fields):
    fields = np.asarray(fields, float)
    spectra = np.abs(np.fft.rfft(fields, axis=-1)) ** 2
    return np.argmax(np.sum(spectra, axis=(1, 2)), axis=-1).astype(int).tolist()


def solve(path):
    case = copy.deepcopy(read_case(path))
    case["modal"]["n_modes"] = N_MODES
    if case["method"] == "MVI":
        case["reference_mvi"]["max_sweeps"] = 32
    result = solve_modal_case(case)
    return {
        "frequencies_hz": np.asarray(result["frequencies_hz"], float).tolist(),
        "wavenumbers": harmonic_numbers(result["mechanical_modes"]),
        "mvi_sweeps": result.get("mvi_sweeps"),
        "global_spectrum_completeness_verified":
            result.get("global_spectrum_completeness_verified")}


def closest_branches(frequencies):
    frequencies = np.asarray(frequencies, float)
    return [{"published_hz": target,
             "computed_hz": float(frequencies[np.argmin(abs(frequencies-target))]),
             "difference_percent": float(
                 100.0 * abs(frequencies[np.argmin(abs(frequencies-target))]-target) / target)}
            for target in TARGET_BRANCHES_HZ]


def main():
    started = time.monotonic()
    report = {
        "scope": "RAWAT2016_CUF_TE1_CC_13x13",
        "published_branches_hz": list(TARGET_BRANCHES_HZ),
        "n_requested": N_MODES,
        "effective_operators": check_comparable_cases(read_case(FDM_CONFIG), read_case(MVI_CONFIG)),
        "runs": {},
    }
    try:
        for name, path in (("FDM", FDM_CONFIG), ("MVI_1", MVI_CONFIG)):
            print(f"[RAWAT] START {name}", flush=True)
            run_started = time.monotonic()
            report["runs"][name] = solve(path)
            report["runs"][name]["elapsed_s"] = round(
                time.monotonic() - run_started, 3)
            report["runs"][name]["closest_published_branches"] = closest_branches(
                report["runs"][name]["frequencies_hz"])
            print("[RAWAT] RESULT " + name + " " +
                  json.dumps(report["runs"][name]), flush=True)
        report["completed"] = True
    except Exception as exc:
        report["completed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[RAWAT] SUMMARY " + json.dumps({
            "completed": report["completed"],
            "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
