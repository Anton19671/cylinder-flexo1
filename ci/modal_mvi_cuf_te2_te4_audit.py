#!/usr/bin/env python3
"""Observe Rawat C-C CUF TE2-TE4 FDM/MVI-1 branches on equal SBP42 grids."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from modal_mvi_rawat_te1_cc_audit import (
    CONFIG, check_comparable_cases, harmonic_numbers, read_case,
)
from cylinder_modal_reference import solve_modal_case


def main():
    model = sys.argv[1]
    if model not in ("TE2", "TE3", "TE4"):
        raise ValueError("Expected TE2, TE3, or TE4")
    paths = {
        "FDM": CONFIG / "fdm_13x13" / f"RAWAT2016_CUF_{model}_13x13_MODAL_FDM.json",
        "MVI_1": CONFIG / "mvi_13x13" / f"RAWAT2016_CUF_{model}_13x13_MODAL_MVI1.json",
    }
    cases = {name: read_case(path) for name, path in paths.items()}
    operators = check_comparable_cases(cases["FDM"], cases["MVI_1"])
    report = {"model": model, "effective_operators": operators,
              "grid": [13, 13], "boundary": "C-C", "runs": {}}
    try:
        for name, case in cases.items():
            case["modal"]["n_modes"] = 8
            if name == "MVI_1":
                case["reference_mvi"]["max_sweeps"] = 32
            print(f"[CUF-AUDIT] START {model} {name}", flush=True)
            started = time.monotonic()
            result = solve_modal_case(case)
            modes = np.asarray(result["mechanical_modes"], dtype=float)
            frequencies = np.asarray(result["frequencies_hz"], dtype=float)
            np.savez_compressed(f"modal_{model}_{name}_modes.npz",
                                mechanical_modes=modes, frequencies_hz=frequencies)
            report["runs"][name] = {
                "frequencies_hz": frequencies.tolist(),
                "wavenumbers": harmonic_numbers(modes),
                "global_spectrum_completeness_verified":
                    result.get("global_spectrum_completeness_verified"),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            print("[CUF-AUDIT] RESULT " + model + " " + name + " " +
                  json.dumps(report["runs"][name]), flush=True)
        report["completed"] = True
    except Exception as exc:
        report["completed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        Path(f"modal_{model}_audit.json").write_text(json.dumps(report, indent=2))
        print("[CUF-AUDIT] SUMMARY " + model + " " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
