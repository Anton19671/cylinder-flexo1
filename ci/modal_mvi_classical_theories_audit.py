#!/usr/bin/env python3
"""Audit C-C KL/FSDT/Reddy FDM and MVI ranks 1 and 2 on equal SBP42 grids."""
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
    if model not in ("KL", "FSDT", "REDDY"):
        raise ValueError("Expected KL, FSDT, or REDDY")
    paths = {
        "FDM": CONFIG / "fdm_13x13" / f"RAWAT2016_{model}_13x13_MODAL_FDM.json",
        "MVI_1": CONFIG / "mvi_13x13" / f"RAWAT2016_{model}_13x13_MODAL_MVI1.json",
        "MVI_2": CONFIG / "mvi_13x13" / f"RAWAT2016_{model}_13x13_MODAL_MVI2.json",
    }
    cases = {name: read_case(path) for name, path in paths.items()}
    operators = {name: check_comparable_cases(cases["FDM"], cases[name])
                 for name in ("MVI_1", "MVI_2")}
    report = {"model": model, "effective_operators": operators,
              "grid": [13, 13], "boundary": "C-C", "runs": {}}
    try:
        for name, case in cases.items():
            case["modal"]["n_modes"] = 8
            if name.startswith("MVI_"):
                case["reference_mvi"]["max_sweeps"] = 32
            print(f"[SHELL-AUDIT] START {model} {name}", flush=True)
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
            print("[SHELL-AUDIT] RESULT " + model + " " + name + " " +
                  json.dumps(report["runs"][name]), flush=True)
        report["completed"] = True
    except Exception as exc:
        report["completed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        Path(f"modal_{model}_audit.json").write_text(json.dumps(report, indent=2))
        print("[SHELL-AUDIT] SUMMARY " + model + " " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
