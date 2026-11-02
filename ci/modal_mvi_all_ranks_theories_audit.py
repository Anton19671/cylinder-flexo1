#!/usr/bin/env python3
"""C-C 13x13 FDM and MVI rank 2-8 audit for classical and CUF theories."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from modal_mvi_rawat_te1_cc_audit import (
    CONFIG, check_comparable_cases, harmonic_numbers, read_case,
)
from cylinder_modal_reference import _modal_spec, solve_modal_case
from modal_mvi_case_comparison import compare_modal_results


def main():
    model, rank = sys.argv[1], int(sys.argv[2])
    if model not in ("KL", "FSDT", "REDDY", "CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4"):
        raise ValueError("Unsupported model")
    if rank not in range(2, 9):
        raise ValueError("Expected MVI rank 2..8")
    rank_name = f"MVI_{rank}"
    stem = f"RAWAT2016_{model}_13x13_MODAL"
    cases = {
        "FDM": read_case(CONFIG / "fdm_13x13" / f"{stem}_FDM.json"),
        rank_name: read_case(CONFIG / "mvi_13x13" / f"{stem}_MVI{rank}.json"),
    }
    operators = check_comparable_cases(cases["FDM"], cases[rank_name])
    report = {"model": model, "rank": rank, "grid": [13, 13],
              "boundary": "C-C", "effective_operators": operators, "runs": {}}
    computed = {}
    try:
        for name, case in cases.items():
            case["modal"]["n_modes"] = 8
            if name != "FDM":
                case["reference_mvi"]["max_sweeps"] = 32
            start = time.monotonic()
            print(f"[RANK-AUDIT] START {model} {name}", flush=True)
            result = solve_modal_case(case)
            frequencies = np.asarray(result["frequencies_hz"], dtype=float)
            modes = np.asarray(result["mechanical_modes"], dtype=float)
            if len(frequencies) != 8 or len(modes) != 8:
                raise ValueError(f"{name} returned {len(frequencies)} frequencies and {len(modes)} modes")
            computed[name] = (frequencies, modes)
            np.savez_compressed(f"modal_{model}_rank{rank}_{name}.npz",
                                frequencies_hz=frequencies, mechanical_modes=modes)
            report["runs"][name] = {
                "frequencies_hz": frequencies.tolist(),
                "wavenumbers": harmonic_numbers(modes),
                "global_spectrum_completeness_verified":
                    result.get("global_spectrum_completeness_verified"),
                "elapsed_s": round(time.monotonic() - start, 3)}
            print("[RANK-AUDIT] RESULT " + model + " " + name + " " +
                  json.dumps(report["runs"][name]), flush=True)
        prepared, density, _ = _modal_spec(cases["FDM"])
        ref, trial = computed["FDM"], computed[rank_name]
        report["comparison"] = compare_modal_results(
            ref[0], trial[0], ref[1], trial[1],
            prepared.problem.functional, density)
        print("[RANK-AUDIT] COMPARISON " + model + " " + rank_name + " " +
              json.dumps(report["comparison"]), flush=True)
        if report["comparison"]["status"] != "PASS":
            raise AssertionError("MODAL_MVI_MKR_MISMATCH: " +
                                 str(report["comparison"]["issues"]))
        report["completed"] = True
    except Exception as exc:
        report["completed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        Path(f"modal_{model}_rank{rank}_audit.json").write_text(json.dumps(report, indent=2))
        print("[RANK-AUDIT] SUMMARY " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
