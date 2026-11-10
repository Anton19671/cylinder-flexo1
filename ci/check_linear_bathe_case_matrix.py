#!/usr/bin/env python3
"""Validate all 63 independent LINEAR Bathe case configurations without solving.

Seven theories, one full-nodal FDM reference and eight MVI ranks per theory.
The generator only reads JSON and prepares dictionaries: no physics is run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci"))

from generate_linear_bathe_case import THEORIES, make_case


def expect_rejected(call, code):
    try:
        call()
    except ValueError as exc:
        if code not in str(exc):
            raise AssertionError(f"Expected {code}, got {exc}") from exc
    else:
        raise AssertionError(f"Expected rejection {code}")


def verify():
    names = set()
    for theory in THEORIES:
        for method, rank in [("FDM", None)] + [("MVI", i) for i in range(1, 9)]:
            label = f"{theory}_{method}{rank if rank is not None else ''}"
            assert label not in names
            names.add(label)
            case = make_case(theory, method, rank, 10, .002, label)
            assert case["method"] == method
            assert case["dynamics"]["integrator"] == "BATHE"
            assert case["dynamics"]["time_step_s"] == .002
            assert case["dynamics"]["n_steps"] == 10
            assert case["geometry_model"] == "LINEAR"
            assert case["mode"] == "LINEAR_ELASTIC"
            assert case["material_model"] == "HOOKE"
            assert case["output_dir"] == "output/" + label
            assert case["theory"] == ("CUF" if theory.startswith("CUF_") else theory)
            if theory.startswith("CUF_"):
                assert case["cuf_model"] == theory.split("_")[1]
                assert case["quadrature"]["n_gauss"] == 2 * int(theory[-1]) + 1
            if method == "MVI":
                assert case["rank"] == rank
                assert "fdm_grid" not in case
                assert case["mvi_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD"
            else:
                assert "rank" not in case
                assert "mvi_grid" not in case
                assert case["fdm_backend"] == "MATRIX_FREE_GMRES"
                assert case["fdm_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD_CSR"
                assert case["reference_newton"]["krylov_max_matvecs"] == (
                    1200 if theory == "CUF_TE4" else 300
                )
    assert len(names) == 63

    for value in (0, -0.001, .00501, float("nan"), float("inf"),
                  -float("inf"), True, "0.002"):
        expect_rejected(
            lambda value=value: make_case("KL", "MVI", 1, 1, value, "KL_MVI1"),
            "BATHE_CONTROL_INVALID_TIME_GRID",
        )
    for value in (0, -1, 1.5, True):
        expect_rejected(
            lambda value=value: make_case("KL", "MVI", 1, value, .002, "KL_MVI1"),
            "BATHE_CONTROL_INVALID_TIME_GRID",
        )
    for name in ("", ".", "..", "../escape", "nested/subfolder",
                 "nested\\subfolder", "/absolute"):
        expect_rejected(
            lambda name=name: make_case("KL", "MVI", 1, 1, .002, name),
            "BATHE_CONTROL_INVALID_OUTPUT_FOLDER",
        )
    for rank in (0, 9, None, 1.5, True):
        expect_rejected(
            lambda rank=rank: make_case("KL", "MVI", rank, 1, .002, "KL_MVI1"),
            "BATHE_MVI_RANK_OUT_OF_RANGE",
        )
    expect_rejected(
        lambda: make_case("KL", "FDM", 1, 1, .002, "KL_FDM"),
        "BATHE_FDM_CANNOT_HAVE_MVI_RANK",
    )
    print("PASS: 63 unique Bathe physical-case specifications and invalid-input guards")


if __name__ == "__main__":
    verify()
