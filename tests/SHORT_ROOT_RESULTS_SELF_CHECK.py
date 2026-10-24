#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast path-only check: no numerical solution is performed."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import final_run_bundle as runbundle


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "SHORT_ROOT_RESULTS_SELF_CHECK_WORK"


def _cfg() -> dict:
    return {
        "theory": "TE1",
        "cuf_model": "TE1",
        "method": "MVI",
        "mode": "NONLINEAR_ELASTIC",
        "rank": 1,
        "geometry": {"L": 100.0, "R": 50.0, "h": 1.0},
        "material": {"E0": 70000.0, "nu0": 0.3},
        "mvi_grid": {"nx": 11, "ns": 11, "stencil": 5},
        "quadrature": {"type": "GAUSS", "n_gauss": 5},
        "boundary_conditions": {"left": "CLAMPED", "right": "CLAMPED"},
        "load_definition": {"type": "UNIFORM_RADIAL_FACE", "q": 500.0},
    }


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    try:
        first = runbundle.prepare_run_bundle(_cfg(), WORK, source_base=ROOT)
        second = runbundle.prepare_run_bundle(_cfg(), WORK, source_base=ROOT)
        first_dir = Path(first["run_dir"])
        second_dir = Path(second["run_dir"])
        assert first_dir.parent == WORK.resolve()
        assert second_dir.parent == WORK.resolve()
        assert first_dir.name.startswith("RUN_001__")
        assert second_dir.name.startswith("RUN_002__")
        assert Path(first["calculation_dir"]) == WORK.resolve()
        assert Path(first["runs_index_csv"]) == WORK.resolve() / "RUNS_INDEX.csv"
        assert not any(p.is_dir() and p.name.startswith("CASE__") for p in WORK.iterdir())
        saved = json.loads((first_dir / "INPUT" / "FINAL_CASE.json").read_text(encoding="utf-8"))
        assert Path(saved["output_dir"]) == first_dir
        print("SHORT_ROOT_RESULTS_SELF_CHECK: PASS")
        print(" ", first_dir)
        print(" ", second_dir)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
