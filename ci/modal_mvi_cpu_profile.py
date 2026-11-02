#!/usr/bin/env python3
"""Profile a REAL Cammalleri CUF TE1 modal calculation without editing physics.

The diagnostic changes only modal.n_modes to one, so its seed search differs
from the published twelve-mode CASE; never treat it as a 12-mode benchmark.
It runs MVI-5 and full-space MKR/FDM on IDENTICAL material, grid and BCs.
It does not patch the solver or modify K, M, physics, convergence, or rank.
"""
from __future__ import annotations

import cProfile
from collections import Counter
import copy
import io
import json
from pathlib import Path
import pstats
import time
import traceback

import numpy as np
from cylinder_paths import activate
activate()
from cylinder_modal_reference import solve_modal_case

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/verification/cammalleri2023/user_13x21"
CASES = {
    "MVI_5": "CAMMALLERI2023_CUF_TE1_13x21_MVI_5.json",
    "MKR_FDM": "CAMMALLERI2023_CUF_TE1_13x21_MKR_FDM.json",
}
OUT = ROOT / "modal_mvi_cpu_profile_result.json"


def profile_case(name):
    case = json.loads((BASE / CASES[name]).read_text(encoding="utf-8"))
    assert case["modal"]["n_modes"] == 12
    assert case["mvi_grid" if name == "MVI_5" else "fdm_grid"]["operator_scheme"].startswith("SBP42")
    assert case["calculation_kind"] == "MODAL"
    assert case["theory"] == "CUF" and case["cuf_model"] == "TE1"
    assert case["geometry"] == {"L": 0.305, "R": 0.076, "h": 0.000254}
    case = copy.deepcopy(case)
    case["modal"]["n_modes"] = 1
    events = Counter()
    def detail(**event):
        stage = event.get("stage")
        if stage:
            events[stage] += 1
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = {"case": CASES[name], "requested_modes": 1, "status": "FAILED"}
    try:
        profiler.enable()
        output = solve_modal_case(case, detail_callback=detail)
        profiler.disable()
        freq = np.asarray(output["frequencies_hz"], dtype=float)
        assert freq.shape == (1,) and np.isfinite(freq).all() and np.all(freq > 0.)
        result.update(
            status="SOLVED",
            frequencies_hz=freq.tolist(),
            rank=(int(case["rank"]) if name == "MVI_5" else None),
            sweeps=output.get("mvi_sweeps"),
            candidate_attempts=events.get("MVI_CANDIDATE_START", 0),
            candidate_converged=events.get("MVI_CANDIDATE_CONVERGED", 0),
            candidate_rejected=events.get("MVI_CANDIDATE_REJECTED", 0),
            halfsteps=events.get("MVI_HALFSTEP_START", 0),
            events=dict(events),
        )
    except Exception as exc:
        profiler.disable()
        result.update(error=repr(exc), traceback=traceback.format_exc(limit=8))
    result["wall_seconds"] = round(time.perf_counter() - start, 3)
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).strip_dirs().sort_stats("cumtime").print_stats(35)
    result["cpu_profile_top"] = report.getvalue()
    print("[MVI-PROFILE] " + name + " " + json.dumps(
        {key: value for key, value in result.items()
         if key not in ("cpu_profile_top", "events", "traceback")},
        ensure_ascii=False), flush=True)
    print("[MVI-PROFILE] " + name + " TOP CALLS\n" + result["cpu_profile_top"], flush=True)
    return result


def main():
    report = {"case_scope": "Cammalleri CUF TE1 13x21, one physical mode, unmodified solver", "runs": {}}
    for name in ("MKR_FDM", "MVI_5"):
        report["runs"][name] = profile_case(name)
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    a, b = report["runs"]["MKR_FDM"], report["runs"]["MVI_5"]
    if a["status"] == "SOLVED" and b["status"] == "SOLVED":
        report["frequency_relative_difference"] = abs(
            a["frequencies_hz"][0] - b["frequencies_hz"][0]) / a["frequencies_hz"][0]
        report["wall_time_ratio_mvi_over_mkr"] = (
            b["wall_seconds"] / a["wall_seconds"])
    report["status"] = "RECORDED" if all(
        x["status"] == "SOLVED" for x in report["runs"].values()) else "INCOMPLETE"
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[MVI-PROFILE] SUMMARY " + json.dumps({
        "status": report["status"],
        "ratio": report.get("wall_time_ratio_mvi_over_mkr"),
        "frequency_relative_difference": report.get("frequency_relative_difference"),
        "elapsed": {k: v.get("wall_seconds") for k, v in report["runs"].items()},
    }), flush=True)
    return int(report["status"] != "RECORDED")


if __name__ == "__main__":
    raise SystemExit(main())
