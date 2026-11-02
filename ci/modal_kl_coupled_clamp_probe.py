#!/usr/bin/env python3
"""Real Cammalleri KL C-C 13x21 one-mode MVI1..8 / FDM smoke audit.

This is deliberately NOT the final 12-mode publication benchmark.  The same
physical CASE is used for each method; only method/rank differs.  We do not
replace a failed MVI by FDM or claim that one successful eigenpair establishes
completeness of the low spectrum.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from cylinder_paths import activate

activate()
from cylinder_modal_reference import solve_modal_case
from modal_primary_mvi_benchmarks import make_case

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "modal_kl_coupled_clamp_one_mode"
GROUPS = {
    "low": ["MVI_1", "MVI_2", "MVI_3", "MVI_4"],
    "high": ["MVI_5", "MVI_6", "MVI_7", "MVI_8"],
    "mkr": ["MKR_FDM"],
}


def run(group):
    if group not in GROUPS:
        raise ValueError("unknown group")
    dest = OUT / group
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for method in GROUPS[group]:
        case = make_case("cammalleri_kl_13x21", method)
        case["modal"]["n_modes"] = 1
        (dest / (method + "_CASE.json")).write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        status = {"method": method, "case": "Cammalleri 2023, KL C-C 13x21",
                  "requested_modes": 1,
                  "spectrum_completeness_claimed": False,
                  "status": "FAILED"}
        t0 = time.perf_counter()
        events = {}
        def detail(**event):
            stage = event.get("stage", "")
            if stage:
                events[stage] = events.get(stage, 0) + 1
        try:
            print(f"[KL-CLAMP] START {method}", flush=True)
            result = solve_modal_case(case, detail_callback=detail)
            freq = np.asarray(result["frequencies_hz"], dtype=float)
            fields = np.asarray(result["mechanical_modes"], dtype=float)
            if (freq.shape != (1,) or fields.shape != (1, 3, 13, 21)
                    or not np.isfinite(freq).all() or
                    not np.isfinite(fields).all() or np.any(freq <= 0.)):
                raise ValueError("KL_CLAMP_INVALID_PHYSICAL_EIGENMODE")
            branch = result.get("modal_solver_branch")
            if method == "MKR_FDM":
                if branch != "MKR_FULL_SPACE" or result.get("method") != "FDM":
                    raise ValueError("KL_CLAMP_MKR_BRANCH_CROSSCONTAMINATION")
            elif (branch != "MVI_FIXED_RANK_YX" or
                  result.get("method") != "MVI" or
                  result.get("rank") != int(method[-1]) or
                  not result.get("classical_unrestricted_mvi") or
                  result.get("harmonic_sector_restriction")):
                raise ValueError("KL_CLAMP_MVI_BRANCH_CROSSCONTAMINATION")
            fft = np.abs(np.fft.rfft(fields[0], axis=-1))**2
            dominant_n = int(np.argmax(np.sum(fft, axis=(0, 1))))
            np.savez_compressed(
                dest / (method + "_PHYSICAL_MODE.npz"),
                mechanical_modes=fields, frequencies_hz=freq,
                dominant_circumferential_n=np.array([dominant_n]))
            status.update(
                status="SOLVED_AXIAL_M_UNVERIFIED",
                frequencies_hz=freq.tolist(),
                dominant_circumferential_n=dominant_n,
                mvi_sweeps=result.get("mvi_sweeps"),
                global_spectrum_completeness_verified=result.get(
                    "global_spectrum_completeness_verified"),
                modal_solver_branch=branch,
                events=events,
                physical_modes_file=method + "_PHYSICAL_MODE.npz")
        except Exception as exc:
            status.update(error=repr(exc), traceback=traceback.format_exc(limit=10))
            print(f"[KL-CLAMP] FAILED {method}: {exc}", flush=True)
        finally:
            status["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
            (dest / (method + "_RESULT.json")).write_text(
                json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print("[KL-CLAMP] RESULT " +
                  json.dumps({k: v for k, v in status.items()
                              if k not in ("events", "traceback")},
                             ensure_ascii=False), flush=True)
            records.append(status)
    summary = {"scope": "KL C-C 13x21 physical one mode only",
               "mvi_primary": True, "mkr_control_only": True,
               "runs": records,
               "all_cases_solved": all(x["status"] ==
                   "SOLVED_AXIAL_M_UNVERIFIED" for x in records)}
    (dest / "group_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return 0 if summary["all_cases_solved"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: modal_kl_coupled_clamp_probe.py low|high|mkr")
    raise SystemExit(run(sys.argv[1]))
