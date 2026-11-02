#!/usr/bin/env python3
"""Actual CUF TE1..TE4 / MVI-1..8 modal checks on one physical Rawat C-C case.

Carrera is the kinematic framework. Rawat 2016 provides the geometrical
setting, NOT a TE1..TE4 Carrera modal frequency table. This test validates
MVI against a separately solved full-space FDM on the SAME discretization.
Do not report its PASS as independent Carrera, 3D or literature validation.

Each group solves FDM once and every assigned MVI rank separately.
Failed/missing comparisons are retained, never replaced with FDM results.
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
from cylinder_modal_reference import _modal_spec, solve_modal_case
from modal_mvi_case_comparison import compare_modal_results
from modal_mvi_rawat_te1_cc_audit import (
    CONFIG, check_comparable_cases, harmonic_numbers, read_case)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark_carrera_cuf_te_ranks"
MODELS = ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
GROUPS = {"low": (1, 2, 3, 4), "high": (5, 6, 7, 8)}
N_MODES = 8


def case(model, rank):
    stem = f"RAWAT2016_{model}_13x13_MODAL"
    path = (CONFIG / "fdm_13x13" / (stem + "_FDM.json") if rank is None
            else CONFIG / "mvi_13x13" /
            (stem + f"_MVI{rank}.json"))
    content = read_case(path)
    content["modal"]["n_modes"] = N_MODES
    content["output_dir"] = f"OUTPUT_CARRERA_TE_RANK_AUDIT_{model}_{rank}"
    if rank is not None:
        content["reference_mvi"]["max_sweeps"] = 45
        content["modal"]["eigen_tolerance"] = 1e-7
        content["modal"]["mode_tolerance"] = 1e-5
    return content


def compute(name, config, destination):
    started = time.perf_counter()
    record = {"name": name, "status": "ERROR", "n_modes_requested": N_MODES}
    print(f"[CARRERA-CUF] START {name}", flush=True)
    try:
        solved = solve_modal_case(config)
        frequencies = np.asarray(solved["frequencies_hz"], dtype=float)
        modes = np.asarray(solved["mechanical_modes"], dtype=float)
        rank = None if name == "FDM" else int(name.split("-")[-1])
        if (frequencies.shape != (N_MODES,) or
                modes.shape[0] != N_MODES or
                not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or
                np.any(np.diff(frequencies) < -1e-8) or
                not np.isfinite(modes).all()):
            raise ValueError("CARRERA_CUF_INCOMPLETE_OR_INVALID_EIGHT_MODES")
        if (rank is None and solved.get("modal_solver_branch") != "MKR_FULL_SPACE"):
            raise ValueError("CARRERA_CUF_FDM_NOT_FULL_SPACE_BRANCH")
        if (rank is not None and
                (solved.get("modal_solver_branch") != "MVI_FIXED_RANK_YX" or
                 solved.get("rank") != rank or
                 solved.get("harmonic_sector_restriction") is not False)):
            raise ValueError("CARRERA_CUF_MVI_NOT_UNRESTRICTED_FIXED_RANK")
        filename = name + "_modes.npz"
        np.savez_compressed(destination / filename, frequencies_hz=frequencies,
                            mechanical_modes=modes)
        record.update(
            status="SOLVED", frequencies_hz=frequencies.tolist(),
            circumferential_n=harmonic_numbers(modes),
            rank=rank, saved_fields=filename,
            solver_branch=solved.get("modal_solver_branch"),
            global_spectrum_completeness_verified=solved.get(
                "global_spectrum_completeness_verified"),
            mvi_sweeps=solved.get("mvi_sweeps"),
            rank_scope=solved.get("rank_scope"),
            global_mode_ranks=solved.get("global_mode_ranks"))
        print(f"[CARRERA-CUF] DONE {name} " + json.dumps({
            "freq_hz": record["frequencies_hz"],
            "mvi_sweeps": record["mvi_sweeps"],
            "circumferential_n": record["circumferential_n"],
            "global_spectrum_complete": record[
                "global_spectrum_completeness_verified"]}), flush=True)
        return record, (frequencies, modes)
    except Exception as exc:
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc(limit=6)
        print(f"[CARRERA-CUF] ERROR {name}: {exc}", flush=True)
        return record, None
    finally:
        record["wall_seconds"] = round(time.perf_counter() - started, 4)
        (destination / (name + "_result.json")).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")


def run(model, group):
    if model not in MODELS or group not in GROUPS:
        raise ValueError("expected CUF_TE1..CUF_TE4 and low/high")
    folder = OUTPUT / model / group
    folder.mkdir(parents=True, exist_ok=True)
    fdm_case = case(model, None)
    ranks = GROUPS[group]
    summary = {
        "theory": model, "ranks": list(ranks), "grid": [13, 13],
        "boundary": "C-C", "kinematic_family": "Carrera CUF Taylor TE1..TE4",
        "physical_case": "Rawat_Matsagar_Nagpal_2016_CC_isotropic_cylinder",
        "independent_carrera_modal_table_matched": False,
        "fdm_same_discretization_not_independent_physics": True,
        "n_modes": N_MODES, "runs": {}
    }
    reference = None
    try:
        for rank in ranks:
            compare = check_comparable_cases(fdm_case, case(model, rank))
            if "SBP42" not in compare["fdm"] and "SBP21" not in compare["fdm"]:
                raise ValueError("UNEXPECTED_MODAL_OPERATOR")
        summary["effective_operators"] = compare
        fdm, reference = compute("FDM", fdm_case, folder)
        summary["runs"]["FDM"] = fdm
        for rank in ranks:
            name = f"MVI-{rank}"
            mvi, trial = compute(name, case(model, rank), folder)
            summary["runs"][name] = mvi
            comparison = {
                "status": "NOT_CHECKED",
                "scope": "identical_discretization_not_independent_Carrera_reference",
            }
            if reference is not None and trial is not None:
                try:
                    prepared, density, _ = _modal_spec(fdm_case)
                    comparison = compare_modal_results(
                        reference[0], trial[0], reference[1], trial[1],
                        prepared.problem.functional, density)
                except Exception as exc:
                    comparison = {"status": "ERROR", "error": repr(exc)}
            mvi["comparison_with_fdm"] = comparison
            (folder / (name + "_comparison.json")).write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            (folder / (name + "_result.json")).write_text(
                json.dumps(mvi, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            print("[CARRERA-CUF] COMPARE " + json.dumps({
                "model": model, "rank": rank,
                "status": comparison["status"],
                "max_relative_frequency_error": comparison.get(
                    "maximum_relative_frequency_error"),
                "issues": comparison.get("issues"),
                "error": comparison.get("error")
            }), flush=True)
            (folder / "group_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        summary["all_solved"] = bool(
            fdm["status"] == "SOLVED" and all(
                summary["runs"][f"MVI-{rank}"]["status"] == "SOLVED"
                for rank in ranks))
        summary["all_fdm_subspaces_match"] = bool(
            summary["all_solved"] and all(
                summary["runs"][f"MVI-{rank}"]
                ["comparison_with_fdm"]["status"] == "PASS" for rank in ranks))
        return 0 if summary["all_fdm_subspaces_match"] else 1
    finally:
        (folder / "group_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print("[CARRERA-CUF] GROUP " + json.dumps({
            "model": model, "group": group,
            "all_solved": summary.get("all_solved", False),
            "all_fdm_subspaces_match": summary.get(
                "all_fdm_subspaces_match", False)
        }), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python script.py CUF_TE1..CUF_TE4 low|high")
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
