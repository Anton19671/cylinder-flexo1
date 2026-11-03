#!/usr/bin/env python3
"""Export the exact effective CASEs used in the eight-mode Rawat CUF CI series.

This does not calculate any eigenfrequencies. It exposes the distinction
between legacy source JSON and the effective MODAL operator after automatic
SBP21 -> SBP42 normalization, and pins the runtime of a successful run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cylinder_paths import activate

activate()

from modal_carrera_cuf_te1_te4_mvi_all_ranks import CONFIG, MODELS, case, read_case
from cylinder_modal_reference import _effective_modal_case

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "benchmark_case_bundle"
RUNTIME = {"python": "3.11", "numpy": "2.4.6", "scipy": "1.17.1",
           "source_workflow_run": 41506564256}
FDM_VARIANT = "MKR_FDM"
METHODS = [FDM_VARIANT] + [f"MVI_{i}" for i in range(1, 9)]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main():
    manifest = {
        "purpose": "Reproduce eight-mode Rawat-geometry CUF TE1..TE4 MVI-1..8 vs MKR/FDM",
        "case_count": len(MODELS) * len(METHODS),
        "runtime_of_successful_CI": RUNTIME,
        "git_sha_of_export": os.environ.get("GITHUB_SHA", "local"),
        "reference_run": "https://github.com/Anton19671/cylinder-flexo1/actions/runs/41506564256",
        "solver_normalizes_legacy_modal_operator": {
            "SBP21_PERIODIC_ODD": "SBP42_FOURIER_ODD",
            "SBP21_PERIODIC_ODD_CSR": "SBP42_FOURIER_ODD_CSR",
        },
        "note": "In this benchmark the source templates name SBP21, but the actual modal calculations used the automatically normalized SBP42. The effective CASE files are directly runnable. MKR is independent full-space reference.",
        "files": [],
    }
    for model in MODELS:
        for name in METHODS:
            rank = None if name == FDM_VARIANT else int(name.split("_")[-1])
            suffix = "FDM" if rank is None else f"MVI{rank}"
            original = CONFIG / ("fdm_13x13" if rank is None else "mvi_13x13") / (
                f"RAWAT2016_{model}_13x13_MODAL_{suffix}.json")
            source = read_case(original)
            used = case(model, rank)
            effective = _effective_modal_case(used)
            if rank is None:
                assert effective["fdm_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD_CSR"
                assert effective["modal"]["n_modes"] == 8
            else:
                assert effective["mvi_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD"
                assert effective["rank"] == rank
                assert effective["modal"]["n_modes"] == 8
                assert effective["reference_mvi"]["max_sweeps"] == 45
            source_path = Path(model) / (name + "_SOURCE_TEMPLATE.json")
            effective_path = Path(model) / (name + "_EFFECTIVE_CASE.json")
            write_json(DEST / source_path, source)
            write_json(DEST / effective_path, effective)
            manifest["files"].append({"theory": model, "method": name,
                                      "original_repo_path": str(original.relative_to(ROOT)),
                                      "source_template": str(source_path),
                                      "run_case": str(effective_path)})
    write_json(DEST / "MANIFEST.json", manifest)
    (DEST / "requirements-successful-CI.txt").write_text(
        "numpy==2.4.6\nscipy==1.17.1\n", encoding="utf-8")
    assert len(manifest["files"]) == 36
    print(f"Exported {len(manifest['files'])} exact solver CASEs without calculations")


if __name__ == "__main__":
    main()
