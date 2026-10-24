#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Directory/layout regression, without running expensive numerical solves."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cylinder_paths import activate, SOURCE_FOLDERS

ALLOWED_ROOT_PY = {
    "cylinder_paths.py", "FINAL_GUI.py", "RUN_FINAL_CASE.py",
    "RUN_MVI_1_8.py", "FINAL_POSTPROCESS.py", "FINAL_QUICK_SUMMARY.py",
    "FINAL_SELF_CHECK.py", "FINAL_RELEASE_REGRESSION.py",
    "FINAL_MVI_SERIES_POSTPROCESS.py", "final_excel_export.py",
    "final_partner_export.py",
}
REQUIRED = (
    "interface/FINAL_GUI.py", "interface/cylinder_final_stage11a_dispatcher.py",
    "mechanics/cylinder_cuf_stage7a_general_mechanics.py",
    "mechanics/cylinder_kl_stage8a_elastic_mechanics.py",
    "mvi/cylinder_cuf_stage7d_general_internal_mvi.py",
    "mkr/cylinder_cuf_stage10a_strong_fdm_linear_elastic.py",
    "flexoelectric/flexo_gradient_g3d.py", "results/FINAL_POSTPROCESS.py",
    "tests/FINAL_SELF_CHECK.py", "config/MATERIAL_LIBRARY.json",
    "requirements.txt",
)

def main():
    activate()
    assert all((ROOT / name).is_file() for name in REQUIRED), "Missing essential file"
    root_py = {p.name for p in ROOT.glob("*.py")}
    assert root_py == ALLOWED_ROOT_PY, f"Unexpected root Python files: {sorted(root_py ^ ALLOWED_ROOT_PY)}"
    for folder in SOURCE_FOLDERS:
        assert (ROOT / folder).is_dir(), f"Missing {folder}"
    for name in (
        "cylinder_cuf_stage7a_general_mechanics",
        "cylinder_cuf_stage7d_general_internal_mvi",
        "cylinder_cuf_stage10a_strong_fdm_linear_elastic",
        "final_capability_registry", "flexo_gradient_g3d",
        "cylinder_final_stage11a_dispatcher",
    ):
        found = importlib.util.find_spec(name)
        assert found and Path(found.origin).resolve().parent != ROOT, f"Broken source lookup: {name}"
    sources = [p for name in SOURCE_FOLDERS for p in (ROOT / name).rglob("*.py")]
    for p in [*(ROOT.glob("*.py")), *sources]:
        compile(p.read_bytes(), str(p), "exec")
    print(f"LAYOUT PASS: {len(root_py)} root entrypoints/loader; {len(sources)} organized Python files")
    print("Note: numerical regression tests are separate (RUN_SELF_CHECK.cmd)")

if __name__ == "__main__":
    main()
