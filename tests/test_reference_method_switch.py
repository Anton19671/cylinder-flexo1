"""Method switch contract: no calculations and no edits to the original CASE."""
import copy
import json
from pathlib import Path

import pytest

from cylinder_reference_method_switch import selected_case, create_launch_case
from cylinder_case_io import load_case_file


def fixture():
    return {
        "solver_backend": "VARIATIONAL_REFERENCE",
        "calculation_kind": "MODAL",
        "method": "FDM",
        "fdm_backend": "DENSE_REFERENCE",
        "fdm_grid": {"nx": 9, "ns": 9, "operator_scheme": "LEGACY_REFERENCE"},
        "output_dir": "../MODAL_OUTPUT",
        "geometry": {"L": 1.0, "R": 0.5, "h": 0.01},
        "modal": {"density_kg_m3": 7800, "n_modes": 6},
    }


def test_switch_preserves_original_and_physics_and_grid():
    original=fixture()
    saved=copy.deepcopy(original)
    for rank in range(1, 9):
        mvi=selected_case(original,"MVI",rank)
        assert mvi["rank"]==rank
        assert mvi["mvi_grid"]==original["fdm_grid"]
        assert "fdm_grid" not in mvi
        assert mvi["geometry"]==original["geometry"]
        assert mvi["modal"]==original["modal"]
        fdm=selected_case(mvi,"FDM")
        assert fdm["fdm_grid"]==original["fdm_grid"]
        assert "rank" not in fdm and "reference_mvi" not in fdm
    assert original==saved


def test_rank_cannot_silently_increase_mesh():
    original=fixture()
    original["fdm_grid"]["nx"]=5
    with pytest.raises(ValueError,match="no automatic resizing"):
        selected_case(original,"MVI",8)
    assert original["fdm_grid"]["nx"]==5


def test_matrixfree_switch_fails_closed():
    original=fixture()
    original["fdm_backend"]="MATRIX_FREE_GMRES"
    with pytest.raises(ValueError,match="matrix-free"):
        selected_case(original,"MVI",1)


def test_launch_creates_unique_case_without_overwriting_original(tmp_path):
    source=tmp_path/"CASE.json"
    data=fixture()
    data["output_dir"]="output"
    original=json.dumps(data,ensure_ascii=False)
    source.write_text(original,encoding="utf-8")
    selected=selected_case(data,"MVI",3)
    launch_path, launch_case=create_launch_case(source,selected)
    assert launch_path!=source
    assert source.read_text(encoding="utf-8")==original
    assert load_case_file(launch_path)==launch_case
    assert launch_case["method"]=="MVI" and launch_case["rank"]==3
    assert launch_case["mvi_grid"]==data["fdm_grid"]
    assert launch_case["output_dir"]!=data["output_dir"]
    assert not (tmp_path/"output").exists()
    assert not (tmp_path/"output__MVI3").exists()
    other_path,other_case=create_launch_case(source,selected)
    assert other_path!=launch_path
    assert other_case["output_dir"]!=launch_case["output_dir"]


def test_named_reference_runs_get_unique_child_directories(tmp_path):
    original=fixture()
    original["output_dir"]=str(tmp_path/"RESULTS")
    source=tmp_path/"reference.json"
    source.write_text(json.dumps(original),encoding="utf-8")
    existing=tmp_path/"RESULTS"/"OLD_CALCULATION"
    existing.mkdir(parents=True)
    (existing/"result.txt").write_text("preserve",encoding="utf-8")
    first_path,first=create_launch_case(source,original,run_label="Проверка TE4")
    second_path,second=create_launch_case(source,original,run_label="Проверка TE4")
    for case in (first,second):
        out=Path(case["output_dir"])
        assert out.parent==tmp_path/"RESULTS"
        assert "Проверка_TE4" in out.name
        assert not out.exists()
    assert first["output_dir"]!=second["output_dir"]
    assert first_path!=second_path
    assert (existing/"result.txt").read_text()=="preserve"
    assert json.loads(source.read_text())==original


def test_user_run_name_cannot_escape_results_root(tmp_path):
    original=fixture()
    original["output_dir"]=str(tmp_path/"OUTPUT")
    source=tmp_path/"input.json"
    source.write_text(json.dumps(original),encoding="utf-8")
    _,case=create_launch_case(source,original,run_label="../test:bad?name")
    assert Path(case["output_dir"]).parent==tmp_path/"OUTPUT"
    assert not Path(case["output_dir"]).exists()
