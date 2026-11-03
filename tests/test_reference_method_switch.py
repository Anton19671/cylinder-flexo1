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


def test_modal_sbp42_mvi17_switches_to_compatible_matrixfree_fdm():
    src=fixture()
    src.update(method="MVI",rank=1,reference_mvi={"max_sweeps":30},
               scale_model="MACRO",fdm_backend="DENSE_REFERENCE")
    src.pop("fdm_grid")
    src["mvi_grid"]={"nx":17,"ns":17,"operator_scheme":"SBP42_FOURIER_ODD"}
    src["reference_newton"]={"tangent_step_scaled":2.e-6,"max_free_dofs":512}
    snapshot=copy.deepcopy(src)
    fdm=selected_case(src,"FDM")
    assert src==snapshot
    assert fdm["method"]=="FDM"
    assert fdm["fdm_grid"]=={"nx":17,"ns":17,
                             "operator_scheme":"SBP42_FOURIER_ODD_CSR"}
    assert fdm["fdm_backend"]=="MATRIX_FREE_GMRES"
    assert fdm["reference_newton"]["max_surface_nodes"]>=289
    assert "max_free_dofs" not in fdm["reference_newton"]
    assert "rank" not in fdm and "reference_mvi" not in fdm
    assert fdm["modal"]==src["modal"]
    back=selected_case(fdm,"MVI",2)
    assert back["method"]=="MVI" and back["rank"]==2
    assert back["fdm_backend"]=="DENSE_REFERENCE"
    assert back["mvi_grid"]==src["mvi_grid"]
    assert "max_total_dofs" not in back["reference_newton"]
    assert "fdm_grid" not in back


def test_modal_sbp42_fdm21_switches_and_remains_compatible():
    src=fixture()
    src.update(scale_model="MACRO",fdm_backend="MATRIX_FREE_GMRES",
               reference_newton={"max_total_dofs":100000,
                                 "max_surface_nodes":8192,
                                 "max_constraint_rows":20000})
    src["fdm_grid"]={"nx":21,"ns":21,
                     "operator_scheme":"SBP42_FOURIER_ODD_CSR"}
    mvi=selected_case(src,"MVI",8)
    assert mvi["fdm_backend"]=="DENSE_REFERENCE"
    assert mvi["mvi_grid"]=={"nx":21,"ns":21,
                            "operator_scheme":"SBP42_FOURIER_ODD"}
    assert mvi["rank"]==8
    assert selected_case(mvi,"FDM")["fdm_grid"]==src["fdm_grid"]


def test_legacy_modal_csr_switch_uses_the_same_effective_sbp42_operator():
    src=fixture()
    src.update(scale_model="MACRO",fdm_backend="MATRIX_FREE_GMRES")
    src["fdm_grid"]["operator_scheme"]="SBP21_PERIODIC_ODD_CSR"
    saved=copy.deepcopy(src)
    selected=selected_case(src,"MVI",1)
    assert src==saved
    assert selected["mvi_grid"]["operator_scheme"]=="SBP42_FOURIER_ODD"
    assert selected["fdm_backend"]=="DENSE_REFERENCE"
    assert selected_case(selected,"FDM")["fdm_grid"]["operator_scheme"]=="SBP42_FOURIER_ODD_CSR"


def test_nonmodal_csr_switch_still_fails_closed():
    src=fixture()
    src.update(scale_model="MACRO",fdm_backend="MATRIX_FREE_GMRES")
    src["fdm_grid"]["operator_scheme"]="SBP42_FOURIER_ODD_CSR"
    src["calculation_kind"]="STATIC"
    with pytest.raises(ValueError,match="incompatible matrix-free/CSR"):
        selected_case(src,"MVI",1)
