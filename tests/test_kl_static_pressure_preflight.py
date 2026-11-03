"""Preflight only: KL C-C uniform dead pressure, 8 genuine MVI ranks and FDM."""
import json
from pathlib import Path
import numpy as np
import pytest
from cylinder_reference_case import prepare_reference_case

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "config/verification/kl_cc_uniform_static"


def case_files():
    return sorted(DIR.glob("KL_CC_9x9_*KPA_*.json"))


def test_kl_cc_case_matrix_has_nine_main_methods_and_two_linearity_controls():
    files = case_files()
    assert len(files) == 11, "9 main methods + 2 half-pressure controls"
    ids = {p.stem for p in files}
    for pressure in ("100KPA",):
        assert f"KL_CC_9x9_{pressure}_MKR_FDM" in ids
        for rank in range(1, 9):
            assert f"KL_CC_9x9_{pressure}_MVI{rank}" in ids
    assert "KL_CC_9x9_50KPA_MKR_FDM" in ids
    assert "KL_CC_9x9_50KPA_MVI1" in ids


@pytest.mark.parametrize("path", case_files(), ids=lambda path: path.stem)
def test_case_preflight_and_identical_discrete_physics(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    before = path.read_bytes()
    prepared = prepare_reference_case(case, base_dir=path.parent)
    assert before == path.read_bytes(), "preflight must not rewrite CASE"
    assert prepared.calculation_kind == "STATIC"
    assert prepared.problem.functional.theory == "KL"
    assert prepared.problem.functional.geometry_model == "LINEAR"
    assert prepared.electrical is None
    assert prepared.problem.functional.material.ell_m == 0.0
    assert case["scale_model"] == "MACRO"
    assert case["geometry"] == {"L": 2.0, "R": 1.0, "h": 0.05}
    assert case["material"] == {"E0": 210000000000.0, "nu0": 0.3}
    assert case["load_definition"]["face"] == "MIDDLE"
    assert case["load_definition"]["kinematics"] == "DEAD_REFERENCE"
    assert case["load_definition"]["direction"] == "RADIAL_INWARD"
    assert set(case["boundary_conditions"]["variational"]["mechanical"]) == {"left", "right"}
    assert all(v["support"] == "CLAMPED" for v in
               case["boundary_conditions"]["variational"]["mechanical"].values())
    assert case["fdm_grid" if prepared.method == "FDM" else "mvi_grid"] == {
        "nx": 9, "ns": 9, "operator_scheme": "SBP42_FOURIER_ODD"}
    assert prepared.mechanical.shape == (3, 9, 9)
    assert np.all(prepared.mechanical == 0.0)
    pressure = float(case["load_definition"]["q"])
    np.testing.assert_allclose(prepared.dead_load[2], -pressure, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prepared.dead_load[:2], 0., rtol=0, atol=0)
    area = float(np.sum(prepared.problem.functional.ops.area_weights))
    assert np.isclose(area, 2.0 * np.pi * 2.0 * 1.0, rtol=1e-12)
    assert np.isclose(np.sum(prepared.dead_load[2] *
                             prepared.problem.functional.ops.area_weights),
                      -pressure * area, rtol=1e-12)
    if prepared.method == "MVI":
        assert case["rank"] in range(1, 9)
        assert prepared.mvi_settings.rank == case["rank"]
    else:
        assert prepared.mvi_settings is None
        assert case["method"] == "FDM"
    assert len(prepared.factors) == 1 and prepared.factors[0] == 1.0
