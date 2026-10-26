"""Elastic-only public route retains both full and reduced flexoelectric models."""
import json
from pathlib import Path
import pytest
from cylinder_final_stage11a_dispatcher import _load_case,FinalCaseError,MODES

def test_modes_elastic_only():
    assert MODES == ("NONLINEAR_ELASTIC",)

def test_no_silent_legacy_fallback(tmp_path):
    path=tmp_path/"case.json"
    path.write_text(json.dumps({"solver_backend":"CLASSICAL"}),encoding="utf-8")
    with pytest.raises(FinalCaseError,match="VARIATIONAL_REFERENCE"):
        _load_case(path)

def test_two_gradient_theories_preserved():
    templates=Path(__file__).resolve().parents[1]/"config"
    for theory in ("G3D","GZ"):
        path=templates/("CASE_REFERENCE_CUF_"+theory+"_STATIC.json")
        case=json.loads(path.read_text(encoding="utf-8"))
        assert case["material_model"]=="HOOKE"
        assert case["micro"]["gradient_model"].upper()==theory
