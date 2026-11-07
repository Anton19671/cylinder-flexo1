"""Independent native cases through the common public final dispatcher.

Every case is a dry-run validation. No physical time integration is run.
The only execution call is monkeypatched to a same-method fake, proving
selection without opening either real numerical solver.
"""
from __future__ import annotations
import json
import pytest

from cylinder_paths import activate
activate()

from cylinder_final_stage11a_dispatcher import (
    validate_final_case,run_final_case,run_mvi_rank_series,FinalCaseError)
from cylinder_independent_dispatch import (
    independent_method,validate_independent_case,run_independent_case)
from independent_mkr import run_models as mkr_runner
from independent_mvi import run_models as mvi_runner
from ci.scientific_complete_method_matrix import (
    independent_catalog,independent_template)


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("theory",(
    "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("integrator",("BATHE","NEWMARK","GAUSS_IRK4"))
def test_native_case_validates_own_solver_for_every_theory_and_time(
        method,theory,integrator,tmp_path
):
    case=independent_template(theory,integrator,method)
    if method=="MVI":
        case["rank"]=8
    path=tmp_path/"own.json"
    path.write_text(json.dumps(case),encoding="utf-8")
    validated=validate_final_case(path)
    assert validated["valid"]
    assert validated["validated_only"]
    assert validated["physical_steps_computed"]==0
    assert validated["no_reference_fallback"]
    assert validated["solver_backend"]=="INDEPENDENT_"+method
    assert validated["integrator"]==integrator
    assert validated["theory"]==theory
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_public_execution_calls_only_selected_independent_runner(
    method,monkeypatch,tmp_path
):
    case=independent_template("CUF_TE3","BATHE",method)
    if method=="MVI":case["rank"]=4
    path=tmp_path/"own.json"
    path.write_text(json.dumps(case))
    selected=mkr_runner if method=="MKR" else mvi_runner
    opposite=mvi_runner if method=="MKR" else mkr_runner
    events=[]
    def own_run(path,*,validate_only):
        events.append((method,validate_only))
        return {"method":selected.METHOD,"status":"CONVERGED"}
    def poison(*a,**kw):
        raise AssertionError("CROSS_METHOD_OR_REFERENCE_SOLVER_CALLED")
    monkeypatch.setattr(selected,"run",own_run)
    monkeypatch.setattr(opposite,"run",poison)
    import cylinder_reference_case as reference
    monkeypatch.setattr(reference,"run_reference_case",poison)
    summary=run_final_case(path)
    assert summary["solver_backend"]=="INDEPENDENT_"+method
    assert summary["no_reference_fallback"] is True
    assert events==[(method,False)]


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_independent_wrong_method_or_unsupported_physics_rejected(
    method,tmp_path
):
    case=independent_template("CUF_TE2","NEWMARK",method)
    if method=="MVI":case["rank"]=2
    case["scale_model"]="MICRO"
    file=tmp_path/"invalid.json"
    file.write_text(json.dumps(case))
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        validate_final_case(file)
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        run_final_case(file)


def test_unknown_independent_schema_never_falls_back(tmp_path):
    file=tmp_path/"invalid.json"
    file.write_text(json.dumps({"schema":"INDEPENDENT_FAKE_V1"}))
    with pytest.raises(ValueError,match="UNSUPPORTED_SCHEMA"):
        validate_final_case(file)
    with pytest.raises(ValueError,match="UNSUPPORTED_SCHEMA"):
        run_final_case(file)


def test_reference_case_does_not_automatically_become_independent(tmp_path):
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/(
        "config/verification/kl_cc_bathe/BATHE_MVI1.json")
    case=json.loads(source.read_text())
    path=tmp_path/"reference.json"
    path.write_text(json.dumps(case))
    original=validate_final_case(path)
    assert original["solver_backend"]=="VARIATIONAL_REFERENCE"


def test_native_rank_sweep_rejected_instead_of_reference_fallback(tmp_path):
    case=independent_template("CUF_TE4","NEWMARK","MVI")
    case["rank"]=2
    path=tmp_path/"native.json"
    path.write_text(json.dumps(case))
    with pytest.raises(FinalCaseError,match="OWN_EXPLICIT_CASES"):
        run_mvi_rank_series(path)
