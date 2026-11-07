"""385 CASE route checks and independent method isolation; no solves."""
import json
import numpy as np
import pytest
from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from independent_mkr.run_models import load_case as load_mkr
from independent_mvi.run_models import load_case as load_mvi
from ci.scientific_complete_method_matrix import (
    THEORIES,INTEGRATORS,catalog,macro_catalog,macro_case,
    independent_catalog,independent_template,export)


def test_inventory_385_unique_without_physical_success_claims(tmp_path):
    data=export(tmp_path/"matrix")
    assert len(catalog())==385
    assert data["family_counts"]=={
        "MICRO_REFERENCE":112,"MACRO_REFERENCE":84,
        "INDEPENDENT_LINEAR_MACRO":189}
    assert not data["physical_solutions_computed"]
    assert not data["scientific_accuracy_certified"]
    assert len({r["id"] for r in data["cases"]})==385
    assert {r["theory"] for r in data["cases"]}==set(THEORIES)
    assert all(r["validation_level"]=="CASE_PREFLIGHT_ONLY"
               for r in data["cases"])


@pytest.mark.parametrize("item",macro_catalog(),ids=lambda e:e["id"])
def test_all_macro_theories_geometries_methods_and_kinds(item,tmp_path):
    unused=tmp_path/"never_run"
    case=macro_case(item,unused)
    assert "micro" not in case
    assert case["scale_model"]=="MACRO"
    assert case["geometry_model"]==item["geometry_model"]
    assert case["method"]==item["method"]
    assert case["calculation_kind"]==item["kind"]
    assert case["load_definition"]["face"]==(
        "MIDDLE" if item["theory"]=="KL" else "OUTER")
    prepared=prepare_reference_case(case,base_dir=tmp_path)
    assert prepared.problem.functional.theory==item["theory"]
    assert prepared.problem.functional.geometry_model==item["geometry_model"]
    assert prepared.problem.functional.material.ell_m==0
    assert prepared.electrical is None
    assert prepared.method==item["method"]
    assert prepared.calculation_kind==item["kind"]
    if item["kind"]=="DYNAMIC":
        assert prepared.dynamics is not None
        assert np.isfinite(prepared.initial_acceleration).all()
    else:
        assert prepared.dynamics is None
    assert not unused.exists()


@pytest.mark.parametrize("item",independent_catalog(),ids=lambda e:e["id"])
def test_all_independent_mkr_mvi_seven_theories_three_integrators_eight_ranks(
        item,tmp_path):
    case=independent_template(item["theory"],item["integrator"],item["method"])
    if item["method"]=="MKR":
        assert "rank" not in case
        path=tmp_path/"mkr.json"
        path.write_text(json.dumps(case),encoding="utf-8")
        model,integrator,dt,steps,output,_=load_mkr(path)
    else:
        case["rank"]=item["rank"]
        path=tmp_path/"mvi.json"
        path.write_text(json.dumps(case),encoding="utf-8")
        model,rank,integrator,dt,steps,output,_=load_mvi(path)
        assert rank==item["rank"]
        assert rank<=min(model.nx-(4 if model.theory=="KL" else 2),model.ns)
    assert model.theory==item["theory"]
    assert integrator==item["integrator"]
    assert model.nx==15 and model.ns==9
    assert model.pressure_kind=="PATCH"
    assert steps==1 and 0<dt<=.1
    assert not output.exists()


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("integrator",INTEGRATORS)
def test_independent_mkr_and_mvi_share_continuum_but_not_case_schema(
        theory,integrator):
    mkr=independent_template(theory,integrator,"MKR")
    mvi=independent_template(theory,integrator,"MVI")
    assert mkr["physics"]==mvi["physics"]
    assert mkr["time"]==mvi["time"]
    assert mkr["schema"]!=mvi["schema"]


def test_micro_does_not_route_to_independent_elastic_only_solver(tmp_path):
    from ci.scientific_real_coupled_case import build_case
    source=build_case(dict(
        theory="CUF_TE4",gradient="G3D",on=True,
        kind="DYNAMIC",method="FDM"),tmp_path/"unused")
    file=tmp_path/"micro.json"
    file.write_text(json.dumps(source),encoding="utf-8")
    with pytest.raises(ValueError,match="MKR_CASE_KEYS_INVALID"):
        load_mkr(file)
    with pytest.raises(ValueError,match="MVI_CASE_KEYS_INVALID"):
        load_mvi(file)
