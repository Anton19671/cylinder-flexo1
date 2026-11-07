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
    independent_catalog,independent_template,
    coupled_optional_time_catalog,coupled_optional_time_case,export)


def test_inventory_385_unique_without_physical_success_claims(tmp_path):
    data=export(tmp_path/"matrix")
    assert len(catalog())==469
    assert data["family_counts"]=={
        "MICRO_REFERENCE":112,"MICRO_BATHE_CASE":56,
        "MICRO_GAUSS_MVI_EXPLICIT_CASE":28,
        "MACRO_REFERENCE":84,"INDEPENDENT_LINEAR_MACRO":189}
    assert not data["physical_solutions_computed"]
    assert not data["scientific_accuracy_certified"]
    assert len({r["id"] for r in data["cases"]})==469
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



def test_optional_time_catalog_coverage_not_conflated_with_newmark():
    rows=coupled_optional_time_catalog()
    assert len(rows)==84
    assert len({r["id"] for r in rows})==84
    bathe=[r for r in rows if r["integrator"]=="BATHE"]
    gauss=[r for r in rows if r["integrator"]=="GAUSS2_IRK4"]
    assert len(bathe)==56 and len(gauss)==28
    assert {r["method"] for r in bathe}=={"FDM","MVI"}
    assert {r["method"] for r in gauss}=={"MVI"}
    assert {r["theory"] for r in gauss}==set(THEORIES)
    assert {r["gradient_model"] for r in gauss}=={"G3D","Gz"}
    assert {r["flexoelectric_enabled"] for r in gauss}=={True,False}
    assert all(r["validation_level"]=="CASE_PREFLIGHT_ONLY" for r in rows)


@pytest.mark.parametrize("item",coupled_optional_time_catalog(),
                         ids=lambda e:e["id"])
def test_84_bathe_gauss_g3d_gz_micro_case_and_dispatch_preflights(item,tmp_path):
    """Exercise actual physical preparers, but never advance a single step."""
    folder=tmp_path/"never_written"
    case=coupled_optional_time_case(item,folder)
    assert case["calculation_kind"]=="DYNAMIC"
    assert case["geometry_model"]=="GREEN_LAGRANGE"
    assert case["scale_model"]=="MICRO"
    assert case["micro"]["gradient_model"]==item["gradient_model"]
    assert case["micro"]["enable_flexoelectric"] is item["flexoelectric_enabled"]
    assert case["method"]==item["method"]
    assert case["load_definition"]["kinematics"]=="DEAD_REFERENCE"
    nx=5 if item["theory"]=="KL" else 4
    assert case["load_definition"]["face"]==(
        "MIDDLE" if item["theory"]=="KL" else "OUTER")
    expected_p=(int(item["theory"][-1]) if
                item["theory"].startswith("CUF_TE") else 1)
    if item["flexoelectric_enabled"]:
        assert case["micro"]["potential_order"]==expected_p
    else:
        assert case["micro"]["potential_order"]==0
    if item["integrator"]=="GAUSS2_IRK4":
        from cylinder_gauss2_case import prepare_gauss2_mvi_case
        assert case["dynamics"].get("integrator") is None
        prepared,settings,mvi,initial,loads=prepare_gauss2_mvi_case(
            case,base_dir=tmp_path)
        assert prepared.method=="MVI"
        assert prepared.dynamics is not None
        assert mvi.rank==1
        assert settings.dt_s==case["dynamics"]["time_step_s"]
        assert initial.mechanical.shape==(prepared.problem.functional.n_mechanical,nx,4)
        for c in ((3-np.sqrt(3))/6,(3+np.sqrt(3))/6):
            mechanical,charge=loads(float(c*settings.dt_s))
            assert mechanical.shape==prepared.mechanical.shape
            if prepared.electrical is None:
                assert charge is None
            else:
                # The electrical field may be coupled with prescribed
                # electrode potential but with ZERO external free-charge
                # loading. In that case loads_at correctly returns None.
                assert charge is None or charge.shape==prepared.electrical.shape
                assert prepared.electrical.shape==(expected_p+1,nx,4)
    else:
        assert case["dynamics"]["integrator"]=="BATHE"
        prepared=prepare_reference_case(case,base_dir=tmp_path)
        assert prepared.method==item["method"]
        assert prepared.dynamics is not None
        assert prepared.dynamics.settings.n_steps==1
        assert prepared.fdm_backend=="DENSE_REFERENCE"
    assert prepared.problem.functional.theory==item["theory"]
    assert prepared.problem.functional.material.gradient_model==(
        item["gradient_model"].upper())
    assert prepared.problem.functional.electrical_on is item["flexoelectric_enabled"]
    assert prepared.electrical is None if not item["flexoelectric_enabled"] else (
        prepared.electrical.shape==(expected_p+1,nx,4))
    assert not folder.exists()
