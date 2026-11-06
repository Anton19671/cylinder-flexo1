"""Static generation and scientific evidence passport tests: never solve physics."""
from pathlib import Path
import pytest
from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from scientific_real_coupled_case import catalog,build_case,expected_accepted_summary

def test_full_and_pilot_scientific_matrices_unique():
    full=catalog(full=True)
    pilot=catalog()
    assert len(full)==112 and len(pilot)==8
    ids={item["id"] for item in full}
    assert len(ids)==112 and all(item["id"] in ids for item in pilot)
    assert {item["theory"] for item in pilot}=={
        "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"}
    assert {item["kind"] for item in pilot}=={"STATIC","DYNAMIC"}
    assert {item["gradient"] for item in pilot}=={"G3D","Gz"}
    assert {item["method"] for item in pilot}=={"FDM","MVI"}

@pytest.mark.parametrize("item",catalog())
def test_real_pilot_cases_preflight_without_simulation_or_output(item,tmp_path):
    folder=tmp_path/"unwritten"
    c=build_case(item,folder)
    assert c["load_definition"]["face"]==("MIDDLE" if item["theory"]=="KL" else "OUTER")
    assert c["load_definition"]["kinematics"]=="DEAD_REFERENCE"
    assert c["geometry_model"]=="GREEN_LAGRANGE"
    assert c["scale_model"]=="MICRO"
    prepared=prepare_reference_case(c)
    assert prepared.problem.functional.theory==item["theory"]
    assert prepared.problem.functional.material.gradient_model==item["gradient"].upper()
    assert prepared.problem.functional.electrical_on is item["on"]
    assert prepared.calculation_kind==item["kind"]
    assert prepared.method==item["method"]
    assert not folder.exists()


@pytest.mark.parametrize("item",catalog())
def test_scientific_summary_expectations_match_public_solver_labels(item,tmp_path):
    case=build_case(item,tmp_path/"unwritten")
    expected=expected_accepted_summary(case)
    assert expected["theory"]==item["theory"]
    assert expected["gradient_model"]==item["gradient"]
    assert expected["active_strain_gradient_components"]==(
        18 if item["gradient"]=="G3D" else 6)
    assert expected["status"]==(
        "CONVERGED_FULL" if item["kind"]=="STATIC" and item["method"]=="MVI"
        else "CONVERGED")
    assert expected["converged"] is True
    assert expected["accepted_steps"]==1
