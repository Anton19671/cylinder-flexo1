"""Static generation and scientific evidence passport tests: never solve physics."""
from pathlib import Path
import pytest
from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from ci.scientific_real_coupled_case import (catalog,build_case,
    expected_accepted_summary,_physical_radial_response)
import numpy as np

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
    if item["kind"]=="STATIC" and item["method"]=="MVI":
        assert "status" not in expected
    else:
        assert expected["status"]=="CONVERGED"
    assert expected["converged"] is True
    assert expected["accepted_steps"]==1


def test_kl_clamped_short_axial_grid_cannot_be_accepted_as_physical_bending(tmp_path):
    entry=next(x for x in catalog() if x["theory"]=="KL")
    c=build_case(entry,tmp_path/"KL")
    assert c["fdm_grid"]=={"nx":5,"ns":4}
    zeros=np.zeros((3,5,4))
    with pytest.raises(AssertionError,match="PHYSICAL_ZERO_RADIAL_RESPONSE"):
        _physical_radial_response(c,zeros)
    zeros[2,2,1]=1.e-12
    assert _physical_radial_response(c,zeros)==pytest.approx(1.e-12)


def test_cuf_real_pilot_preserves_small_grid_and_radial_component_order(tmp_path):
    entry=next(x for x in catalog() if x["theory"]=="CUF_TE4")
    c=build_case(entry,tmp_path/"TE4")
    assert c["fdm_grid"]=={"nx":4,"ns":4}
    trial=np.zeros((15,4,4))
    trial[10,2,2]=3.e-10
    assert _physical_radial_response(c,trial)==pytest.approx(3.e-10)
    trial[10,2,2]=0.
    trial[0,2,2]=1.e-3
    with pytest.raises(AssertionError,match="PHYSICAL_ZERO_RADIAL_RESPONSE"):
        _physical_radial_response(c,trial)

@pytest.mark.parametrize("theory,order",[
    ("CUF_TE1",1),("CUF_TE2",2),("CUF_TE3",3),("CUF_TE4",4)
])
@pytest.mark.parametrize("kind,method",[
    ("STATIC","FDM"),("DYNAMIC","MVI")
])
def test_full_electric_thickness_order_is_exercised_not_template_default(
    theory,order,kind,method,tmp_path
):
    entry=dict(theory=theory,gradient="G3D",on=True,
               kind=kind,method=method)
    case=build_case(entry,tmp_path/"no_outputs")
    assert case["micro"]["potential_order"]==order
    prepared=prepare_reference_case(case)
    assert prepared.electrical.shape==(order+1,4,4)
    assert prepared.problem.functional.electrical_on
    assert not (tmp_path/"no_outputs").exists()


@pytest.mark.parametrize("order",range(1,5))
def test_flexoelectric_off_retains_gradient_but_no_electric_dofs(order,tmp_path):
    entry=dict(theory=f"CUF_TE{order}",gradient="Gz",
               on=False,kind="STATIC",method="FDM")
    case=build_case(entry,tmp_path/"no_outputs")
    assert case["micro"]["potential_order"]==0
    prepared=prepare_reference_case(case)
    assert prepared.electrical is None
    assert prepared.problem.functional.material.ell_m>0
    assert prepared.problem.functional.material.gradient_model=="GZ"


def _synthetic_one_step_summary(status, *, full):
    """Fake public metadata only; it never invokes a physical solver."""
    return {
        "theory":"FSDT","method":"MVI","calculation_kind":"STATIC",
        "gradient_model":"Gz","electrical_enabled":False,
        "active_strain_gradient_components":6,
        "converged":True,"accepted_steps":1,"status":status,
        "steps":[{
            "status":status,"converged":True,
            "state_file":"REFERENCE_STEP_0001.npz",
            "full_equilibrium_satisfied":full,
            "mvi_certificate":{
                "accepted":True,"projected_accepted":True,
                "solution_status":status,
                "projected_equilibrium_satisfied":True,
                "full_equilibrium_satisfied":full,
                "boundary_constraint_satisfied":True,
                "profile_reconstruction_satisfied":True,
            },
        }],
    }


@pytest.mark.parametrize("status,full,expected_class",[
    ("CONVERGED_PROJECTED",False,"RANK_CONSTRAINED_ONLY"),
    ("CONVERGED_FULL",True,"FULL_SPACE_STATIONARY"),
])
def test_real_112_case_qualifier_accepts_only_consistent_static_mvi_certificate(
    status,full,expected_class
):
    from ci.scientific_acceptance import classify_accepted_steps
    accepted=classify_accepted_steps(
        _synthetic_one_step_summary(status,full=full),
        dict(method="MVI",kind="STATIC",steps=1))
    assert accepted["solution_class"]==expected_class
    assert accepted["full_space_stationarity_confirmed"] is full
    assert accepted["fixed_rank_stationarity_confirmed"] is True


@pytest.mark.parametrize("change",[
    ("converged",False),
    ("accepted_steps",0),
    ("status","FAILED"),
])
def test_real_campaign_rejects_false_top_level_acceptance(change):
    from ci.scientific_acceptance import classify_accepted_steps
    bad=_synthetic_one_step_summary("CONVERGED_PROJECTED",full=False)
    bad[change[0]]=change[1]
    with pytest.raises(ValueError):
        classify_accepted_steps(bad,dict(method="MVI",kind="STATIC",steps=1))


@pytest.mark.parametrize("certificate_field,forged",[
    ("accepted",False),
    ("projected_accepted",False),
    ("full_equilibrium_satisfied",True),
    ("boundary_constraint_satisfied",False),
    ("profile_reconstruction_satisfied",False),
    ("projected_equilibrium_satisfied",False),
])
def test_real_campaign_rejects_inconsistent_finite_rank_certificate(
    certificate_field,forged
):
    from ci.scientific_acceptance import classify_accepted_steps
    bad=_synthetic_one_step_summary("CONVERGED_PROJECTED",full=False)
    bad["steps"][0]["mvi_certificate"][certificate_field]=forged
    with pytest.raises(ValueError,match="CERTIFICATE_INCONSISTENT"):
        classify_accepted_steps(bad,dict(method="MVI",kind="STATIC",steps=1))


def test_mvi_summary_expectations_do_not_force_full_space_stationarity(tmp_path):
    spec=dict(theory="FSDT",gradient="Gz",on=False,
              kind="STATIC",method="MVI")
    case=build_case(spec,tmp_path/"unused")
    metadata=expected_accepted_summary(case)
    assert "status" not in metadata
