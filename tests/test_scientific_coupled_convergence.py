"""Real G3D/Gz convergence study generation, physics invariance and evidence gates.

This module never advances a physical load or time step. All real
calculations are isolated in GitHub Actions after CASE preflight.
"""
from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from ci.scientific_coupled_convergence import (
    STUDIES, PILOT_IDS, catalog, case_from_entry,
    physical_identity, _probe, compare,
)


def test_pilot_and_full_cover_all_studies_with_nontrivial_pairs():
    full=catalog("full")
    pilot=catalog("pilot")
    assert len(full)==13
    assert len(pilot)==8
    assert len({e["id"] for e in full})==13
    assert {e["id"] for e in pilot}==set(PILOT_IDS)
    assert {e["study"] for e in pilot}==set(STUDIES)
    for study in STUDIES:
        assert len([e for e in pilot if e["study"]==study])==2
    with pytest.raises(ValueError,match="SCIENTIFIC_SCOPE_INVALID"):
        catalog("broad")


@pytest.mark.parametrize("entry",catalog("full"),ids=lambda e:e["id"])
def test_every_convergence_case_preflights_without_output(entry,tmp_path):
    out=tmp_path/entry["id"]
    case=case_from_entry(entry,out)
    assert case["scale_model"]=="MICRO"
    assert case["geometry_model"]=="GREEN_LAGRANGE"
    assert case["micro"]["ell_m"]>0
    assert case["micro"]["gradient_model"]==entry["gradient"]
    assert case["micro"]["enable_flexoelectric"] is entry["on"]
    assert case["micro"]["potential_order"]==(
        1 if entry["on"] else 0)
    assert case["load_definition"]["kinematics"]=="DEAD_REFERENCE"
    assert case["load_definition"]["face"]==(
        "MIDDLE" if entry["theory"]=="KL" else "OUTER")
    grid=case["fdm_grid" if entry["method"]=="FDM" else "mvi_grid"]
    assert (grid["nx"],grid["ns"])==(entry["nx"],entry["ns"])
    assert grid["operator_scheme"]=="LEGACY_REFERENCE"
    prepared=prepare_reference_case(case,base_dir=tmp_path)
    assert prepared.calculation_kind==entry["kind"]
    assert prepared.method==entry["method"]
    assert prepared.problem.functional.theory==entry["theory"]
    assert prepared.problem.functional.electrical_on==entry["on"]
    assert prepared.problem.functional.material.gradient_model==entry["gradient"].upper()
    if entry["method"]=="MVI":
        assert prepared.mvi_settings.rank==entry["rank"]
    if entry["kind"]=="DYNAMIC":
        assert prepared.dynamics.settings.n_steps==entry["steps"]
        assert np.isclose(prepared.dynamics.settings.dt_s*entry["steps"],.02)
    assert not out.exists()


def test_all_series_change_discretization_but_not_continuum_physics(tmp_path):
    for study in STUDIES:
        rows=[e for e in catalog("full") if e["study"]==study]
        signatures={physical_identity(case_from_entry(e,tmp_path/e["id"]))
                    for e in rows}
        assert len(signatures)==1,(study,signatures)


def test_flexoelectric_material_variation_changes_physical_signature(tmp_path):
    entry=catalog("pilot")[2]
    base=case_from_entry(entry,tmp_path/entry["id"])
    other=copy.deepcopy(base)
    other["micro"]["mu1_c_per_m"]+=.0002
    assert physical_identity(base)!=physical_identity(other)
    other=copy.deepcopy(base)
    other["geometry"]["R"]+=.01
    assert physical_identity(base)!=physical_identity(other)


def test_same_physical_probe_works_for_different_even_odd_grids():
    for nx in (4,5,6,7,9):
        for ns in (4,5,7,9):
            arr=np.full((nx,ns),3.25)
            assert _probe(arr,0.6,np.pi,1.2)==pytest.approx(3.25)
    field=np.zeros((5,5))
    field[2,:]=2.
    assert _probe(field,.6,np.pi,1.2)==pytest.approx(2.)


def test_incomplete_comparison_is_marked_failure_not_zero_data(tmp_path):
    location=tmp_path/"cases"
    location.mkdir()
    summary=tmp_path/"comparison"
    with pytest.raises(RuntimeError,match="FAILED_OR_MISSING"):
        compare(location,summary,scope="pilot")
    records=json.loads((summary/"SCIENTIFIC_COMPARISON.json").read_text())
    assert records["all_accepted"] is False
    assert all(r["status"]=="MISSING_OR_DUPLICATED" for r in records["records"])
    assert len(records["records"])==8
    assert (summary/"SCIENTIFIC_CONVERGENCE_DATA.csv").is_file()


def test_pretend_one_case_does_not_certify_entire_convergence_series(tmp_path):
    location=tmp_path/"cases"
    location.mkdir()
    first=catalog("pilot")[0]
    folder=location/first["id"]
    folder.mkdir()
    (folder/"CONVERGENCE_EVIDENCE.json").write_text(
        json.dumps(dict(case_id=first["id"],status="ACCEPTED",
                        study=first["study"],method=first["method"],
                        physics_sha256="same",observations=dict(w_mid_probe_m=1.))),
        encoding="utf-8")
    with pytest.raises(RuntimeError,match="FAILED_OR_MISSING"):
        compare(location,tmp_path/"summary",scope="pilot")


def _accepted_mvi_summary(*,full=False):
    status="CONVERGED_FULL" if full else "CONVERGED_PROJECTED"
    certificate={
        "solution_status":status,
        "accepted":True,"projected_accepted":True,
        "boundary_constraint_satisfied":True,
        "profile_reconstruction_satisfied":True,
        "projected_equilibrium_satisfied":True,
        "full_equilibrium_satisfied":full,
    }
    return {
        "status":status,"converged":True,"accepted_steps":1,
        "steps":[{
            "converged":True,"status":status,
            "full_equilibrium_satisfied":full,
            "mvi_certificate":certificate,
        }],
    }


def test_mvi_rank_limited_stable_solution_is_not_falsely_failed_or_certified():
    from ci.scientific_coupled_convergence import classify_accepted_steps
    entry=next(e for e in catalog("pilot") if e["id"]=="TE1_RANK_MVI_R1")
    qualification=classify_accepted_steps(
        _accepted_mvi_summary(full=False),entry)
    assert qualification["solution_class"]=="RANK_CONSTRAINED_ONLY"
    assert qualification["full_space_stationarity_confirmed"] is False
    assert qualification["fixed_rank_stationarity_confirmed"] is True
    assert qualification["spatial_rank_limit_detected"] is True


def test_mvi_full_solution_gets_distinct_explicit_stationarity_class():
    from ci.scientific_coupled_convergence import classify_accepted_steps
    entry=next(e for e in catalog("pilot") if e["id"]=="TE1_RANK_MVI_R1")
    qualification=classify_accepted_steps(
        _accepted_mvi_summary(full=True),entry)
    assert qualification["solution_class"]=="FULL_SPACE_STATIONARY"
    assert qualification["full_space_stationarity_confirmed"] is True


def test_mvi_rank_limited_cannot_masquerade_as_full_or_bad_certificate():
    from ci.scientific_coupled_convergence import classify_accepted_steps
    entry=next(e for e in catalog("pilot") if e["id"]=="TE1_RANK_MVI_R1")
    bad=_accepted_mvi_summary(full=False)
    bad["steps"][0]["mvi_certificate"]["full_equilibrium_satisfied"]=True
    with pytest.raises(ValueError,match="CERTIFICATE_INCONSISTENT"):
        classify_accepted_steps(bad,entry)
    bad=_accepted_mvi_summary(full=False)
    bad["steps"][0]["mvi_certificate"]["projected_accepted"]=False
    with pytest.raises(ValueError,match="CERTIFICATE_INCONSISTENT"):
        classify_accepted_steps(bad,entry)
    bad=_accepted_mvi_summary(full=False)
    bad["steps"][0]["status"]="CONVERGED_FULL"
    with pytest.raises(ValueError,match="CERTIFICATE_INCONSISTENT"):
        classify_accepted_steps(bad,entry)


def test_mvi_failed_step_not_allowed_even_if_status_looks_accepted():
    from ci.scientific_coupled_convergence import classify_accepted_steps
    entry=next(e for e in catalog("pilot") if e["id"]=="TE1_RANK_MVI_R1")
    bad=_accepted_mvi_summary(full=False)
    bad["steps"][0]["converged"]=False
    with pytest.raises(ValueError,match="STEPS_NOT_ACCEPTED"):
        classify_accepted_steps(bad,entry)


def test_comparison_marks_rank_constrained_study_without_publishing_it_as_full(tmp_path):
    from ci.scientific_coupled_convergence import compare,physical_identity
    location=tmp_path/"cases"
    location.mkdir()
    for entry in catalog("pilot"):
        folder=location/entry["id"]
        folder.mkdir()
        case=case_from_entry(entry,folder)
        rank_limited=entry["id"]=="TE1_RANK_MVI_R1"
        record=dict(
            case_id=entry["id"],status="ACCEPTED",
            study=entry["study"],method=entry["method"],
            physics_sha256=physical_identity(case),
            solution_class=("RANK_CONSTRAINED_ONLY" if rank_limited
                            else "FULL_SPACE_STATIONARY"),
            full_space_stationarity_confirmed=not rank_limited,
            observations=dict(w_mid_probe_m=1.))
        (folder/"CONVERGENCE_EVIDENCE.json").write_text(
            json.dumps(record),encoding="utf-8")
    report=compare(location,tmp_path/"summary",scope="pilot")
    assert report["all_accepted"] is True
    assert report["all_static_cases_full_space_stationary"] is False
    assert report["rank_constrained_cases"]==["TE1_RANK_MVI_R1"]
    assert report["series"]["TE1_RANK"]["rank_constrained_case_ids"]==[
        "TE1_RANK_MVI_R1"]
