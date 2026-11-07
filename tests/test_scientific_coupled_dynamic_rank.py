"""Physical dynamic CUF TE1 rank 1..8 CASE and history integrity tests.

No actual time integration is run here. GitHub Actions physical job
is the only place that invokes real_newmark solver.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case
from scientific_coupled_dynamic_rank import (
    catalog,build_case,compare,accepted_time_states,NX,NS,DT,STEPS,
)
from scientific_coupled_convergence import physical_identity,digest


def test_dynamic_mvi_1_to_8_plus_fdm_and_bounded_pilot():
    assert len(catalog("full"))==9
    assert len(catalog("pilot"))==4
    assert len({x["id"] for x in catalog("full")})==9
    assert [x["rank"] for x in catalog("full")[1:]]==list(range(1,9))
    assert [x["rank"] for x in catalog("pilot")[1:]]==[1,2,3]
    with pytest.raises(ValueError,match="SCOPE_INVALID"):
        catalog("all")


@pytest.mark.parametrize("entry",catalog("full"),ids=lambda x:x["id"])
def test_real_dynamic_cases_preserve_matching_physics_and_fixed_rank(entry,tmp_path):
    folder=tmp_path/entry["id"]
    case=build_case(entry,folder)
    assert case["calculation_kind"]=="DYNAMIC"
    assert case["geometry_model"]=="GREEN_LAGRANGE"
    assert case["micro"]["gradient_model"]=="Gz"
    assert case["micro"]["enable_flexoelectric"] is True
    assert case["micro"]["potential_order"]==1
    assert case["load_definition"]["type"]=="PATCH_RADIAL_FACE"
    assert case["load_definition"]["face"]=="OUTER"
    assert case["load_definition"]["kinematics"]=="DEAD_REFERENCE"
    assert case["dynamics"]["n_steps"]==2
    assert case["dynamics"]["time_step_s"]==.01
    assert case["dynamics"].get("integrator") is None
    assert case["reference_newton"]["max_free_dofs"]==512
    grid=case["fdm_grid" if entry["method"]=="FDM" else "mvi_grid"]
    assert (grid["nx"],grid["ns"])==(NX,NS)
    assert grid["operator_scheme"]=="LEGACY_REFERENCE"
    prepared=prepare_reference_case(case,base_dir=tmp_path)
    assert prepared.calculation_kind=="DYNAMIC"
    assert prepared.method==entry["method"]
    assert prepared.problem.functional.theory=="CUF_TE1"
    assert prepared.electrical.shape==(2,8,8)
    assert prepared.dynamics.settings.n_steps==STEPS
    assert prepared.dynamics.settings.dt_s==DT
    if entry["method"]=="MVI":
        assert prepared.mvi_settings.rank==entry["rank"]
    assert not folder.exists()


def test_all_dynamic_ranks_identical_continuum_physics(tmp_path):
    signatures={physical_identity(build_case(e,tmp_path/e["id"]))
                for e in catalog("full")}
    assert len(signatures)==1


def test_incomplete_dynamic_run_cannot_claim_comparison(tmp_path):
    with pytest.raises(RuntimeError,match="INCOMPLETE"):
        compare(tmp_path,tmp_path/"tables","pilot")
    data=json.loads((tmp_path/"tables"/"DYNAMIC_MVI_VS_FDM.json").read_text())
    assert data["accepted_cases"]==0
    assert data["all_requested_ranks_compared"] is False
    assert len(data["failed_or_missing"])==4
    assert not data["physical_accuracy_certified"]


def test_dynamic_history_requires_all_real_steps_and_valid_hashes(tmp_path):
    root=tmp_path/"case"
    output=root/"RESULTS"
    output.mkdir(parents=True)
    base={"status":"CONVERGED","converged":True,
          "requested_steps":STEPS,"accepted_steps":STEPS,
          "electric_inertia":False,
          "steps":[{"converged":True,"time_s":i*DT,
                    "state_file":f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz"}
                   for i in range(1,STEPS+1)]}
    (output/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(base))
    fake={"state_sha256":"no_such_hash"}
    with pytest.raises(ValueError,match="STEP_FILE_MISSING"):
        accepted_time_states(fake,root)
    base["accepted_steps"]=1
    (output/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(base))
    with pytest.raises(ValueError,match="INCOMPLETE_TIME_HISTORY"):
        accepted_time_states(fake,root)
