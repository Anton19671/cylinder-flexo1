"""Strict energy-FDM CASE routing, preflight and toy runner contracts.

Only zero/manufactured algebraic work is used for an actual run: no
physical shell constitutive load case or time history is evaluated.
"""
import json

import numpy as np
import pytest

from tests.test_native_coupled_mkr_case_dispatch import make_case
from independent_mkr import run_energy_fdm_research as energy
from independent_mkr import run_coupled_research as strong
from cylinder_independent_dispatch import (
    independent_method,run_independent_case,validate_independent_case,
)
from cylinder_native_energy_fdm import NativeEnergyFDM,NativeEnergyWork


@pytest.mark.parametrize("kind,integrator",(
    ("STATIC",None),
    ("DYNAMIC","NEWMARK"),
    ("DYNAMIC","BATHE"),
    ("DYNAMIC","GAUSS_IRK4"),
))
@pytest.mark.parametrize("electric",(False,True))
def test_independent_energy_case_preflight_without_solve_or_output(
        tmp_path,kind,integrator,electric):
    path,case=make_case(tmp_path,theory="KL",kind=kind,
                        integrator=integrator,electric=electric)
    case["schema"]=energy.SCHEMA
    path.write_text(json.dumps(case),encoding="utf-8")
    parsed=energy.load_case(path)
    assert 0<parsed["space"].reduced_size<parsed["model"].shape[0] * (
        parsed["model"].grid.nx*parsed["model"].grid.ns+
        int(electric)*parsed["model"].grid.nx*parsed["model"].grid.ns)
    assert parsed["reduced_state_scale"].shape==(
        parsed["space"].reduced_size,)
    assert np.all(parsed["reduced_equation_scale"]>0.)
    answer=validate_independent_case(path)
    assert answer["schema"]==energy.SCHEMA
    assert answer["method"]==energy.METHOD
    assert answer["solver_backend"]=="INDEPENDENT_MKR"
    assert answer["spatial_method"]=="NATIVE_ENERGY_FDM_NO_MVI_SPATIAL_REUSE"
    assert answer["validated_only"]
    assert answer["physical_steps_computed"]==0
    assert answer["scientific_accuracy_certified"] is False
    assert not (tmp_path/"result").exists()


def test_energy_schema_does_not_silently_replace_direct_strong(tmp_path):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC")
    assert independent_method(case)=="INDEPENDENT_MKR"
    assert strong.load_case(path)["case"]["schema"]==strong.SCHEMA
    with pytest.raises(ValueError,match="SCHEMA"):
        energy.load_case(path)
    case["schema"]=energy.SCHEMA
    path.write_text(json.dumps(case),encoding="utf-8")
    assert independent_method(case)=="INDEPENDENT_MKR"
    with pytest.raises(ValueError,match="SCHEMA"):
        strong.load_case(path)
    assert energy.load_case(path)["case"]["schema"]==energy.SCHEMA


def test_native_energy_preflight_respects_nonuniform_physical_scales(tmp_path):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC")
    case["schema"]=energy.SCHEMA
    q=np.zeros((3,9,5))
    varying=np.broadcast_to(
        1.+np.linspace(0.,.5,9)[None,:,None],q.shape).copy()
    # Strict CASE stores scales as JSON arrays and validates them
    # independently of a chosen reduction or physical units.
    case["solver"]["mechanical_state_scale"]=varying.tolist()
    case["solver"]["mechanical_equation_scale"]=(5.*varying).tolist()
    path.write_text(json.dumps(case),encoding="utf-8")
    data=energy.load_case(path)
    assert np.all(data["reduced_state_scale"]>0.)
    assert np.all(np.isfinite(data["reduced_equation_scale"]))
    assert np.max(data["reduced_state_scale"])>np.min(
        data["reduced_state_scale"])


@pytest.mark.parametrize("integrator",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_run_native_energy_drivers_on_zero_manufactured_work_only(
        tmp_path,integrator,monkeypatch):
    path,case=make_case(tmp_path,theory="KL",kind="DYNAMIC",
                        integrator=integrator,electric=False)
    case["schema"]=energy.SCHEMA
    case["solver"]["steps"]=1
    path.write_text(json.dumps(case),encoding="utf-8")
    seen=[]
    def toy_work(self,q,p,*,acceleration=None,velocity=None,
                 mechanical_load=None,electrical_charge=None,**kwargs):
        assert p is None
        assert velocity is not None and acceleration is not None
        seen.append(1)
        return NativeEnergyWork(
            acceleration+velocity+q,None)
    monkeypatch.setattr(NativeEnergyFDM,"balance",toy_work)
    summary=run_independent_case(path)
    assert seen
    assert summary["method"]==energy.METHOD
    assert summary["status"]=="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
    assert summary["accepted_steps"]==1
    assert summary["physical_steps_computed"]==1
    with np.load(tmp_path/"result"/"STEP_0001.npz") as state:
        np.testing.assert_array_equal(state["mechanical"],0.)
        np.testing.assert_array_equal(state["acceleration"],0.)
    assert (tmp_path/"result"/"STEP_0000.npz").exists()


def test_static_energy_runner_accepts_toy_reduced_zero_equilibrium(
        tmp_path,monkeypatch):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC")
    case["schema"]=energy.SCHEMA
    path.write_text(json.dumps(case),encoding="utf-8")
    def toy_work(self,q,p,**kwargs):
        assert p is None
        return NativeEnergyWork(np.asarray(q,float),None)
    monkeypatch.setattr(NativeEnergyFDM,"balance",toy_work)
    summary=run_independent_case(path)
    assert summary["status"]=="EXPERIMENTAL_STATIONARY_UNCERTIFIED"
    assert summary["accepted_steps"]==1
    with np.load(tmp_path/"result"/"STATIC_ACCEPTED.npz") as f:
        np.testing.assert_array_equal(f["mechanical"],0.)
