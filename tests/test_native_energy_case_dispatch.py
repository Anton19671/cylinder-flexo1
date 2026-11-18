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


@pytest.mark.parametrize("field,amplitude",(
    ("mechanical",1.e-12),("velocity",1.e-12),("acceleration",1.),
))
def test_energy_preflight_rejects_incompatible_initial_supports_without_solving(
        tmp_path,monkeypatch,field,amplitude):
    path,case=make_case(tmp_path,theory="KL",kind="DYNAMIC")
    case["schema"]=energy.SCHEMA
    path.write_text(json.dumps(case),encoding="utf-8")
    with np.load(tmp_path/"initial.npz") as archive:
        data={key:archive[key].copy() for key in archive.files}
    data.setdefault(field,np.zeros_like(data["mechanical"]))
    data[field][0,0,0]=amplitude
    np.savez_compressed(tmp_path/"initial.npz",**data)
    def forbidden(*args,**kwargs):
        pytest.fail("preflight must not evaluate the constitutive law or solve")
    monkeypatch.setattr(NativeEnergyFDM,"balance",forbidden)
    with pytest.raises(ValueError,match="NATIVE_ENERGY_INCOMPATIBLE"):
        validate_independent_case(path)
    assert not (tmp_path/"result").exists()


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
    assert summary["steps"]==1
    assert summary["requested_steps"]==1
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
    # The saved case is portable after the original inputs are removed.
    (tmp_path/"initial.npz").unlink()
    (tmp_path/"loads.npz").unlink()
    replay=energy.load_case(tmp_path/"result"/"CASE.json")
    assert replay["output"]==tmp_path/"result"/"replay_output"
    manifest=json.loads((tmp_path/"result"/"INPUT_MANIFEST.json").read_text())
    assert set(manifest["input_sha256"])=={
        "INPUT_INITIAL.npz","INPUT_LOADS.npz","CASE.json"}
    assert (tmp_path/"result"/"ORIGINAL_CASE.json").exists()



@pytest.mark.parametrize("integrator,multiplier",(
    (None,1),("GAUSS_IRK4",2),
))
def test_energy_dof_budget_is_reduced_not_old_full_strong_budget(
        tmp_path,integrator,multiplier):
    kind="STATIC" if integrator is None else "DYNAMIC"
    path,case=make_case(tmp_path,theory="KL",kind=kind,
                        integrator=integrator,electric=False)
    case["schema"]=energy.SCHEMA
    # Three KL fields, nx=9, ns=5, two fixed ends per field.
    reduced=3*(9-2)*5*multiplier
    full=3*9*5*multiplier
    assert reduced<full
    case["solver"]["max_dofs"]=reduced
    path.write_text(json.dumps(case),encoding="utf-8")
    parsed=energy.load_case(path)
    assert parsed["settings"].max_dofs==reduced
    case["solver"]["max_dofs"]=reduced-1
    path.write_text(json.dumps(case),encoding="utf-8")
    with pytest.raises(ValueError,match="REDUCED_DOF_BUDGET_TOO_SMALL"):
        energy.load_case(path)


def test_energy_case_source_is_strictly_disjoint_from_mvi_numeric_backend():
    import ast
    from pathlib import Path
    for module in (energy,):
        root=ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports=[]
        for node in ast.walk(root):
            if isinstance(node,ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(p.startswith((
            "independent_mvi","cylinder_variational_mvi",
            "cylinder_sbp_green_reduction","cylinder_reference_case",
        )) for p in imports)


@pytest.mark.parametrize("electric",(False,True))
def test_equation_density_scales_follow_area_without_false_zero_convergence(
        tmp_path,electric):
    """Algebraic virtual load work only; no physical shell is solved."""
    norms=[]
    for index,factor in enumerate((1.,1.e-3,1.e-6)):
        directory=tmp_path/str(index)
        directory.mkdir()
        path,case=make_case(directory,theory="KL",kind="STATIC",electric=electric)
        case["schema"]=energy.SCHEMA
        case["grid"]["length_m"]*=factor
        case["grid"]["radius_m"]*=factor
        case["physics"]["thickness_m"]*=factor
        case["physics"]["material"]["ell_m"]*=factor
        case["solver"]["mechanical_equation_scale"]=1.e4
        if electric:
            case["solver"]["electrical_equation_scale"]=1.e4
        path.write_text(json.dumps(case),encoding="utf-8")
        data=energy.load_case(path)
        model=data["model"]
        force=np.full(model.shape,100.)
        coefficients=np.zeros((len(model.derivative_orders),*model.shape))
        ec=np.zeros((3,1,*model.shape[1:])) if electric else None
        charge=np.full((1,*model.shape[1:]),100.) if electric else None
        work=data["energy"].jet_action(
            coefficients,ec,mechanical_load=force,electrical_charge=charge)
        residual=data["space"].residual(work)/data["reduced_equation_scale"]
        norms.append(np.max(np.abs(residual)))
        assert norms[-1]>data["settings"].scaled_absolute_tolerance
        np.testing.assert_allclose(np.abs(residual),.01,rtol=2.e-14)
    np.testing.assert_allclose(norms,norms[0],rtol=2.e-14)


@pytest.mark.parametrize("factor",(1.e-200,1.e200))
def test_reduced_scale_projection_does_not_square_dimensional_extremes(tmp_path,factor):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC")
    case["schema"]=energy.SCHEMA
    path.write_text(json.dumps(case),encoding="utf-8")
    data=energy.load_case(path)
    value=energy._norm_projection(
        data["space"],np.full(data["model"].shape,factor),None,"TEST")
    np.testing.assert_allclose(value/factor,1.,rtol=1.e-14)


def test_small_geometry_toy_loaded_static_run_requires_a_nonzero_solution(tmp_path,monkeypatch):
    """Synthetic Hooke spring work only, not a physical shell solve."""
    path,case=make_case(tmp_path,theory="KL",kind="STATIC")
    case["schema"]=energy.SCHEMA
    case["grid"].update(length_m=1.2e-6,radius_m=2.e-6)
    case["physics"]["thickness_m"]=1.e-7
    case["physics"]["material"]["ell_m"]=1.e-8
    case["solver"].update(mechanical_state_scale=1.e-4,
                         mechanical_equation_scale=1.e4)
    path.write_text(json.dumps(case))
    def toy_work(self,q,p,**kwargs):
        return NativeEnergyWork(self.weight[None,:,:]*(1.e6*q-100.),None)
    monkeypatch.setattr(NativeEnergyFDM,"balance",toy_work)
    summary=energy.run(path,validate_only=False)
    assert summary["static_nonlinear_diagnostics"]["iterations"]>0
    with np.load(tmp_path/"result"/"STATIC_ACCEPTED.npz") as f:
        np.testing.assert_allclose(f["mechanical"][:,1:-1,:],1.e-4,rtol=1.e-7)
