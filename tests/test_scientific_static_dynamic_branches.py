"""Static + dynamic branch matrix for seven coupled cylinder theories.

Scientific coverage: 7 theories x 2 gradient models x 2 flexo switches,
with both stationary variational first variations and dynamic
inertia/viscosity actions. A second parametrization checks that both
FDM and rank-one MVI CASE routes accept these combinations for STATIC
and DYNAMIC, or fails explicitly on a regression.

All cases use *production-allowed* MICRO + GREEN_LAGRANGE. LINEAR and
VON_KARMAN are valid local-kernel research options but currently fail
MICRO CASE validation by design. No grid convergence, physical time
history, or independent coupled FDM/MVI accuracy is claimed here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_paths import activate
activate()

from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_functional_assembly import CylinderFunctional, surface_sbp21
from cylinder_reference_case import prepare_reference_case
from flexo_gradient_g3d import G3DMaterial
from flexo_scale_policy import micro_material_from_case
from ci.scientific_real_coupled_case import build_case as build_canonical_micro_case


THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
GRADIENTS=("G3D","Gz")
SWITCHES=(False,True)
KINDS=("STATIC","DYNAMIC")
METHODS=("FDM","MVI")
ROOT=Path(__file__).resolve().parents[1]
SURFACE=surface_sbp21(4,5,1.2,2.)


def _material(mode,on):
    return G3DMaterial(100.,.25,.08,
                       .001 if on else 0.,
                       -.002 if on else 0.,
                       1.5 if on else 0.,
                       gradient_model=mode)


def _integrated_points(theory):
    return 2*int(theory[-1])+1 if theory.startswith("CUF_TE") else (
        7 if theory=="REDDY" else 3)


def _physical_problem(theory,mode,on,kind):
    f=CylinderFunctional(
        theory,_material(mode,on),radius_m=2.,thickness_m=.2,
        surface_operators=SURFACE,n_gauss=_integrated_points(theory),
        geometry_model="GREEN_LAGRANGE")
    bc={"mechanical":{
        "left":{"support":"CLAMPED","higher":"NATURAL"},
        "right":{"traction":[.005]*f.n_mechanical,
                 "spring":[.03]*f.n_mechanical}},
        "electrical":({"inner":{"mode":"POTENTIAL","value":0.}} if on else {})}
    return BoundarySpatialProblem(f,bc_spec=bc,calculation_kind=kind)


def _example_fields(f,on,seed):
    rng=np.random.default_rng(seed)
    q=rng.normal(0.,2.e-5,(f.n_mechanical,*SURFACE.shape))
    electric_count=(int(f.theory[-1])+1
                    if f.theory.startswith("CUF_TE") else 2)
    p=rng.normal(0.,8.e-4,(electric_count,*SURFACE.shape)) if on else None
    dq=rng.normal(0.,7.e-4,q.shape)
    dp=rng.normal(0.,4.e-3,p.shape) if on else None
    v=rng.normal(0.,4.e-4,q.shape)
    a=rng.normal(0.,.002,q.shape)
    return q,p,dq,dp,v,a


def _energy(problem,q,p):
    energy,residual,_=problem.evaluate_effective_residual(q,p)
    return energy,residual


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("mode",GRADIENTS)
@pytest.mark.parametrize("on",SWITCHES)
def test_static_energy_and_dynamic_inertia_damping_all_physics_branches(
    theory,mode,on
):
    """28 material branches, each checked in STATIC and DYNAMIC."""
    static=_physical_problem(theory,mode,on,"STATIC")
    dynamic=_physical_problem(theory,mode,on,"DYNAMIC")
    f=static.functional
    assert f.theory==theory
    assert f.material.gradient_model==mode.upper()
    assert f.material.ell_m>0
    assert f.electrical_on is on
    q,p,dq,dp,v,a=_example_fields(
        f,on,735+THEORIES.index(theory)*5+GRADIENTS.index(mode)*2+int(on))
    fields=f.evaluate(q,p,energy_breakdown=True,
                      check_charge_balance=False)
    assert fields.energy_parts_j["gradient"]>=0.
    if on:
        assert fields.electrical_residual.shape==p.shape
        assert np.isfinite(fields.energy_parts_j["flexoelectric"])
    else:
        assert fields.electrical_residual is None
        assert fields.energy_parts_j["flexoelectric"]==0.
        assert fields.energy_parts_j["dielectric"]==0.

    value,derivative=_energy(static,q,p)
    h=1.e-4
    ep=_energy(static,q+h*dq,None if p is None else p+h*dp)[0]
    em=_energy(static,q-h*dq,None if p is None else p-h*dp)[0]
    variation=float(np.sum(derivative[:q.size]*dq.ravel()))
    if on:
        variation+=float(np.sum(derivative[q.size:]*dp.ravel()))
    np.testing.assert_allclose((ep-em)/(2*h),variation,
                               rtol=1.e-5,atol=2.e-7)
    assert np.isfinite(value)

    # The dynamic physical operator is the same stationary functional
    # plus mechanical inertia, mass damping and Kelvin-Voigt viscosity.
    # The electric field has neither velocity nor inertia.
    _,static_at_t,_=dynamic.evaluate_effective_residual(
        q,p,time_s=.02,acceleration=np.zeros_like(q),
        density_kg_m3=1300.)
    _,loaded,_=dynamic.evaluate_effective_residual(
        q,p,time_s=.02,velocity=v,acceleration=a,
        density_kg_m3=1300.,mass_damping_rate_per_s=.03,eta_s=.002)
    correction=(
        f.mass_action(a,density_kg_m3=1300.,
                      mechanical=q,velocity=v)
        +f.mass_proportional_damping_action(
            v,density_kg_m3=1300.,
            mass_damping_rate_per_s=.03,mechanical=q)
        +f.kelvin_voigt_action(q,v,eta_s=.002).mechanical_force)
    np.testing.assert_allclose(
        loaded[:q.size]-static_at_t[:q.size],correction.ravel(),
        rtol=2.e-9,atol=2.e-8)
    if on:
        np.testing.assert_allclose(loaded[q.size:],static_at_t[q.size:],
                                   rtol=0.,atol=2.e-9)
    assert np.isfinite(loaded).all()


def _case(theory,mode,on,kind,method):
    # Match the production physical pilot exactly, including TE-N
    # potential_order=N and KL nondegenerate five-axial-node trial space.
    return build_canonical_micro_case(
        dict(theory=theory,gradient=mode,on=on,kind=kind,method=method),
        ROOT/"SCIENTIFIC_BRANCH_PRECHECK_NO_WRITE")


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("mode",GRADIENTS)
@pytest.mark.parametrize("on",SWITCHES)
@pytest.mark.parametrize("kind",KINDS)
@pytest.mark.parametrize("method",METHODS)
def test_static_dynamic_case_branch_validation_without_result_writes(
    theory,mode,on,kind,method,tmp_path
):
    """112 actual CASE preflights, no accepted time or load step is solved."""
    cfg=_case(theory,mode,on,kind,method)
    cfg["output_dir"]=str(tmp_path/"not_created")
    prepared=prepare_reference_case(cfg,base_dir=ROOT/"config")
    assert prepared.calculation_kind==kind
    assert prepared.method==method
    assert prepared.problem.functional.theory==theory
    assert prepared.problem.functional.material.gradient_model==mode.upper()
    assert prepared.problem.functional.electrical_on is on
    assert (prepared.electrical is None) is (not on)
    grid=cfg["fdm_grid" if method=="FDM" else "mvi_grid"]
    nx_expected=5 if theory=="KL" else 4
    assert (grid["nx"],grid["ns"])==(nx_expected,4)
    assert cfg["load_definition"]["face"]==(
        "MIDDLE" if theory=="KL" else "OUTER")
    assert cfg["load_definition"]["kinematics"]=="DEAD_REFERENCE"
    expected_order=(int(theory[-1]) if theory.startswith("CUF_TE")
                    else 1) if on else 0
    assert cfg["micro"]["potential_order"]==expected_order
    if on:
        assert prepared.electrical.shape==(expected_order+1,nx_expected,4)
    else:
        assert prepared.electrical is None
    assert prepared.problem.kind==kind
    if kind=="DYNAMIC":
        assert prepared.dynamics is not None
        assert prepared.initial_acceleration.shape==prepared.mechanical.shape
        assert np.isfinite(prepared.initial_acceleration).all()
    else:
        assert prepared.dynamics is None
    assert not (tmp_path/"not_created").exists()


@pytest.mark.parametrize("geometry",("LINEAR","VON_KARMAN"))
@pytest.mark.parametrize("kind",KINDS)
@pytest.mark.parametrize("method",METHODS)
def test_micro_case_rejects_non_gl_geometry_without_silent_fallback(
    geometry,kind,method
):
    cfg=_case("CUF_TE2","G3D",True,kind,method)
    cfg["geometry_model"]=geometry
    cfg["mode"]="LINEAR_ELASTIC" if geometry=="LINEAR" else "NONLINEAR_ELASTIC"
    with pytest.raises(ValueError,match="full Green-Lagrange"):
        micro_material_from_case(cfg)
    with pytest.raises(ValueError,match="full Green-Lagrange"):
        prepare_reference_case(cfg,base_dir=ROOT/"config")
