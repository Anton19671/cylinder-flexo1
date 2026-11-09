"""Pure preflight and dispatch tests for the opt-in coupled MVI CASE.

Tests create only synthetic input NPZ files and call validate-only.
No physical nonlinear material solve, loaded shell or time history runs.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as native
from cylinder_independent_dispatch import validate_independent_case


def make_case(tmp_path,*,theory="CUF_TE1",electric=False,kind="DYNAMIC",
              integrator="NEWMARK"):
    n=3*(int(theory[-1])+1) if theory.startswith("CUF_TE") else (
        3 if theory=="KL" else 5)
    nx=9;ns=5; rank=1
    ec=1 if electric else 0
    x=np.zeros((n+ec,rank,nx))
    x[:,:,1:-1]=.001
    y=np.ones((n+ec,rank,ns))
    initial=tmp_path/"initial.npz"
    np.savez_compressed(initial,x_profiles=x,y_profiles=y,
                        **({"velocity":np.zeros((n,nx,ns))}
                           if kind=="DYNAMIC" else {}))
    loads=tmp_path/"loads.npz"
    np.savez_compressed(loads,mechanical_load=np.zeros((n,nx,ns)),
                        **({"free_charge":np.zeros((ec,nx,ns))}
                           if electric else {}))
    case=dict(schema=native.SCHEMA,research_opt_in=True,
              physics=dict(theory=theory,geometry_model="LINEAR",
                material=dict(young_pa=1.e9,poisson=.3,ell_m=1.e-6,
                  mu1_c_per_m=1.e-6 if electric else 0.,
                  mu2_c_per_m=1.e-6 if electric else 0.,
                  kappa_f_per_m=1.e-8 if electric else 0.,
                  gradient_model="G3D"),
                thickness_m=.1,density_kg_m3=2500.,
                eta_s=0.,mass_damping_rate_per_s=0.,
                n_gauss=2,potential_order=0 if electric else None),
              grid=dict(nx=nx,ns=ns,length_m=1.,radius_m=2.),
              rank=rank,boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                  electric="GROUNDED_BOTH" if electric else "OFF"),
              solver=dict(kind=kind,
                 integrator=integrator if kind=="DYNAMIC" else None,
                 dt_s=.01 if kind=="DYNAMIC" else None,
                 steps=2 if kind=="DYNAMIC" else 0,
                 max_sweeps=4,max_acceleration_coordinates=256,
                 field_scales=[1.]* (n+ec),energy_scale_j=1.),
              initial_state_npz="initial.npz",loads_npz="loads.npz",
              output_dir="output")
    file=tmp_path/"case.json"
    file.write_text(json.dumps(case),encoding="utf8")
    return file,case


@pytest.mark.parametrize("theory",native.THEORIES)
@pytest.mark.parametrize("integrator",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_all_seven_coupled_theories_validate_only_without_output(
    tmp_path,theory,integrator):
    case,_=make_case(tmp_path,theory=theory,integrator=integrator)
    result=validate_independent_case(case)
    assert result["valid"] and result["validated_only"]
    assert result["solver_backend"]=="INDEPENDENT_MVI"
    assert result["integrator"]==integrator
    assert result["physical_steps_computed"]==0
    assert result["scientific_accuracy_certified"] is False
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("electric", (False,True))
@pytest.mark.parametrize("kind", ("STATIC","DYNAMIC"))
def test_coupled_opt_in_case_accepts_explicit_physics_but_never_silently_solves(
    tmp_path,electric,kind):
    path,_=make_case(tmp_path,electric=electric,kind=kind)
    result=native.run(path,validate_only=True)
    assert result["calculation_kind"]==kind
    assert result["physical_boundary_closure_certified"] is False
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("key,value,error",[
    ("research_opt_in",False,"RESEARCH_OPT_IN_REQUIRED"),
    ("schema","INDEPENDENT_MVI_SEVEN_MODELS_V1","SCHEMA_MISMATCH"),
    ("rank",9,"FIXED_RANK"),
    ("output_dir",".","UNSAFE_OUTPUT_DIRECTORY"),
])
def test_opt_in_and_invalid_case_fail_closed(tmp_path,key,value,error):
    path,document=make_case(tmp_path)
    document[key]=value
    path.write_text(json.dumps(document),encoding="utf8")
    with pytest.raises(ValueError,match=error):
        native.run(path,validate_only=True)
    assert not (tmp_path/"output").exists()


def test_reference_schema_never_accepted_as_coupled_research(tmp_path):
    path,_=make_case(tmp_path)
    data=json.loads(path.read_text())
    data["schema"]="VARIATIONAL_REFERENCE"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="UNSUPPORTED_SCHEMA"):
        validate_independent_case(path)
