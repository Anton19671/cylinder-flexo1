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
                 max_sweeps=4,max_acceleration_coordinates=1024,
                 max_halfstep_coordinates=512,
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


def test_high_rank_preflight_is_explicit_and_no_solver_starts(tmp_path):
    path,document=make_case(tmp_path,theory="CUF_TE4")
    rank=8
    nx=11
    ns=9
    nfields=15
    document["grid"]["nx"]=nx
    document["grid"]["ns"]=ns
    x=np.zeros((nfields,rank,nx))
    x[:,:,1:-1]=.001
    y=np.ones((nfields,rank,ns))
    np.savez_compressed(tmp_path/"initial.npz",x_profiles=x,y_profiles=y,
                        velocity=np.zeros((nfields,nx,ns)))
    np.savez_compressed(tmp_path/"loads.npz",
                        mechanical_load=np.zeros((nfields,nx,ns)))
    document["rank"]=rank
    document["solver"]["max_acceleration_coordinates"]=1400
    document["solver"]["max_halfstep_coordinates"]=192
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="HALFSTEP_CAP_TOO_SMALL"):
        native.run(path,validate_only=True)
    document["solver"]["max_halfstep_coordinates"]=1200
    path.write_text(json.dumps(document))
    assert native.run(path,validate_only=True)["rank"]==rank
    assert not (tmp_path/"output").exists()

def test_dynamic_mass_preflight_prevents_unusable_initial_solver(tmp_path):
    path,data=make_case(tmp_path,theory="CUF_TE4")
    data["solver"]["max_acceleration_coordinates"]=10
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="ACCELERATION_CAP_TOO_SMALL"):
        native.run(path,validate_only=True)
    assert not (tmp_path/"output").exists()


def test_public_final_runner_reaches_only_native_coupled_mvi(monkeypatch,tmp_path):
    from cylinder_final_stage11a_dispatcher import run_final_case
    from independent_mvi import run_models as legacy_linear
    from independent_mkr import run_models as opposite
    case,_=make_case(tmp_path,theory="REDDY",integrator="GAUSS_IRK4")
    called=[]
    def fake_native(path,*,validate_only):
        assert not validate_only
        called.append(str(path))
        return dict(method=native.METHOD,
                    status="EXPERIMENTAL_COMPLETED_UNCERTIFIED",
                    scientific_accuracy_certified=False)
    def never(*a,**kw):
        raise AssertionError("legacy linear, MKR or reference solver invoked")
    monkeypatch.setattr(native,"run",fake_native)
    monkeypatch.setattr(legacy_linear,"run",never)
    monkeypatch.setattr(opposite,"run",never)
    import cylinder_reference_case as ref
    monkeypatch.setattr(ref,"run_reference_case",never)
    response=run_final_case(case)
    assert called==[str(case.resolve())]
    assert response["solver_backend"]=="INDEPENDENT_MVI"
    assert response["no_reference_fallback"] is True
    assert response["scientific_accuracy_certified"] is False
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("solver",("AUTO","DENSE_LU","MATRIX_FREE_GMRES"))
def test_optional_halfstep_linear_solver_configuration_reaches_native_settings(
        tmp_path,solver):
    case,document=make_case(tmp_path,theory="CUF_TE2")
    document["solver"].update(
        linear_solver=solver,dense_coordinate_limit=24,
        krylov_restart=12,krylov_maxiter=20,
        krylov_relative_tolerance=1.e-5)
    case.write_text(json.dumps(document))
    parsed=native.load_case(case)
    selected=parsed["settings"].halfstep
    assert selected.linear_solver==solver
    assert selected.dense_coordinate_limit==24
    assert selected.krylov_restart==12
    assert selected.krylov_maxiter==20
    assert selected.krylov_relative_tolerance==pytest.approx(1.e-5)
    assert not (tmp_path/"output").exists()


def test_unknown_krylov_backend_never_defaults_to_dense_or_mkr(tmp_path):
    case,document=make_case(tmp_path)
    document["solver"]["linear_solver"]="SVD"
    case.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="MVI_UNKNOWN_HALFSTEP_LINEAR_SOLVER"):
        native.load_case(case)
    assert not (tmp_path/"output").exists()


def test_legacy_coupled_case_without_optional_controls_remains_valid(tmp_path):
    case,_=make_case(tmp_path)
    result=native.load_case(case)
    assert result["settings"].halfstep.linear_solver=="AUTO"
    assert result["settings"].halfstep.dense_coordinate_limit==96
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("theory",native.THEORIES)
def test_electric_face_charge_preflight_enters_generalized_electric_work(
        tmp_path,theory):
    case,document=make_case(tmp_path,theory=theory,electric=True)
    document["electric_faces"]={
        "inner":{"mode":"CHARGE","charge_c_per_m2":2.},
        "outer":{"mode":"CHARGE","charge_c_per_m2":-.5},
    }
    case.write_text(json.dumps(document),encoding="utf8")
    parsed=native.load_case(case)
    h=parsed["surface"].thickness_m
    radius=parsed["surface"].grid.radius_m
    expected=(1.-h/(2*radius))*2.+(1.+h/(2*radius))*(-.5)
    np.testing.assert_allclose(parsed["load"]["free_charge"][0],expected)
    assert parsed["electric_faces"]==document["electric_faces"]
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("condition",("POTENTIAL","UNKNOWN"))
def test_unsupported_face_modes_do_not_launch_solver(tmp_path,condition):
    case,document=make_case(tmp_path,electric=True)
    document["electric_faces"]={
        "inner":None,
        "outer":{"mode":condition,"charge_c_per_m2":None},
    }
    case.write_text(json.dumps(document))
    expected=("MVI_FACE_DIRICHLET_POTENTIAL_NOT_IMPLEMENTED"
              if condition=="POTENTIAL" else "MVI_ELECTRIC_FACE_MODE_UNSUPPORTED")
    with pytest.raises(ValueError,match=expected):
        native.run(case,validate_only=True)
    assert not (tmp_path/"output").exists()


def test_electricity_disabled_rejects_face_charge_in_preflight(tmp_path):
    case,document=make_case(tmp_path,electric=False)
    document["electric_faces"]={
        "inner":{"mode":"CHARGE","charge_c_per_m2":.2},
        "outer":None,
    }
    case.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="MVI_ELECTRIC_FACE_WITH_ELECTRICITY_OFF"):
        native.run(case,validate_only=True)
    assert not (tmp_path/"output").exists()


def test_old_coupled_case_omitting_faces_remains_compatible(tmp_path):
    case,_=make_case(tmp_path,electric=True)
    parsed=native.load_case(case)
    np.testing.assert_array_equal(
        parsed["load"]["free_charge"],
        np.zeros_like(parsed["load"]["free_charge"]))
    assert parsed["electric_faces"]=={"inner":None,"outer":None}
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("integrator",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_native_runner_delivers_tabulated_forcing_to_actual_stage_times(
        tmp_path,monkeypatch,integrator):
    """Full CASE routing and output with numerical drivers mocked: no solve."""
    import csv
    from types import SimpleNamespace
    from independent_mvi.coupled_newmark import CoupledMVIDynamicState
    from independent_mvi.coupled_initial import MVIAccelerationResult

    case,document=make_case(
        tmp_path,theory="CUF_TE3",kind="DYNAMIC",integrator=integrator)
    document["load_time_law"]=dict(
        kind="TABULATED_LINEAR",
        time_s=[0.,.005,.01,.02],
        mechanical_scale=[0.,.3,1.,0.])
    case.write_text(json.dumps(document))
    n=12
    np.savez_compressed(
        tmp_path/"loads.npz",
        mechanical_load=np.ones((n,9,5)))
    events=[]

    def initial(config,factors,velocity,**kwargs):
        events.append(("INITIAL",0.,float(np.mean(kwargs["mechanical_load"]))))
        state=CoupledMVIDynamicState(0.,factors,velocity,np.zeros_like(velocity))
        return MVIAccelerationResult(
            state,"MVI_INITIAL_EQUILIBRIUM_CLOSED",None,None)

    monkeypatch.setattr(native,"close_mvi_instantaneous_state",initial)
    monkeypatch.setattr(native,"_dynamic_green_audit",lambda *a,**kw:{})
    def fake_advance(self,state,dt_s,*,loads_at=None,
                     mechanical_load=None,free_charge=None,**kwargs):
        if loads_at is None:
            load=mechanical_load
            events.append((integrator,state.time_s+dt_s,float(np.mean(load))))
        else:
            from independent_mvi.coupled_gauss2 import GAUSS_C
            from independent_mvi.coupled_bathe import BATHE_GAMMA
            samples=(
                (state.time_s+BATHE_GAMMA*dt_s,state.time_s+dt_s)
                if integrator=="BATHE" else
                tuple(state.time_s+c*dt_s for c in GAUSS_C)
                +(state.time_s+dt_s,))
            for time in samples:
                load,charge=loads_at(time)
                assert charge is None
                events.append((integrator,time,float(np.mean(load))))
        future=CoupledMVIDynamicState(
            state.time_s+dt_s,state.factors,state.velocity,state.acceleration)
        return SimpleNamespace(state=future,status="MOCK_ACCEPTED")

    if integrator=="NEWMARK":
        monkeypatch.setattr(native.NativeCoupledMVINewmark,"advance",fake_advance)
    elif integrator=="BATHE":
        monkeypatch.setattr(native.NativeCoupledMVIBathe,"advance",fake_advance)
    else:
        monkeypatch.setattr(native.NativeCoupledMVIGauss2,"advance",fake_advance)
    result=native.run(case,validate_only=False)
    assert result["accepted_steps"]==2
    assert result["time_dependent_loads"] is True
    times=np.array(document["load_time_law"]["time_s"])
    scales=np.array(document["load_time_law"]["mechanical_scale"])
    assert events
    for _method,t,observed in events:
        assert observed==pytest.approx(np.interp(t,times,scales),abs=2.e-14)
    with (tmp_path/"output"/"PHYSICAL_RADIAL_HISTORY.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3
    assert [int(row["step"]) for row in rows]==[0,1,2]
    with np.load(tmp_path/"output"/"STEP_0002.npz") as result_npz:
        assert result_npz["physical_z_stations_m"].shape==(3,)
        assert result_npz["physical_uvw_stations_m"].shape==(3,3,9,5)
    # The independent physical calculation was never invoked.


def test_dynamic_time_law_is_preflight_only_and_does_not_create_output(tmp_path):
    case,document=make_case(tmp_path,theory="CUF_TE4")
    document["load_time_law"]=dict(
        kind="TABULATED_LINEAR",time_s=[0.,.01,.02],
        mechanical_scale=[0.,1.,0.])
    case.write_text(json.dumps(document))
    result=native.run(case,validate_only=True)
    assert result["time_dependent_loads"] is True
    assert result["physical_steps_computed"]==0
    assert not (tmp_path/"output").exists()


def test_static_case_cannot_silently_accept_dynamic_load_law(tmp_path):
    case,document=make_case(tmp_path,kind="STATIC")
    document["load_time_law"]=dict(
        kind="TABULATED_LINEAR",time_s=[0.,.01],
        mechanical_scale=[0.,1.])
    case.write_text(json.dumps(document))
    with pytest.raises(ValueError,match="STATIC_FORBIDS_TIME_DEPENDENT_LOAD"):
        native.run(case,validate_only=True)
    assert not (tmp_path/"output").exists()
