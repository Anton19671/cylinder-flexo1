"""No-solve CASE/dispatcher contracts for native nonlinear strong MKR.

These tests construct physical fields and explicit numerical end programs,
but do NOT run a loaded shell, Newton step or dynamic simulation.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cylinder_paths import activate
activate()

from independent_mkr import run_coupled_research as native
from cylinder_independent_dispatch import validate_independent_case


THEORIES=native.THEORIES


def make_case(tmp_path,theory="REDDY",kind="DYNAMIC",
              integrator="NEWMARK",electric=False):
    n=3 if theory=="KL" else 5 if theory in ("FSDT","REDDY") else (
        3*(int(theory[-1])+1))
    nx,ns=9,5
    q=np.zeros((n,nx,ns))
    state={"mechanical":q}
    if kind=="DYNAMIC":
        state["velocity"]=np.zeros_like(q)
    ne=1 if electric else 0
    if electric:
        state["electrical"]=np.zeros((ne,nx,ns))
    np.savez_compressed(tmp_path/"initial.npz",**state)
    loads={"mechanical_load":np.zeros_like(q)}
    if electric:
        loads["electrical_charge"]=np.zeros((ne,nx,ns))
    np.savez_compressed(tmp_path/"loads.npz",**loads)
    case={
      "schema":native.SCHEMA,
      "research_opt_in":True,
      "physics":{
        "theory":theory,"geometry_model":"LINEAR",
        "material":{"young_pa":1.e9,"poisson":.3,"ell_m":.01,
                    "mu1_c_per_m":.0,"mu2_c_per_m":.0,
                    "kappa_f_per_m":1.e-7 if electric else .0,
                    "gradient_model":"G3D"},
        "thickness_m":.1,"density_kg_m3":2500.,
        "eta_s":0.,"mass_damping_rate_per_s":0.,
        "n_gauss":2,"potential_order":0 if electric else None,
      },
      "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
      # Explicit axial half-orders represent only a preflight contract.
      # This synthetic test does not certify source physical PDE orders.
      "boundary":{"mechanical":"IMMOBILE_PARALLEL_ENDS",
                  "electric":"GROUNDED_BOTH" if electric else "OFF",
                  "axial_orders":[1]*n},
      "solver":{"kind":kind,
                "integrator":integrator if kind=="DYNAMIC" else None,
                "dt_s":.01 if kind=="DYNAMIC" else None,
                "steps":2 if kind=="DYNAMIC" else 0,
                "max_dofs":2200,"max_iterations":3,
                "mechanical_state_scale":1.,
                "mechanical_equation_scale":1.,
                "electrical_state_scale":1. if electric else None,
                "electrical_equation_scale":1. if electric else None,
                "potential_scale_v":1.},
      "initial_state_npz":"initial.npz",
      "loads_npz":"loads.npz",
      "output_dir":"result",
    }
    path=tmp_path/"case.json"
    path.write_text(json.dumps(case),encoding="utf-8")
    return path,case


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("integrator",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_21_own_strong_mkr_case_routes_without_solving(
        theory,integrator,tmp_path):
    path,case=make_case(tmp_path,theory=theory,integrator=integrator)
    answer=validate_independent_case(path)
    assert answer["valid"] is True
    assert answer["solver_backend"]=="INDEPENDENT_MKR"
    assert answer["validated_only"] is True
    assert answer["physical_steps_computed"]==0
    assert answer["scientific_accuracy_certified"] is False
    assert answer["integrator"]==integrator
    assert not (tmp_path/"result").exists()


@pytest.mark.parametrize("theory",THEORIES)
def test_static_strong_mkr_case_is_method_owned_and_no_output(tmp_path,theory):
    path,case=make_case(tmp_path,theory=theory,kind="STATIC")
    answer=native.run(path,validate_only=True)
    assert answer["kind"]=="STATIC"
    assert answer["method"]==native.METHOD
    assert answer["spatial_method"]=="DIRECT_STRONG_MKR_NO_MVI_SPATIAL_REUSE"
    assert not (tmp_path/"result").exists()


@pytest.mark.parametrize("electric",(False,True))
def test_own_electric_on_off_produces_matching_native_end_program(
        tmp_path,electric):
    path,_=make_case(tmp_path,electric=electric)
    parsed=native.load_case(path)
    end=parsed["program"]
    assert (end.electrical_ends is not None)==electric
    assert (end.electric_gauge is not None) is False
    assert not (tmp_path/"result").exists()


@pytest.mark.parametrize("integrator",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_electric_on_own_strong_integrator_case_preflight(
        tmp_path,integrator):
    path,_=make_case(tmp_path,electric=True,integrator=integrator)
    assert native.run(path,validate_only=True)["validated_only"]


def test_electric_floating_gauge_is_one_physical_global_mode(tmp_path):
    path,case=make_case(tmp_path,electric=True)
    case["boundary"]["electric"]="INSULATED_FLOATING"
    path.write_text(json.dumps(case))
    result=native.load_case(path)
    gauge=result["program"].electric_gauge
    assert gauge is not None
    assert float(np.sum(gauge.weights[0]))==pytest.approx(1.)
    assert result["engine"].electric_gauge is gauge


def test_curved_inner_outer_electrode_charge_is_native_and_independent(tmp_path):
    path,case=make_case(tmp_path,electric=True)
    case["electric_faces"]={
        "inner":{"mode":"CHARGE","charge_c_per_m2":2.},
        "outer":{"mode":"CHARGE","charge_c_per_m2":-.5}}
    path.write_text(json.dumps(case))
    parsed=native.load_case(path)
    r=parsed["model"].grid.radius_m
    h=parsed["model"].thickness_m
    expected=(1.-h/(2*r))*2.+(1.+h/(2*r))*(-.5)
    np.testing.assert_allclose(parsed["sources"]["electrical_charge"][0],expected)


def test_invalid_mechanical_end_order_is_rejected(tmp_path):
    path,case=make_case(tmp_path,theory="CUF_TE2")
    case["boundary"]["axial_orders"]=[3]*9
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match="CUF_MAXIMUM_HALF_ORDER"):
        native.load_case(path)


def test_invalid_face_potential_is_never_silently_accepted(tmp_path):
    path,case=make_case(tmp_path,electric=True)
    case["electric_faces"]={"inner":None,
        "outer":{"mode":"POTENTIAL","charge_c_per_m2":None}}
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match="FACE_POTENTIAL_NOT_IMPLEMENTED"):
        native.load_case(path)
    assert not (tmp_path/"result").exists()


def test_no_mvi_fallback_import_in_own_native_mkr_runner():
    import ast
    source=Path(native.__file__).read_text(encoding="utf-8")
    tree=ast.parse(source)
    imported=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith((
        "independent_mvi","cylinder_variational_mvi",
        "cylinder_reference_case","cylinder_functional_assembly"))
        for name in imported)
