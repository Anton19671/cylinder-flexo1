"""Seven theories x all MVI ranks 1..8 x all three time integrators.

Physical G3D/Gz CASE preflight ONLY. This verifies independent
rank-size budgets, full native module dispatch, geometry selection,
electric ON/OFF, and no output on --validate-only. The resulting
simulation is not launched here; no source microfree validation.
"""
import json
import numpy as np
import pytest

from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as native

THEORIES=native.THEORIES
INTEGRATORS=("NEWMARK","BATHE","GAUSS_IRK4")


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("rank",range(1,9))
@pytest.mark.parametrize("integrator",INTEGRATORS)
@pytest.mark.parametrize("electric",(False,True))
def test_all_native_mvi_physical_case_routes_on_high_rank_grid(
        tmp_path,theory,rank,integrator,electric):
    nx,ns=11,9
    mechanical=(3*(int(theory[-1])+1) if theory.startswith("CUF")
                else 3 if theory=="KL" else 5)
    ne=2 if electric else 0
    total=mechanical+ne
    rng=np.random.default_rng(122+rank)
    # An admissible separated rank-R representation with homogeneous
    # mechanical ends and all electric grounded ends.
    x=np.zeros((total,rank,nx))
    y=rng.normal(size=(total,rank,ns))
    x[:,:,1:-1]=rng.normal(size=(total,rank,nx-2))*.001
    np.savez_compressed(
        tmp_path/"initial.npz",x_profiles=x,y_profiles=y,
        velocity=np.zeros((mechanical,nx,ns)))
    source=dict(mechanical_load=np.zeros((mechanical,nx,ns)))
    if electric:
        source["free_charge"]=np.zeros((ne,nx,ns))
    np.savez_compressed(tmp_path/"load.npz",**source)
    case={
        "schema":native.SCHEMA,
        "research_opt_in":True,
        "physics":{
            "theory":theory,
            "geometry_model":("LINEAR","VON_KARMAN","GREEN_LAGRANGE")[
                (rank-1)%3],
            "material":{
                "young_pa":1.e9,"poisson":.3,"ell_m":1.e-6,
                "mu1_c_per_m":0.,"mu2_c_per_m":0.,
                "kappa_f_per_m":1.e-8 if electric else 0.,
                "gradient_model":("G3D" if rank%2 else "GZ")},
            "thickness_m":.1,"density_kg_m3":2500.,
            "eta_s":0.,"mass_damping_rate_per_s":0.,
            "n_gauss":2,"potential_order":1 if electric else None,
        },
        "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
        "boundary":{
            "mechanical":"IMMOBILE_PARALLEL_ENDS",
            "electric":"GROUNDED_BOTH" if electric else "OFF"},
        "rank":rank,
        "solver":{
            "kind":"DYNAMIC","integrator":integrator,
            "dt_s":.001,"steps":1,"field_scales":[1.]*total,
            "energy_scale_j":1.,"max_sweeps":4,
            "max_acceleration_coordinates":4096,
            "max_halfstep_coordinates":4096,
            "linear_solver":"AUTO",
            "dense_coordinate_limit":96,
            "krylov_restart":24,
            "krylov_maxiter":80,
            "krylov_relative_tolerance":1.e-5,
            "acceleration_linear_solver":"AUTO",
            "acceleration_dense_limit":48,
            "acceleration_krylov_restart":30,
            "acceleration_krylov_maxiter":100,
            "acceleration_krylov_relative_tolerance":1.e-8,
        },
        "initial_state_npz":"initial.npz",
        "loads_npz":"load.npz",
        "output_dir":"output",
    }
    case_file=tmp_path/"case.json"
    case_file.write_text(json.dumps(case),encoding="utf-8")
    result=native.run(case_file,validate_only=True)
    assert result["rank"]==rank
    assert result["integrator"]==integrator
    assert result["theory"]==theory
    assert result["physical_steps_computed"]==0
    assert result["scientific_accuracy_certified"] is False
    assert not (tmp_path/"output").exists()
