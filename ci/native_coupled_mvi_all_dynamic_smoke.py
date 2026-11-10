"""Twenty-one tiny PHYSICAL zero-state dynamic MVI execution checks.

Runs the actual independent coupled seven-theory physical constitutive,
inertia, fixed-rank X/Y, and native Newmark/Bathe/Gauss2 integrators for
one step each. No MKR spatial operator, SVD, reference solver, benchmark
or long transient is used. This does NOT certify natural microfree BCs.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as mvi

THEORIES=mvi.THEORIES
METHODS=("NEWMARK","BATHE","GAUSS_IRK4")


def run_one(root,theory,integrator):
    name=theory+"_"+integrator
    folder=root/name
    folder.mkdir()
    nx,ns=9,5
    n=(3 if theory=="KL" else 5 if theory in ("FSDT","REDDY")
       else 3*(int(theory[-1])+1))
    x=np.zeros((n,1,nx))
    y=np.ones((n,1,ns))
    np.savez_compressed(folder/"initial.npz",
        x_profiles=x,y_profiles=y,velocity=np.zeros((n,nx,ns)))
    np.savez_compressed(folder/"loads.npz",
        mechanical_load=np.zeros((n,nx,ns)))
    case={
      "schema":mvi.SCHEMA,"research_opt_in":True,
      "physics":{
        "theory":theory,"geometry_model":"LINEAR",
        "material":{
          "young_pa":1.e9,"poisson":.3,"ell_m":1.e-6,
          "mu1_c_per_m":0.,"mu2_c_per_m":0.,
          "kappa_f_per_m":0.,"gradient_model":"G3D"},
        "thickness_m":.1,"density_kg_m3":2500.,
        "eta_s":0.,"mass_damping_rate_per_s":0.,
        "n_gauss":2,"potential_order":None},
      "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
      "boundary":{"mechanical":"IMMOBILE_PARALLEL_ENDS",
                  "electric":"OFF"},
      "rank":1,
      "solver":{
        "kind":"DYNAMIC","integrator":integrator,
        "dt_s":.0002,"steps":1,
        "field_scales":[1.]*n,"energy_scale_j":1.,
        "max_sweeps":4,"max_acceleration_coordinates":2048,
        "max_halfstep_coordinates":512,
        "linear_solver":"AUTO",
        "acceleration_linear_solver":"MATRIX_FREE_GMRES",
        "acceleration_dense_limit":24,
        "acceleration_krylov_relative_tolerance":1.e-8,
      },
      "initial_state_npz":"initial.npz",
      "loads_npz":"loads.npz","output_dir":"out"}
    file=folder/"case.json"
    file.write_text(json.dumps(case))
    output=mvi.run(file,validate_only=False)
    assert output["accepted_steps"]==1
    assert output["scientific_accuracy_certified"] is False
    assert output["physical_boundary_closure_certified"] is False
    for step in (0,1):
        with np.load(folder/"out"/f"STEP_{step:04d}.npz",
                     allow_pickle=False) as data:
            assert data["x_profiles"].shape[1]==1
            np.testing.assert_allclose(
                data["mechanical"],0.,rtol=0.,atol=1e-11)
            np.testing.assert_allclose(
                data["velocity"],0.,rtol=0.,atol=1e-11)
            np.testing.assert_allclose(
                data["acceleration"],0.,rtol=0.,atol=1e-11)
    print(json.dumps({"theory":theory,"integrator":integrator,
                      "accepted":output["accepted_steps"],
                      "status":output["status"]}),flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="mvi-seven-dynamic-") as work:
        base=Path(work)
        for theory in THEORIES:
            for integrator in METHODS:
                run_one(base,theory,integrator)
    print(json.dumps({
        "status":"TWENTY_ONE_PHYSICAL_ZERO_STATE_DYNAMIC_SMOKES_COMPLETE",
        "scientific_accuracy_certified":False,
        "microfree_conditions_certified":False
    }),flush=True)


if __name__=="__main__":
    main()
