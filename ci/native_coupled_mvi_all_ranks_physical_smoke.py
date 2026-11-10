"""Actual independent fixed-rank MVI-1..8 execution, not just CASE validation.

KL, LINEAR, G3D, 11x9, all three independent time schemes, zero-load
one-step physical shell. The zero state checks rank preservation and
physical mass/time/endpoints without a loaded-physics accuracy claim.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as mvi

METHODS=("NEWMARK","BATHE","GAUSS_IRK4")


def physical_run(base,rank,method):
    folder=base/f"R{rank}_{method}"
    folder.mkdir()
    nx,ns=11,9
    n=3
    x=np.zeros((n,rank,nx))
    # The angular rank partners are an independent orthonormal Fourier
    # family. Even with zero displacement, no partner is fabricated
    # by a singular QR normalizer.
    theta=2.*np.pi*np.arange(ns)/ns
    basis=[np.ones(ns)]
    for k in range(1,5):
        basis.extend((np.cos(k*theta),np.sin(k*theta)))
    y=np.broadcast_to(np.asarray(basis[:rank]),(n,rank,ns)).copy()
    np.savez_compressed(
        folder/"initial.npz",x_profiles=x,y_profiles=y,
        velocity=np.zeros((n,nx,ns)))
    np.savez_compressed(folder/"loads.npz",
                        mechanical_load=np.zeros((n,nx,ns)))
    case={
        "schema":mvi.SCHEMA,"research_opt_in":True,
        "physics":{
            "theory":"KL","geometry_model":"LINEAR",
            "material":{"young_pa":1.e9,"poisson":.3,"ell_m":1.e-6,
                "mu1_c_per_m":0.,"mu2_c_per_m":0.,
                "kappa_f_per_m":0.,"gradient_model":"G3D"},
            "thickness_m":.1,"density_kg_m3":2500.,
            "eta_s":0.,"mass_damping_rate_per_s":0.,
            "n_gauss":2,"potential_order":None,
        },
        "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
        "boundary":{"mechanical":"IMMOBILE_PARALLEL_ENDS","electric":"OFF"},
        "rank":rank,
        "solver":{
            "kind":"DYNAMIC","integrator":method,"dt_s":.0002,
            "steps":1,"field_scales":[1.]*n,"energy_scale_j":1.,
            "max_sweeps":3,"max_acceleration_coordinates":4096,
            "max_halfstep_coordinates":4096,
            "linear_solver":"AUTO","acceleration_linear_solver":"MATRIX_FREE_GMRES",
            "acceleration_dense_limit":48,
            "acceleration_krylov_relative_tolerance":1.e-8,
        },
        "initial_state_npz":"initial.npz",
        "loads_npz":"loads.npz","output_dir":"out",
    }
    file=folder/"case.json"
    file.write_text(json.dumps(case))
    result=mvi.run(file,validate_only=False)
    if result["accepted_steps"]!=1:
        raise AssertionError("not accepted: "+str(result))
    for index in (0,1):
        with np.load(folder/"out"/f"STEP_{index:04d}.npz",
                     allow_pickle=False) as saved:
            assert saved["x_profiles"].shape==(n,rank,nx)
            assert saved["y_profiles"].shape==(n,rank,ns)
            np.testing.assert_allclose(saved["mechanical"],0.,atol=1e-10,rtol=0.)
            np.testing.assert_allclose(saved["velocity"],0.,atol=1e-10,rtol=0.)
    print(json.dumps({"rank":rank,"integrator":method,
                      "status":result["status"]}),flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="mvi-physical-ranks-") as root:
        for rank in range(1,9):
            for method in METHODS:
                physical_run(Path(root),rank,method)
    print(json.dumps({"status":"MVI_RANKS_1_TO_8_DYNAMIC_PHYSICAL_ZERO_STATE_COMPLETE",
                      "loaded_rank_accuracy_verified":False,
                      "scientific_accuracy_certified":False}),flush=True)


if __name__=="__main__":
    main()
