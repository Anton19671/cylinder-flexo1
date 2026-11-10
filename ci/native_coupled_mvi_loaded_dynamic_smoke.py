"""Tiny NONZERO loaded physically independent MVI KL one-step dynamics.

Explicit rank 1, LINEAR, G3D, radial pressure and own Newmark. This
checks the entire actual material+inertia+X/Y+NPZ path. No source G3D
microfree certification, no MKR reuse, no long transient or benchmark.
"""
from pathlib import Path
import tempfile
import json
import numpy as np

from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as mvi


def main():
    with tempfile.TemporaryDirectory(prefix="mvi-kl-loaded-dynamic-") as tmp:
        path=Path(tmp)
        nx,ns=9,5
        x=np.zeros((3,1,nx))
        y=np.ones((3,1,ns))
        load=np.zeros((3,nx,ns))
        load[2]=-1000.
        np.savez_compressed(
            path/"initial.npz",x_profiles=x,y_profiles=y,
            velocity=np.zeros((3,nx,ns)))
        np.savez_compressed(path/"load.npz",mechanical_load=load)
        case={
            "schema":mvi.SCHEMA,"research_opt_in":True,
            "physics":{
                "theory":"KL","geometry_model":"LINEAR",
                "material":{
                    "young_pa":1.e9,"poisson":.3,"ell_m":0.,
                    "mu1_c_per_m":0.,"mu2_c_per_m":0.,
                    "kappa_f_per_m":0.,"gradient_model":"G3D"},
                "thickness_m":.08,"density_kg_m3":2500.,
                "eta_s":0.,"mass_damping_rate_per_s":0.,
                "n_gauss":2,"potential_order":None},
            "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
            "boundary":{"mechanical":"IMMOBILE_PARALLEL_ENDS",
                        "electric":"OFF"},
            "rank":1,
            "solver":{
                "kind":"DYNAMIC","integrator":"NEWMARK",
                "dt_s":.0002,"steps":1,
                "field_scales":[1.e-4]*3,"energy_scale_j":1.,
                "max_sweeps":35,"max_acceleration_coordinates":200,
                "max_halfstep_coordinates":512,
                "linear_solver":"AUTO",
                "acceleration_linear_solver":"AUTO",
                "acceleration_dense_limit":192,
            },
            "initial_state_npz":"initial.npz",
            "loads_npz":"load.npz","output_dir":"result",
        }
        file=path/"case.json"
        file.write_text(json.dumps(case),encoding="utf-8")
        result=mvi.run(file,validate_only=False)
        with np.load(path/"result"/"STEP_0000.npz",allow_pickle=False) as initial:
            a0=float(initial["acceleration"][2,nx//2,ns//2])
            t0=float(initial["time_s"])
        with np.load(path/"result"/"STEP_0001.npz",allow_pickle=False) as last:
            q=last["mechanical"]
            v=last["velocity"]
            a=last["acceleration"]
            w=float(q[2,nx//2,ns//2])
            vel=float(v[2,nx//2,ns//2])
            acc=float(a[2,nx//2,ns//2])
            assert np.isfinite(q).all() and np.isfinite(v).all()
            assert np.isfinite(a).all()
            assert last["x_profiles"].shape[1]==1
            assert float(last["time_s"])==.0002
        assert result["accepted_steps"]==1
        assert a0<0 and w<0 and vel<0
        print(json.dumps({
            "method":"MVI_RANK_1_OWN_G3D_LOADED_NEWMARK",
            "initial_radial_acceleration_m_s2":a0,
            "radial_displacement_m":w,
            "radial_velocity_m_s":vel,
            "radial_acceleration_m_s2":acc,
            "scientific_accuracy_certified":False,
            "physical_microfree_boundary_certified":False,
            "status":"LOADED_PHYSICAL_ONE_STEP_ACCEPTED",
        }),flush=True)


if __name__=="__main__":
    main()
