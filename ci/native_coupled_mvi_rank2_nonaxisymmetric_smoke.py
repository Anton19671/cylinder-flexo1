"""Physical nonaxisymmetric loaded MVI rank-2 Newmark control.

The spatial forcing contains an angular Fourier mode, so the rank-2
test exercises a nonconstant Y partner in the independent weak method.
This is a numerical integration smoke, not a benchmark or a statement
of G3D microfree boundary accuracy. Do not run large time series.
"""
from pathlib import Path
import tempfile,json
import numpy as np
from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as mvi


def main():
    with tempfile.TemporaryDirectory(prefix="mvi-r2-loaded-") as tmp:
        folder=Path(tmp)
        nx,ns=11,9
        angle=2.*np.pi*np.arange(ns)/ns
        x=np.zeros((3,2,nx))
        y=np.broadcast_to(
            np.stack((np.ones(ns),np.cos(angle))), (3,2,ns)).copy()
        forcing=np.zeros((3,nx,ns))
        forcing[2]=(-600.*(1.+.2*np.cos(angle)))[None,:]
        np.savez_compressed(folder/"initial.npz",
            x_profiles=x,y_profiles=y,
            velocity=np.zeros((3,nx,ns)))
        np.savez_compressed(folder/"loads.npz",mechanical_load=forcing)
        case={
            "schema":mvi.SCHEMA,"research_opt_in":True,
            "physics":{
                "theory":"KL","geometry_model":"LINEAR",
                "material":{"young_pa":1.e9,"poisson":.3,"ell_m":0.,
                            "mu1_c_per_m":0.,"mu2_c_per_m":0.,
                            "kappa_f_per_m":0.,"gradient_model":"G3D"},
                "thickness_m":.08,"density_kg_m3":2500.,
                "eta_s":0.,"mass_damping_rate_per_s":0.,
                "n_gauss":2,"potential_order":None},
            "grid":{"nx":nx,"ns":ns,"length_m":1.2,"radius_m":2.},
            "boundary":{"mechanical":"IMMOBILE_PARALLEL_ENDS",
                        "electric":"OFF"},
            "rank":2,
            "solver":{
                "kind":"DYNAMIC","integrator":"NEWMARK",
                "dt_s":.0001,"steps":1,
                "field_scales":[1.e-4]*3,"energy_scale_j":1.,
                "max_sweeps":30,
                "max_acceleration_coordinates":1024,
                "max_halfstep_coordinates":1024,
                "linear_solver":"AUTO",
                "acceleration_linear_solver":"MATRIX_FREE_GMRES",
                "acceleration_dense_limit":48,
                "acceleration_krylov_relative_tolerance":1.e-9,
            },
            "initial_state_npz":"initial.npz","loads_npz":"loads.npz",
            "output_dir":"out",
        }
        file=folder/"case.json"
        file.write_text(json.dumps(case))
        answer=mvi.run(file,validate_only=False)
        if answer["accepted_steps"]!=1:
            raise AssertionError("rank-2 step not accepted")
        with np.load(folder/"out"/"STEP_0001.npz",
                     allow_pickle=False) as data:
            q=np.asarray(data["mechanical"])
            a=np.asarray(data["acceleration"])
            assert data["x_profiles"].shape==(3,2,nx)
            assert data["y_profiles"].shape==(3,2,ns)
            if not (np.isfinite(q).all() and np.isfinite(a).all()):
                raise AssertionError("nonfinite rank-2 loaded field")
            midpoint=q[2,nx//2]
            assert midpoint[0]<0
            assert abs(midpoint[0]-midpoint[ns//2])>1.e-13
        print(json.dumps({
            "status":"LOADED_NONAXISYMMETRIC_MVI_RANK2_ONE_STEP_ACCEPTED",
            "central_displacement_at_theta0_m":float(midpoint[0]),
            "central_displacement_at_opposite_theta_m":float(midpoint[ns//2]),
            "scientific_accuracy_certified":False,
            "microfree_boundary_certified":False,
        }),flush=True)


if __name__=="__main__":
    main()
