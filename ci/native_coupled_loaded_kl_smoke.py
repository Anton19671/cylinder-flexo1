"""Small nonzero-loaded cylindrical KL control for both native methods.

This is an executable numerical-method integration check, NOT a benchmark
and NOT a certification of gradient G3D/Gz microfree face conditions.
The classical elastic linear KL limit ell=0 has axial half-orders
(u,v,w)=(1,1,2) with immobile q at both ends and natural w bending
moment. Independent strong-MKR and SBP/Fourier weak-MVI use their own
discrete spatial operators; one constant circumferential pressure has
a rank-1 admissible solution in the linear rotationally symmetric
subspace. The methods need not share discretization values.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
from cylinder_paths import activate
activate()
from independent_mkr import run_coupled_research as mkr
from independent_mvi import run_coupled_research as mvi

L=1.2
R=2.
H=.08
E=1.e9
P=-1000.
NX=9
NS=5

def physics():
    return dict(
        theory="KL",geometry_model="LINEAR",
        material=dict(
            young_pa=E,poisson=.3,ell_m=0.,
            mu1_c_per_m=0.,mu2_c_per_m=0.,
            kappa_f_per_m=0.,gradient_model="G3D"),
        thickness_m=H,density_kg_m3=2500.,
        eta_s=0.,mass_damping_rate_per_s=0.,n_gauss=2,
        potential_order=None)

def load():
    f=np.zeros((3,NX,NS))
    f[2]=P
    return f

def work_mkr(root):
    work=root/"mkr"
    work.mkdir()
    np.savez_compressed(work/"seed.npz",
                        mechanical=np.zeros((3,NX,NS)))
    np.savez_compressed(work/"load.npz",mechanical_load=load())
    case=dict(
        schema=mkr.SCHEMA,research_opt_in=True,
        physics=physics(),
        grid=dict(nx=NX,ns=NS,length_m=L,radius_m=R),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                      electric="OFF",axial_orders=[1,1,2]),
        solver=dict(kind="STATIC",integrator=None,dt_s=None,
                    steps=0,max_dofs=512,max_iterations=12,
                    linear_solver="AUTO",dense_lu_dof_limit=384,
                    mechanical_state_scale=1.e-4,
                    mechanical_equation_scale=1.e4,
                    electrical_state_scale=None,
                    electrical_equation_scale=None,
                    potential_scale_v=1.),
        initial_state_npz="seed.npz",loads_npz="load.npz",
        output_dir="out")
    casefile=work/"CASE.json"
    casefile.write_text(json.dumps(case))
    result=mkr.run(casefile,validate_only=False)
    with np.load(work/"out"/"STATIC_ACCEPTED.npz",
                 allow_pickle=False) as f:
        q=np.array(f["mechanical"])
    return dict(status=result["status"],
        displacement_centre_m=float(q[2,NX//2,NS//2]),
        displacement_max_abs_m=float(np.max(np.abs(q[2]))),
        returned_nonzero=bool(np.max(np.abs(q[2]))>1.e-12))

def work_mvi(root):
    work=root/"mvi"
    work.mkdir()
    x=np.zeros((3,1,NX))
    y=np.ones((3,1,NS))
    np.savez_compressed(work/"seed.npz",
                        x_profiles=x,y_profiles=y)
    np.savez_compressed(work/"load.npz",mechanical_load=load())
    case=dict(
        schema=mvi.SCHEMA,research_opt_in=True,
        physics=physics(),
        grid=dict(nx=NX,ns=NS,length_m=L,radius_m=R),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",
                      electric="OFF"),
        solver=dict(kind="STATIC",integrator=None,dt_s=None,
                    steps=0,field_scales=[1.e-4]*3,
                    energy_scale_j=1.,max_sweeps=30,
                    max_acceleration_coordinates=256,
                    max_halfstep_coordinates=256),
        rank=1,initial_state_npz="seed.npz",
        loads_npz="load.npz",output_dir="out")
    casefile=work/"CASE.json"
    casefile.write_text(json.dumps(case))
    result=mvi.run(casefile,validate_only=False)
    with np.load(work/"out"/"STATIC_ACCEPTED.npz",
                 allow_pickle=False) as f:
        q=np.array(f["mechanical"])
    return dict(status=result["status"],
        displacement_centre_m=float(q[2,NX//2,NS//2]),
        displacement_max_abs_m=float(np.max(np.abs(q[2]))),
        returned_nonzero=bool(np.max(np.abs(q[2]))>1.e-12))

def main():
    with tempfile.TemporaryDirectory(prefix="native-loaded-kl-") as raw:
        root=Path(raw)
        a=work_mkr(root)
        print(json.dumps({"method":"INDEPENDENT_STRONG_MKR","result":a}),flush=True)
        b=work_mvi(root)
        print(json.dumps({"method":"INDEPENDENT_FIXED_RANK_MVI","result":b}),flush=True)
        if not (a["returned_nonzero"] and b["returned_nonzero"]):
            raise AssertionError("LOADED_NONZERO_RESPONSE_REQUIRED")
        if not (a["displacement_centre_m"]<0.
                and b["displacement_centre_m"]<0.):
            raise AssertionError("INCORRECT_SIGN_OF_APPLIED_RADIAL_LOAD")
        print(json.dumps({
            "status":"NONZERO_LOADED_PHYSICAL_KL_METHOD_SMOKE_NOT_BENCHMARK",
            "relative_central_difference":abs(
                a["displacement_centre_m"]-b["displacement_centre_m"])/max(
                    abs(a["displacement_centre_m"]),
                    abs(b["displacement_centre_m"]),1.e-30),
            "scientific_accuracy_certified":False,
            "source_gradient_microfree_certified":False,
        }),flush=True)

if __name__=="__main__":
    main()
