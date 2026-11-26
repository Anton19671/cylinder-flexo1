"""Bounded native strong-SAT physical pilot, not a mesh benchmark."""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()
from independent_mkr import run_coupled_research as native
from cylinder_native_mkr_end_program import reference_axial_half_orders
from cylinder_kl_face_load import kl_radial_displacement

NX,NS=21,4
L,R,H=1.2,2.,.08
PRESSURE=100.
DT=1.e-4

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def make_case(root,theory,gradient,kind,*,nx=NX,n_gauss=12,steps=1,dt=DT):
    if not 20<=nx<=81 or not 5<=n_gauss<=24 or not 1<=steps<=4 or not math.isfinite(dt) or dt<=0:
        raise ValueError("MATCHED_SAT_PILOT_PARAMETER_RANGE")
    nm=3 if theory=="KL" else 5 if theory in ("FSDT","REDDY") else 3*(int(theory[-1])+1)
    pe=int(theory[-1]) if theory.startswith("CUF_TE") else 1
    q=np.zeros((nm,nx,NS));p=np.zeros((pe+1,nx,NS))
    load=np.zeros_like(q)
    # Outer surface radial dead traction, converted to reference surface
    # generalized forces for FSDT/Reddy/CUF. KL uses exact director work.
    if theory.startswith("CUF_TE"):
        load[2*(pe+1):]=-PRESSURE*(1.+H/(2*R))
    elif theory!="KL":
        load[2]=-PRESSURE*(1.+H/(2*R))
    initial=dict(mechanical=q,electrical=p)
    if kind!="STATIC":
        initial["velocity"]=q.copy()
    np.savez_compressed(root/"seed.npz",**initial)
    np.savez_compressed(root/"loads.npz",mechanical_load=load,electrical_charge=p)
    case=dict(schema=native.SCHEMA,research_opt_in=True,
        physics=dict(theory=theory,geometry_model="GREEN_LAGRANGE",
            material=dict(young_pa=1.e9,poisson=.3,ell_m=.002,mu1_c_per_m=.001,
                mu2_c_per_m=-.0005,kappa_f_per_m=1.e-8,gradient_model=gradient),
            thickness_m=H,density_kg_m3=2500.,eta_s=0.,
            mass_damping_rate_per_s=0.,n_gauss=n_gauss,potential_order=pe),
        grid=dict(nx=nx,ns=NS,length_m=L,radius_m=R,
                  axial_scheme="MATCHED_QUARTIC_SAT_V1"),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",electric="GROUNDED_BOTH",
                      axial_orders=list(reference_axial_half_orders(theory,gradient))),
        solver=dict(kind="STATIC" if kind=="STATIC" else "DYNAMIC",
            integrator=None if kind=="STATIC" else kind,
            dt_s=None if kind=="STATIC" else dt,steps=0 if kind=="STATIC" else steps,
            max_dofs=2048,max_iterations=8,linear_solver="DENSE_LU",dense_lu_dof_limit=2048,
            axisymmetric=True,
            mechanical_state_scale=1.e-4,mechanical_equation_scale=1.e4,
            electrical_state_scale=1.,electrical_equation_scale=1.e-3,potential_scale_v=1.),
        initial_state_npz="seed.npz",loads_npz="loads.npz",output_dir="RESULTS")
    if theory=="KL":
        case["kl_radial_face_traction"]=dict(face="OUTER",traction_pa=-PRESSURE)
    path=root/"CASE.json";write(path,case)
    return path,nm,pe

def radial_surfaces(model,q,pe):
    if model.theory.startswith("CUF_TE"):
        radial=q[2*(pe+1):]
        signs=(-1.)**np.arange(pe+1)
        return radial[0],radial.sum(axis=0),np.einsum("i,ijk->jk",signs,radial)
    if model.theory=="KL":
        jets=model.field_jets.prepare(q,model.derivative_orders)[:3]
        kw=dict(radius_m=model.grid.radius_m,geometry_model=model.geometry_model)
        return q[2],kl_radial_displacement(*jets,z_m=model.thickness_m/2,**kw),kl_radial_displacement(*jets,z_m=-model.thickness_m/2,**kw)
    return q[2],q[2],q[2]


def run(root,theory,gradient,kind,*,nx=NX,n_gauss=12,steps=1,dt=DT):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status="STARTED",theory=theory,gradient=gradient,kind=kind,
        grid=[nx,NS],pressure_pa=PRESSURE,n_gauss=n_gauss,
        steps_requested=0 if kind=="STATIC" else steps,
        dt_s=None if kind=="STATIC" else dt,axial_scheme="MATCHED_QUARTIC_SAT_V1",
        git_sha=os.environ.get("GITHUB_SHA"),scientific_accuracy_certified=False,
        physical_boundary_closure_certified=False,mesh_convergence_certified=False)
    marker=root/"PILOT.json";write(marker,report)
    try:
        path,nm,pe=make_case(root,theory,gradient,kind,nx=nx,n_gauss=n_gauss,steps=steps,dt=dt)
        prepared=native.load_case(path)
        preview=native.run(path,validate_only=True)
        assert preview["axial_scheme"]=="MATCHED_QUARTIC_SAT_V1"
        assert preview["axisymmetric"] is True
        print("PILOT_BEGIN",theory,gradient,kind,flush=True)
        started=time.perf_counter()
        ans=native.run(path,validate_only=False)
        report["solve_elapsed_s"]=time.perf_counter()-started
        assert ans["accepted_steps"]==(1 if kind=="STATIC" else steps)
        snapshot=root/"RESULTS"/("STATIC_ACCEPTED.npz" if kind=="STATIC" else f"STEP_{steps:04d}.npz")
        with np.load(snapshot,allow_pickle=False) as data:
            q=np.asarray(data["mechanical"],float);p=np.asarray(data["electrical"],float)
            assert q.shape==(nm,nx,NS) and p.shape==(pe+1,nx,NS)
            assert np.isfinite(q).all() and np.isfinite(p).all()
            assert np.all(q==q[..., :1]) and np.all(p==p[..., :1])
            mid,outer,inner=radial_surfaces(prepared["model"],q,pe)
            assert all(np.isfinite(v).all() for v in (mid,outer,inner))
            assert mid[nx//2,0]<-1.e-14 and np.max(abs(p))>1.e-14
            z=np.linspace(-1.,1.,81)
            phi=np.einsum("zk,kij->zij",z[:,None]**np.arange(pe+1)[None,:],p)
            assert np.max(abs(q[:,[0,-1],:]))<=1.e-9
            assert np.max(abs(p[:,[0,-1],:]))<=1.e-8
            report.update(midpoint_mid_surface_radial_m=float(mid[nx//2,0]),
                midpoint_outer_surface_radial_m=float(outer[nx//2,0]),
                midpoint_inner_surface_radial_m=float(inner[nx//2,0]),
                outer_radial_max_abs_m=float(np.max(abs(outer))),
                potential_coefficient_max_abs_v=float(np.max(abs(p))),
                sampled_physical_potential_max_abs_v=float(np.max(abs(phi))),
                potential_thickness_sample_count=len(z),
                mechanical_support_max_abs=float(np.max(abs(q[:,[0,-1],:]))),
                electric_support_max_abs_v=float(np.max(abs(p[:,[0,-1],:]))))
            if kind!="STATIC":
                for key in ("velocity","acceleration"):
                    a=np.asarray(data[key],float)
                    assert a.shape==q.shape and np.isfinite(a).all()
                    assert np.max(abs(a[:,[0,-1],:]))<=1.e-9
                    report[key+"_max_abs"]=float(np.max(abs(a)))
                assert math.isclose(float(data["time_s"]),steps*dt,rel_tol=1.e-12,abs_tol=1.e-14)
        report.update(status="ACCEPTED_RESEARCH_ONLY",solver_status=ans["status"],
            static_diagnostics=ans.get("static_nonlinear_diagnostics"),
            initial_diagnostics=ans.get("initial_nonlinear_diagnostics"))
    except Exception as exc:
        report.update(status="FAILED",error_type=type(exc).__name__,error=str(exc))
        write(marker,report)
        print("PILOT_FAILURE",json.dumps(report),flush=True)
        raise
    write(marker,report);print("PILOT_RESULT",json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--theory",choices=native.THEORIES,required=True)
    p.add_argument("--gradient",choices=("G3D","Gz"),required=True)
    p.add_argument("--kind",choices=("STATIC","NEWMARK","BATHE","GAUSS_IRK4"),required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--nx",type=int,default=NX)
    p.add_argument("--n-gauss",type=int,default=12)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--dt",type=float,default=DT)
    a=p.parse_args();run(a.output,a.theory,a.gradient,a.kind,nx=a.nx,n_gauss=a.n_gauss,steps=a.steps,dt=a.dt)
