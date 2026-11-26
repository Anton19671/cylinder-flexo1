"""Analytic CC longitudinal GradEla limit of the full CUF TE1 shell.

Reference: V. Dimosthenis, JAES 22(3), 665-680 (2024),
DOI 10.5937/jaes0-51432, equations (4),(5),(17),(24),(27).
The paper is a bar model. For LINEAR CUF, nu=0, electricity OFF and
u_x(x,t) constant through thickness/circumference, its axial solution
is an exact special solution of the cylindrical three-dimensional energy:
W = E/2 (u_x'**2 + ell**2*u_x''**2).
Every other strain and its covariant gradient vanishes. The area
2*pi*R*h cancels from the free vibration equation. The paper supplies E,
rho and ell/L; L,R,h below choose a shell realization of the normalized
problem, not the dimensional geometry of the paper's bar.
The static sine-load extension is independently derived from its
differential equation, not represented as a published static result.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()
from independent_mkr import run_coupled_research as native
from independent_mvi import run_coupled_research as mvi
from cylinder_native_mkr_end_program import reference_axial_half_orders

E,RHO,L,R,H,ELL,A = 1.e9,2000.,1.,2.,.08,.2,1.e-5
NS=9
KINDS=('STATIC','NEWMARK','BATHE','GAUSS_IRK4')
RANKS=tuple(range(9))  # 0 denotes independent FDM, 1..8 fixed-rank MVI
K=math.pi/L
OMEGA_CL=K*math.sqrt(E/RHO)
OMEGA=OMEGA_CL*math.sqrt(1.+ELL**2*K**2)
DT=2.*math.pi/(40.*OMEGA)
STEPS=2

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def prepare(root,nx,kind,rank):
    if nx not in (21,41) or kind not in KINDS or rank not in RANKS:
        raise ValueError("UNKNOWN_BENCHMARK")
    root.mkdir(parents=True,exist_ok=False)
    x=np.linspace(0.,L,nx)
    shape=np.sin(K*x)
    shape[[0,-1]]=0.
    q=np.zeros((6,nx,NS))
    exact=q.copy()
    exact[0]=A*shape[:,None]
    load=q.copy()
    if kind=="STATIC":
        # Volume force f_x=E*(k^2+ell^2*k^4)*A*sin(k*x).
        # Its generalized TE0,TE1 loads per midsurface area:
        # integral (1+z/R)*[1,2z/h] dz = [h,h^2/(6R)].
        load[:2]=E*(K**2+ELL**2*K**4)*exact[0][None,:,:]*np.array([H,H**2/(6.*R)])[:,None,None]
        initial=dict(mechanical=q)
    else:
        initial=dict(mechanical=exact,velocity=q)
    np.savez_compressed(root/"seed.npz",**initial)
    np.savez_compressed(root/"loads.npz",mechanical_load=load)
    case=dict(schema=native.SCHEMA,research_opt_in=True,
        physics=dict(theory="CUF_TE1",geometry_model="LINEAR",
            material=dict(young_pa=E,poisson=0.,ell_m=ELL,
                mu1_c_per_m=0.,mu2_c_per_m=0.,kappa_f_per_m=0.,
                gradient_model="G3D"),
            thickness_m=H,density_kg_m3=RHO,eta_s=0.,
            mass_damping_rate_per_s=0.,n_gauss=8,potential_order=None),
        grid=dict(nx=nx,ns=NS,length_m=L,radius_m=R,
            axial_scheme="MATCHED_QUARTIC_SAT_V1"),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",electric="OFF",
            axial_orders=list(reference_axial_half_orders("CUF_TE1","G3D"))),
        solver=dict(kind="STATIC" if kind=="STATIC" else "DYNAMIC",
            integrator=None if kind=="STATIC" else kind,
            dt_s=None if kind=="STATIC" else DT,
            steps=0 if kind=="STATIC" else STEPS,
            axisymmetric=True,max_dofs=2048,max_iterations=10,
            linear_solver="DENSE_LU",dense_lu_dof_limit=2048,
            mechanical_state_scale=1.e-3,mechanical_equation_scale=1.e4,
            electrical_state_scale=None,electrical_equation_scale=None,
            potential_scale_v=1.),
        initial_state_npz="seed.npz",loads_npz="loads.npz",output_dir="RESULTS")
    if rank:
        # Independent MVI factors, not a FDM solution or full-space correction.
        # Every slot is retained; inactive physical terms may be exactly zero.
        # At t=0 dynamic displacement must be the prescribed sine field.
        # Independent admissible sine trial partners are supplied for all slots.
        # Static initial displacement is identically zero.
        xp=np.broadcast_to(
            np.sin(np.arange(1,rank+1)[:,None]*K*x[None,:])[None,:,:],
            (6,rank,nx)).copy()
        xp[:,:,0]=0.;xp[:,:,-1]=0.
        yp=np.zeros((6,rank,NS))
        if kind!="STATIC":
            yp[0,0,:]=A
        initial=dict(x_profiles=xp,y_profiles=yp)
        if kind!="STATIC":
            initial["velocity"]=q
        np.savez_compressed(root/"seed.npz",**initial)
        case.update(schema=mvi.SCHEMA,rank=rank,
            grid={**{k:v for k,v in case["grid"].items() if k!="axial_scheme"}, "axial_scheme":"SBP84_PAIRED_V1"},
            boundary={k:v for k,v in case["boundary"].items() if k!="axial_orders"},
            solver=dict(kind="STATIC" if kind=="STATIC" else "DYNAMIC",
                integrator=None if kind=="STATIC" else kind,
                dt_s=None if kind=="STATIC" else DT,
                steps=0 if kind=="STATIC" else STEPS,
                field_scales=[1.e-5]*6,energy_scale_j=.1,
                max_sweeps=40,max_acceleration_coordinates=5000,
                max_halfstep_coordinates=5000,
                allow_projected_static=kind=="STATIC",
                allow_projected_dynamic=kind!="STATIC",
                linear_solver="MATRIX_FREE_GMRES",dense_coordinate_limit=96,
                krylov_restart=64,krylov_maxiter=60,krylov_relative_tolerance=1.e-7,
                acceleration_linear_solver="AUTO",acceleration_dense_limit=192))
        np.testing.assert_allclose(np.einsum("crx,crs->cxs",xp,yp),
            q if kind=="STATIC" else exact,rtol=1.e-14,atol=1.e-18)
    path=root/"CASE.json";save(path,case)
    return path,exact

def relative(actual,reference):
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))

def run(root,nx,kind,rank):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status="STARTED",method="MVI" if rank else "FDM",rank=rank or None,axial_scheme="SBP84_PAIRED_V1" if rank else "MATCHED_QUARTIC_SAT_V1",git_sha=os.environ.get("GITHUB_SHA"),
        source_doi="10.5937/jaes0-51432",source_equation="27a,b",
        theory="CUF_TE1",gradient="G3D",geometry="LINEAR",electric="OFF",
        grid=[nx,NS],kind=kind,ell_over_L=ELL/L,
        E_pa=E,rho_kg_m3=RHO,poisson=0.,
        chosen_shell_geometry_m=[L,R,H],
        omega_exact_rad_s=OMEGA,omega_classical_rad_s=OMEGA_CL,
        normalized_omega_exact=OMEGA/OMEGA_CL,
        external_scope="EXACT_AXIAL_LIMIT_ONLY",
        static_reference_is_new_analytic_extension=True)
    marker=root/"EVIDENCE.json";save(marker,report)
    try:
        path,exact=prepare(root/"input",nx,kind,rank)
        owner=mvi if rank else native
        preview=owner.run(path,validate_only=True)
        if rank:
            assert preview["rank"]==rank
        else:
            assert preview["axisymmetric"] and preview["axial_scheme"]=="MATCHED_QUARTIC_SAT_V1"
        print("GRAD_ELA_BEGIN",rank,kind,nx,flush=True)
        answer=owner.run(path,validate_only=False)
        if rank:
            assert answer["rank"]==rank and answer["rank_reduction_performed"] is False
            report.update(solution_status=answer.get("solution_status"),
                solution_quality=answer.get("solution_quality"),
                static_sweeps=answer.get("sweeps"),
                dynamic_step_quality=answer.get("dynamic_solution_quality"),
                rank_reduction_performed=False,
                last_separated_term_norms=answer.get("last_separated_term_norms"))
        out=root/"input"/"RESULTS"
        if kind=="STATIC":
            assert answer["accepted_steps"]==1
            with np.load(out/"STATIC_ACCEPTED.npz",allow_pickle=False) as state:
                q=np.array(state["mechanical"])
            report.update(exact_midpoint_m=A,calculated_midpoint_m=float(q[0,nx//2,0]),
                full_field_relative_error=relative(q,exact),
                off_mode_relative_norm=float(np.linalg.norm(q[1:])/np.linalg.norm(exact)),
                support_max_abs_m=float(np.max(abs(q[:,[0,-1],:]))))
            passed=report["full_field_relative_error"]<=.02 and report["support_max_abs_m"]<=1.e-9
        else:
            assert answer["accepted_steps"]==STEPS
            with np.load(out/"STEP_0000.npz",allow_pickle=False) as state:
                acc=np.array(state["acceleration"])
                initial_q=np.array(state["mechanical"])
            # This is an acceleration-projected frequency estimate,
            # not a fitted eigenvalue of the entire discrete shell.
            omega2=-float(np.sum(acc*initial_q)/np.sum(initial_q**2))
            if not math.isfinite(omega2) or omega2<=0:
                raise ValueError("NONPOSITIVE_MEASURED_FREQUENCY_SQUARED")
            omega_est=math.sqrt(omega2)
            report.update(omega_acceleration_projected_rad_s=omega_est,
                normalized_omega_calculated=omega_est/OMEGA_CL,
                frequency_relative_error=abs(omega_est/OMEGA-1.),
                initial_acceleration_relative_error=relative(acc,-OMEGA**2*exact),
                initial_off_mode_acceleration_relative_norm=float(np.linalg.norm(acc[1:])/(OMEGA**2*np.linalg.norm(exact))),
                dt_s=DT)
            history=[]
            for j in range(1,STEPS+1):
                with np.load(out/f"STEP_{j:04d}.npz",allow_pickle=False) as state:
                    t=float(state["time_s"]);q=np.array(state["mechanical"]);v=np.array(state["velocity"])
                target=exact*math.cos(OMEGA*t)
                vtarget=-OMEGA*exact*math.sin(OMEGA*t)
                history.append(dict(step=j,time_s=t,
                    exact_midpoint_m=float(target[0,nx//2,0]),
                    calculated_midpoint_m=float(q[0,nx//2,0]),
                    full_field_relative_error=relative(q,target),
                    velocity_relative_error=relative(v,vtarget),
                    support_max_abs_m=float(np.max(abs(q[:,[0,-1],:])))))
            report["history"]=history
            passed=(report["frequency_relative_error"]<=.02
                    and report["initial_acceleration_relative_error"]<=.1
                    and max(h["full_field_relative_error"] for h in history)<=.02
                    and max(h["velocity_relative_error"] for h in history)<=.05
                    and max(h["support_max_abs_m"] for h in history)<=1.e-9)
        report["status"]="PASS_EXACT_AXIAL_LIMIT" if passed else "NUMERICAL_DISCREPANCY"
        report["solver_status"]=answer["status"]
        save(marker,report)
        print("GRAD_ELA_RESULT",json.dumps(report),flush=True)
        if not passed:raise AssertionError("ANALYTIC_AXIAL_LIMIT_OUTSIDE_PREDECLARED_TOLERANCE")
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
        if report["status"]=="STARTED":report["status"]="FAILED"
        save(marker,report)
        print("GRAD_ELA_FAILURE",json.dumps(report),flush=True)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--nx",type=int,choices=(21,41))
    p.add_argument("--kind",choices=KINDS)
    p.add_argument("--rank",type=int,choices=RANKS,default=0)
    p.add_argument("--output",required=True)
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args()
    if a.preflight:
        for n in (21,41):
            for k in KINDS:
                for rank in RANKS:
                    path,exact=prepare(Path(a.output)/f"{k}-{n}-R{rank}",n,k,rank)
                    (mvi if rank else native).load_case(path)
        print("GRAD_ELA_PREFLIGHT_OK 72 CASEs",flush=True)
    else:
        run(a.output,a.nx,a.kind,a.rank)
