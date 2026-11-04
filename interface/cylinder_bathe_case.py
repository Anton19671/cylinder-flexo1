"""Opt-in physical Bathe time history, independently of Newmark and Gauss.

Each full Bathe step solves two real variational MVI/FDM displacement
problems and accepts the state only AFTER both substeps and all kinematic,
physical-trace and spatial acceptance gates pass. Rank fixed for q only.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import numpy as np
from cylinder_bathe_mvi import BATHE_GAMMA,solve_bathe_variational_step
from cylinder_dynamic_trace import (mechanical_trace_matrix,
    sparse_mechanical_trace_matrix,effective_step_acceptance)


def _json(path, data):
    path=Path(path)
    p=path.with_name(path.name+".tmp")
    with p.open("w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(p,path)


def _npz(path,**data):
    path=Path(path);p=path.with_name(path.name+".tmp")
    with p.open("wb") as f:np.savez_compressed(f,**data)
    os.replace(p,path)


def _check_trace(problem,q,p,result,time_s,initial_target,*,matrix_free=False):
    if matrix_free:
        from cylinder_variational_boundary import compile_boundary_program
        edge=compile_boundary_program(problem.functional,result.mechanical,
            result.electrical,spec=problem.spec,calculation_kind="DYNAMIC",
            time_s=time_s)
    else:
        edge=problem.assemble(result.mechanical,result.electrical,
            acceleration=np.zeros_like(result.mechanical),
            density_kg_m3=1.,time_s=time_s).edge
    trace=(sparse_mechanical_trace_matrix if matrix_free
           else mechanical_trace_matrix)
    c,d=trace(edge.constraints,result.mechanical.size)
    if c.shape!=initial_target[0].shape:
        raise ValueError("BATHE_MECHANICAL_TRACE_MATRIX_CHANGED")
    c0,d0=initial_target
    if matrix_free:
        # Compare sparse physical trace entries without converting a
        # high-order CUF full-nodal operator to an enormous dense matrix.
        difference=(c-c0).tocsr()
        matrix_equal=(difference.shape==c0.shape and
            float(np.max(np.abs(difference.data),initial=0.))<=1.e-12)
    else:
        matrix_equal=np.allclose(c,c0,rtol=1.e-12,atol=1.e-12)
    if not matrix_equal or not np.allclose(d,d0,rtol=1e-12,atol=1e-12):
        raise ValueError("BATHE_MOVING_SUPPORT_NOT_YET_SUPPORTED")
    errors={
        "displacement_m":float(np.max(abs(c@result.mechanical.ravel()-d),initial=0.)),
        "velocity_m_s":float(np.max(abs(c@result.velocity.ravel()),initial=0.)),
        "acceleration_m_s2":float(np.max(abs(c@result.acceleration.ravel()),initial=0.))}
    if (errors["displacement_m"]>1e-9 or errors["velocity_m_s"]>1e-8 or
            errors["acceleration_m_s2"]>1e-7):
        raise ValueError("BATHE_MECHANICAL_BOUNDARY_KINEMATICS_DRIFT")
    return errors


def _accept(result,prepared,rel_tolerance):
    accepted,status=effective_step_acceptance(
        bool(result.converged),result.residual_relative_to_initial,rel_tolerance,
        absolute_residual=result.admissible_residual_scaled,
        absolute_tolerance=prepared.settings.absolute_tolerance_scaled)
    if not accepted:
        raise ValueError("BATHE_SPATIAL_ACCEPTANCE_"+status)
    if prepared.method=="MVI":
        if result.mvi_certificate is None or not result.mvi_certificate.get("accepted"):
            raise ValueError("BATHE_MVI_FIXED_RANK_CERTIFICATE_FAILED")


def run_bathe_case(case,case_path,*,prepared=None):
    from cylinder_reference_case import prepare_reference_case
    if str(case.get("calculation_kind","")).upper()!="DYNAMIC" or str(
        (case.get("dynamics") or {}).get("integrator","")).upper()!="BATHE":
        raise ValueError("BATHE_REQUIRES_EXPLICIT_DYNAMIC_CASE_SELECTION")
    prepared=(prepare_reference_case(case,base_dir=Path(case_path).parent)
              if prepared is None else prepared)
    if prepared.calculation_kind!="DYNAMIC" or prepared.method not in ("MVI","FDM"):
        raise ValueError("BATHE_REQUIRES_VARIATIONAL_MVI_OR_FDM")
    matrix_free=(prepared.fdm_backend=="MATRIX_FREE_GMRES")
    if matrix_free and prepared.method!="FDM":
        raise ValueError("BATHE_MATRIX_FREE_IS_FDM_ONLY")
    dyn=prepared.dynamics
    settings=dyn.settings
    if abs(settings.beta-.25)>1e-13 or abs(settings.gamma-.5)>1e-13:
        raise ValueError("BATHE_NEWMARK_PREDICTOR_REQUIRES_BETA_025_GAMMA_05")
    if dyn.mass_damping_rate_per_s or dyn.eta_s:
        # The mechanical algebra works with damping, but energetic/phase
        # regression has not yet verified this physical extension.
        raise ValueError("BATHE_DAMPED_PHYSICS_NOT_YET_VERIFIED")
    out=Path(case["output_dir"])
    if not out.is_absolute():out=Path(case_path).parent/out
    out=out.resolve()
    restart=case["dynamics"].get("bathe_resume_accepted",False)
    if type(restart) is not bool:
        raise ValueError("BATHE_RESTART_FLAG_MUST_BE_BOOLEAN")
    existing=out.exists() and any(out.iterdir())
    if existing and not restart:
        raise ValueError("OUTPUT_DIR_NOT_EMPTY: Bathe results cannot mix with Newmark")
    out.mkdir(parents=True,exist_ok=True)
    if not existing:
        _json(out/"REFERENCE_CASE.json",case)
    f=prepared.problem.functional
    q=prepared.mechanical.copy()
    p=None if prepared.electrical is None else prepared.electrical.copy()
    v=np.zeros_like(q)
    if prepared.initial_acceleration is None:
        raise ValueError("BATHE_CONSISTENT_INITIAL_ACCELERATION_REQUIRED")
    a=np.asarray(prepared.initial_acceleration,float).copy()
    if matrix_free:
        from cylinder_variational_boundary import compile_boundary_program
        edge=compile_boundary_program(f,q,p,spec=prepared.problem.spec,
            calculation_kind="DYNAMIC",time_s=settings.t_start_s)
    else:
        edge=prepared.problem.assemble(q,p,time_s=settings.t_start_s,
            acceleration=np.zeros_like(q),density_kg_m3=dyn.density_kg_m3).edge
    trace=(sparse_mechanical_trace_matrix if matrix_free
           else mechanical_trace_matrix)
    initial_trace=trace(edge.constraints,q.size)
    if np.max(abs(initial_trace[0]@q.ravel()-initial_trace[1]),initial=0.)>1e-9:
        raise ValueError("BATHE_INITIAL_SUPPORT_INCONSISTENT")
    for value in (v,a):
        if np.max(abs(initial_trace[0]@value.ravel()),initial=0.)>1e-7:
            raise ValueError("BATHE_INITIAL_SUPPORT_VELOCITY_ACCELERATION_INCONSISTENT")
    if not existing:
        _npz(out/"REFERENCE_DYNAMIC_INITIAL.npz",time_s=np.array(settings.t_start_s),
             mechanical=q,velocity=v,acceleration=a,**({} if p is None else {"electrical":p}))
    prev=None
    debug_restart=(case["dynamics"].get("bathe_debug_cold_start_each_full_step",False))
    if type(debug_restart) is not bool:
        raise ValueError("BATHE_DEBUG_COLD_START_FLAG_MUST_BE_BOOLEAN")
    summary={"solver_backend":"VARIATIONAL_REFERENCE","calculation_kind":"DYNAMIC",
        "integrator":"BATHE","integrator_order":2,"experimental":True,
        "method":prepared.method,"theory":f.theory,"geometry_model":case["geometry_model"],
        "density_kg_m3":dyn.density_kg_m3,"fdm_backend":prepared.fdm_backend,"rank":(prepared.mvi_settings.rank if prepared.method=="MVI" else None),
        "status":"NOT_STARTED","converged":False,"accepted_steps":0,
        "requested_steps":settings.n_steps,"steps":[],
        "time_step_s":settings.dt_s,"start_time_s":settings.t_start_s,
        "gamma_bathe":BATHE_GAMMA,"debug_cold_start_each_full_step":debug_restart,
        "time_step_convergence_verified":False,
        "spatial_rank_convergence_verified":False}
    all_history=[]
    start_step=1
    if existing:
        from cylinder_bathe_checkpoint import load_checked_bathe_checkpoint
        state,summary,all_history,prev,start_step=load_checked_bathe_checkpoint(
            out,case,prepared,dyn,q0=q,p0=p,v0=v,a0=a,
            initial_trace=initial_trace,debug_restart=debug_restart)
        q,p,v,a=state
        if debug_restart:
            prev=None
    else:
        _json(out/"FINAL_RUN_SUMMARY.json",summary)
    for i in range(start_step,settings.n_steps+1):
        t0=settings.t_start_s+(i-1)*settings.dt_s
        def loads(t):
            factor=dyn.time_law.value(t)
            return factor*prepared.dead_load,(
                None if prepared.free_charge is None else factor*prepared.free_charge)
        try:
            result=solve_bathe_variational_step(
                prepared.problem,q,p,v,a,time_s=t0,h=settings.dt_s,
                density_kg_m3=dyn.density_kg_m3,loads_at=loads,
                method=prepared.method,settings=prepared.settings,
                fdm_backend=prepared.fdm_backend,
                mvi_settings=prepared.mvi_settings,
                previous_mvi_result=(None if debug_restart else prev))
            t1=t0+BATHE_GAMMA*settings.dt_s
            t2=t0+settings.dt_s
            if abs(result.time_first_s-t1)>1e-12 or abs(result.time_final_s-t2)>1e-12:
                raise ValueError("BATHE_PHYSICAL_TIME_DRIFT")
            first,final=result.first,result.final
            _accept(first,prepared,settings.residual_tolerance_relative)
            _accept(final,prepared,settings.residual_tolerance_relative)
            first_trace=_check_trace(prepared.problem,q,p,first,t1,initial_trace,matrix_free=matrix_free)
            final_trace=_check_trace(prepared.problem,q,p,final,t2,initial_trace,matrix_free=matrix_free)
            for stage,state,t in (("FIRST",first,t1),("FINAL",final,t2)):
                fields={"time_s":np.array(t),"mechanical":state.mechanical,
                     "velocity":state.velocity,"acceleration":state.acceleration,
                     "load_factor":np.array(dyn.time_law.value(t)),
                     "full_residual":state.full_residual,
                     "constraint_reaction":state.constraint_reaction}
                if state.electrical is not None:fields["electrical"]=state.electrical
                if prepared.method=="MVI":
                    fields.update(x_profiles=state.x_profiles,
                        y_profiles=state.y_profiles,state_scales=state.scales)
                name=(f"BATHE_SUBSTEP_{i:04d}_GAMMA.npz" if stage=="FIRST"
                      else f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz")
                _npz(out/name,**fields)
            # CUF coefficients are grouped [U_0..U_p, V_0..V_p, W_0..W_p].
            # The physical midsurface radial displacement is W_0, not W_p.
            radial=2*(int(f.theory[-1])+1) if f.theory.startswith("CUF_") else 2
            entry={"index":i,"time_s":t2,
                "time_gamma_s":t1,
                "load_factor":float(dyn.time_law.value(t2)),
                "status":final.status,"converged":True,
                "debug_cold_start_each_full_step":debug_restart,
                "first_substep_status":first.status,
                "first_substep_trace":first_trace,
                "final_trace":final_trace,
                "bathe_kinematic_defects":result.coefficient.kappa,
                "iterations":first.iterations+final.iterations,
                "sweeps":first.sweeps+final.sweeps,
                "first_substep_file":f"BATHE_SUBSTEP_{i:04d}_GAMMA.npz",
                "state_file":f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz",
                "w_middle_abs_max_m":float(np.max(abs(final.mechanical[radial]))),
                "velocity_abs_max_m_per_s":float(np.max(abs(final.velocity))),
                "acceleration_abs_max_m_per_s2":float(np.max(abs(final.acceleration)))}
            all_history.append(entry)
            summary["steps"].append(entry)
            q,p,v,a=final.mechanical,final.electrical,final.velocity,final.acceleration
            if prepared.method=="MVI":prev=final
            summary["accepted_steps"]=i
            summary["converged"]=i==settings.n_steps
            summary["status"]="CONVERGED" if summary["converged"] else "IN_PROGRESS"
            _json(out/"FINAL_RUN_SUMMARY.json",summary)
            print(f"BATHE_{prepared.method} rank={summary['rank']} step={i}/{settings.n_steps} "
                  f"t={t2:.9g} s first={first.status} final={final.status}",flush=True)
        except Exception as exc:
            summary["status"]="BATHE_STEP_REJECTED"
            summary["failed_step_index"]=i
            summary["failed_step_code"]=type(exc).__name__
            summary["failed_step_message"]=str(exc)
            _json(out/"FINAL_RUN_SUMMARY.json",summary)
            raise
    if len(all_history)!=settings.n_steps or not summary["converged"]:
        raise ValueError("BATHE_HISTORY_NOT_ALL_STEPS_ACCEPTED")
    with (out/"REFERENCE_DYNAMIC_HISTORY.csv").open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(all_history[0]))
        writer.writeheader();writer.writerows(all_history)
    return summary

__all__=("run_bathe_case",)
