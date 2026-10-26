#!/usr/bin/env python3
"""Small-grid coupled Newmark solves in full nodal or separated MVI spaces."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
from cylinder_residual_norms import admissible_nodal_inf_norm, admissible_nodal_metrics
from cylinder_variational_fdm import StaticFDMSettings, state_scales, solve_checked_reduced_tangent
from cylinder_variational_mvi import (
    StaticMVISettings, reconstruct_profiles, _initial_profiles, _normalize_partner,
)
from cylinder_mvi_cold_start import seed_inactive_mvi_partners
from cylinder_mvi_acceptance import certify_mvi_solution


@dataclass(frozen=True)
class DynamicSpatialResult:
    converged: bool
    status: str
    mechanical: np.ndarray
    electrical: np.ndarray | None
    velocity: np.ndarray
    acceleration: np.ndarray
    full_residual: np.ndarray
    constraint_reaction: np.ndarray
    admissible_residual_scaled: float
    residual_relative_to_initial: float
    constraint_defect_scaled: float
    history: tuple
    iterations: int
    sweeps: int = 0
    projected_residual_scaled: float | None = None
    projected_equilibrium_satisfied: bool | None = None
    x_profiles: np.ndarray | None = None
    y_profiles: np.ndarray | None = None
    lift_scaled: np.ndarray | None = None
    scales: np.ndarray | None = None
    mvi_certificate: dict | None = None


def _effective_evaluator(problem, q_shape, p_shape, q_predictor, v_predictor,
                         *, time_s, dt_s, beta, gamma, density_kg_m3,
                         dead_load, free_charge, scales, basis, energy_scale,
                         mass_damping_rate_per_s=0.0, eta_s=0.0):
    q_size=int(np.prod(q_shape))
    coefficient=1./(beta*dt_s*dt_s)

    def evaluate(v):
        state=scales*v
        q=state[:q_size].reshape(q_shape)
        p=None if p_shape is None else state[q_size:].reshape(p_shape)
        acceleration=(q-q_predictor)*coefficient
        velocity=v_predictor+gamma*dt_s*acceleration
        _,residual,_=problem.evaluate_effective_residual(
            q,p,dead_load=dead_load,free_charge=free_charge,
            time_s=time_s,acceleration=acceleration,density_kg_m3=density_kg_m3,
            velocity=velocity if (mass_damping_rate_per_s or eta_s or
                                  problem.functional.theory=="KL") else None,
            mass_damping_rate_per_s=mass_damping_rate_per_s,
            eta_s=eta_s)
        assembled=None  # No boundary SVD during tangent/line-search evaluations.
        g=basis.T@(scales*residual)/energy_scale
        if not np.isfinite(residual).all() or not np.isfinite(g).all():
            raise ValueError("nonfinite dynamic residual")
        return assembled,residual,g,q,p,velocity,acceleration
    return evaluate


def solve_dynamic_subspace(problem, mechanical, electrical, *, q_predictor,
                           velocity_predictor, time_s, dt_s, beta, gamma,
                           density_kg_m3, dead_load=None, free_charge=None,
                           settings=None, directions=None, on_iteration=None,
                           mass_damping_rate_per_s=0.0, eta_s=0.0):
    """Newton solve of one effective Newmark residual in a chosen space."""
    if problem.kind!="DYNAMIC":raise ValueError("dynamic solve requires DYNAMIC problem")
    settings=StaticFDMSettings() if settings is None else settings
    if not isinstance(settings,StaticFDMSettings):raise TypeError("invalid Newton settings")
    for value,label in ((time_s,"time_s"),(dt_s,"dt_s"),(beta,"beta"),
                        (gamma,"gamma"),(density_kg_m3,"density_kg_m3")):
        if isinstance(value,bool) or not np.isfinite(float(value)):
            raise ValueError(label+": finite number required")
    if dt_s<=0 or beta<=0 or density_kg_m3<=0:raise ValueError("invalid dynamic parameter sign")
    q=np.asarray(mechanical,float);p=None if electrical is None else np.asarray(electrical,float)
    qpred=np.asarray(q_predictor,float);vpred=np.asarray(velocity_predictor,float)
    if qpred.shape!=q.shape or vpred.shape!=q.shape or not np.isfinite(qpred).all() or not np.isfinite(vpred).all():
        raise ValueError("Newmark predictors must match mechanical field")
    f=problem.functional
    scales=state_scales(f,q,p,settings)
    zero_acceleration=np.zeros_like(q)
    first=problem.assemble(q,p,dead_load=dead_load,free_charge=free_charge,time_s=time_s,
        acceleration=zero_acceleration,density_kg_m3=density_kg_m3)
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    space=affine_trace_space(first.space.matrix*scales[None,:],first.space.target,
                             first.state/scales,directions=directions)
    b=space.basis
    if b.shape[1]>settings.max_free_dofs:
        raise ValueError(f"reference dense solve has {b.shape[1]} free DOFs; limit={settings.max_free_dofs}")
    evaluate=_effective_evaluator(problem,q.shape,None if p is None else p.shape,qpred,vpred,
        time_s=float(time_s),dt_s=float(dt_s),beta=float(beta),gamma=float(gamma),
        density_kg_m3=float(density_kg_m3),dead_load=dead_load,free_charge=free_charge,
        scales=scales,basis=b,energy_scale=energy_scale,
        mass_damping_rate_per_s=mass_damping_rate_per_s,
            eta_s=eta_s)
    v=space.origin.copy()
    assembled,r,g,q,p,velocity,acceleration=evaluate(v)
    initial=admissible_nodal_inf_norm(b,g)
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    history=[];iterations=0;status="MAX_ITERATIONS";converged=False
    for iteration in range(settings.max_iterations+1):
        norm=admissible_nodal_inf_norm(b,g)
        record={"iteration":iteration,"residual_scaled":norm,
                "residual_relative_to_initial":norm/initial if initial else 0.,
                "basis_coordinate_residual_scaled":float(np.max(abs(g),initial=0))}
        history.append(record)
        if on_iteration is not None:on_iteration(dict(record))
        if norm<=tolerance:converged=True;status="CONVERGED";break
        if iteration==settings.max_iterations:break
        tangent=np.empty((len(g),len(g)))
        failed=False
        for j,direction in enumerate(b.T):
            step=settings.tangent_step_scaled
            for _ in range(settings.max_halvings+1):
                try:
                    gp=evaluate(v+step*direction)[2]
                    gm=evaluate(v-step*direction)[2]
                    break
                except ValueError as exc:
                    if "INVALID_" not in str(exc):raise
                    step*=.5
            else:status="INVALID_TANGENT_PERTURBATION";failed=True;break
            tangent[:,j]=(gp-gm)/(2*step)
        if failed:break
        record["tangent_symmetry_relative"]=float(
            np.linalg.norm(tangent-tangent.T)/max(np.linalg.norm(tangent),1.e-300))
        try:
            correction,estimate=solve_checked_reduced_tangent(tangent,g)
            record["tangent_condition"]=estimate
            record["tangent_condition_method"]="LU_1_NORM_ESTIMATE"
        except np.linalg.LinAlgError:
            record["tangent_condition"]=float("inf")
            status="SINGULAR_TANGENT_CHECK_SUPPORTS_AND_GAUGE";break
        merit=admissible_nodal_inf_norm(b,g)
        for halving in range(settings.max_halvings+1):
            alpha=2.**(-halving);trial=v+b@(alpha*correction)
            try:data=evaluate(trial)
            except ValueError as exc:
                if "INVALID_" not in str(exc):raise
                continue
            trial_g=data[2]
            trial_norm=admissible_nodal_inf_norm(b,trial_g)
            if (trial_norm<=(1.-1.e-4*alpha)*merit or
                    trial_norm<=tolerance):
                record["step_factor"]=alpha
                record["increment_scaled"]=float(np.max(abs(alpha*correction),initial=0))
                v=trial;assembled,r,g,q,p,velocity,acceleration=data
                iterations+=1;break
        else:status="RESIDUAL_LINE_SEARCH_FAILED";break
    scaled_r=scales*r
    admissible=b@(b.T@scaled_r)
    reaction=(scaled_r-admissible)/scales
    defect=float(np.max(abs(space.defect(v)),initial=0))
    if defect>1.e-9:converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    norm=admissible_nodal_inf_norm(b,g)
    return DynamicSpatialResult(converged,status,q,p,velocity,acceleration,r.copy(),reaction,
        norm,norm/initial if initial else 0.,defect,tuple(history),iterations)


def solve_dynamic_fdm(problem, mechanical, electrical=None, **kwargs):
    return solve_dynamic_subspace(problem,mechanical,electrical,**kwargs)


def solve_dynamic_mvi(problem, mechanical, electrical=None, *, q_predictor,
                      velocity_predictor, time_s, dt_s, beta, gamma,
                      density_kg_m3, dead_load=None, free_charge=None,
                      settings=None, mvi_settings=None, on_iteration=None,
                      mass_damping_rate_per_s=0.0, eta_s=0.0):
    """Complete rank-fixed frozen-partner X/Y cycle for a Newmark step."""
    if problem.kind!="DYNAMIC":
        raise ValueError("solve_dynamic_mvi requires DYNAMIC problem")
    settings=StaticFDMSettings() if settings is None else settings
    mvi_settings=StaticMVISettings() if mvi_settings is None else mvi_settings
    if not isinstance(settings,StaticFDMSettings):
        raise TypeError("invalid Newton settings")
    if not isinstance(mvi_settings,StaticMVISettings):
        raise TypeError("invalid MVI settings")
    q=np.asarray(mechanical,float);p=None if electrical is None else np.asarray(electrical,float)
    f=problem.functional;scales=state_scales(f,q,p,settings)
    first=problem.assemble(q,p,dead_load=dead_load,free_charge=free_charge,time_s=time_s,
        acceleration=np.zeros_like(q),density_kg_m3=density_kg_m3)
    c=first.space.matrix*scales[None,:];target=first.space.target
    full=affine_trace_space(c,target,np.zeros_like(scales))
    if full.basis.shape[1]>settings.max_free_dofs:raise ValueError("reference free-DOF limit exceeded")
    lift=full.origin.copy();state=first.state/scales
    variation=state-lift if np.max(abs(full.defect(state)),initial=0)<=1.e-12 else full.basis@(full.basis.T@state)
    nc=q.shape[0]+(0 if p is None else p.shape[0]);shape=(nc,*q.shape[1:])
    component_scale=np.maximum(np.linalg.norm(state.reshape(shape),axis=(1,2)),
                               np.linalg.norm(lift.reshape(shape),axis=(1,2)))
    x,y=_initial_profiles(variation.reshape(shape),mvi_settings.rank,component_scale)
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))

    def fields():
        v=lift+reconstruct_profiles(x,y).ravel();u=scales*v
        return v,u[:q.size].reshape(q.shape),None if p is None else u[q.size:].reshape(p.shape)
    def full_data():
        v,qm,pm=fields()
        evaluate=_effective_evaluator(problem,q.shape,None if p is None else p.shape,
            np.asarray(q_predictor,float),np.asarray(velocity_predictor,float),time_s=time_s,
            dt_s=dt_s,beta=beta,gamma=gamma,density_kg_m3=density_kg_m3,
            dead_load=dead_load,free_charge=free_charge,scales=scales,
            basis=full.basis,energy_scale=energy_scale,
            mass_damping_rate_per_s=mass_damping_rate_per_s,
            eta_s=eta_s)
        return (v,)+evaluate(v)
    def projected_norm(v,r):
        values=[]
        for axis in ("X","Y"):
            xx,yy=_normalize_partner(x,y,axis)
            directions=separated_profile_directions(xx,yy,axis)
            space=affine_trace_space(c,target,v,directions=directions)
            reduced=space.basis.T@(scales*r)/energy_scale
            values.append(admissible_nodal_inf_norm(space.basis,reduced))
        return max(values)

    history=[];iterations=0;sweeps=0;status="MAX_SWEEPS";converged=False
    v,assembled,r,g,qm,pm,velocity,acceleration=full_data()
    # Effective Newmark residual, NOT the static residual, seeds inactive
    # factors so nonaxisymmetric inertia/load cannot stall the first sweep.
    admissible=(full.basis@g).reshape(shape)
    x,y=seed_inactive_mvi_partners(x,y,admissible)
    v,assembled,r,g,qm,pm,velocity,acceleration=full_data()
    initial=admissible_nodal_inf_norm(full.basis,g)
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    projected=projected_norm(v,r)
    def record(sweep,previous_v=None):
        metrics=admissible_nodal_metrics(full.basis,g,q.size)
        row={"event":"sweep","sweep":sweep,"iteration":iterations,
             "residual_scaled":metrics["full"],
             "mechanical_residual_scaled":metrics["mechanical"],
             "electrical_residual_scaled":metrics["electrical"],
             "basis_coordinate_residual_scaled":metrics["basis_coordinates"],
             "residual_relative_to_initial":metrics["full"]/initial if initial else 0.,
             "projected_residual_scaled":projected,
             "sweep_field_increment_scaled":(
                 0. if previous_v is None else float(np.max(abs(v-previous_v),initial=0))),
             "constraint_defect_scaled":float(np.max(abs(full.defect(v)),initial=0))}
        history.append(row)
        if on_iteration is not None:on_iteration(dict(row))
    record(0)
    if initial<=tolerance:converged=True;status="CONVERGED"
    for sweep in range(1,mvi_settings.max_sweeps+1):
        if converged:break
        failed=False
        previous_v=v.copy()
        for axis in ("X","Y"):
            x,y=_normalize_partner(x,y,axis);v,qm,pm=fields()
            directions=separated_profile_directions(x,y,axis)
            def report(row):
                if on_iteration is not None:on_iteration({**row,"event":"halfstep","sweep":sweep,"direction":axis})
            result=solve_dynamic_subspace(problem,qm,pm,q_predictor=q_predictor,
                velocity_predictor=velocity_predictor,time_s=time_s,dt_s=dt_s,beta=beta,
                gamma=gamma,density_kg_m3=density_kg_m3,dead_load=dead_load,
                free_charge=free_charge,settings=settings,directions=directions,on_iteration=report,
                mass_damping_rate_per_s=mass_damping_rate_per_s,
            eta_s=eta_s)
            history.extend({**row,"event":"halfstep","sweep":sweep,"direction":axis}
                           for row in result.history)
            state=result.mechanical.ravel() if p is None else np.concatenate(
                (result.mechanical.ravel(),result.electrical.ravel()))
            delta=state/scales-v;increment=directions.T@delta
            if np.linalg.norm(delta-directions@increment)>1.e-10*max(1.,np.linalg.norm(delta)):
                raise ValueError("MVI correction escaped frozen-partner space")
            if axis=="X":x+=increment.reshape(x.shape)
            else:y+=increment.reshape(y.shape)
            iterations+=result.iterations
            if not result.converged:
                status="HALFSTEP_"+axis+"_"+result.status;failed=True;break
        sweeps=sweep
        v,assembled,r,g,qm,pm,velocity,acceleration=full_data();projected=projected_norm(v,r);record(sweep,previous_v)
        if failed:break
        if admissible_nodal_inf_norm(full.basis,g)<=tolerance:
            converged=True;status="CONVERGED";break
        # The X/Y projected residual is only a diagnostic. Never accept or
        # terminate a fixed-rank sweep while the full coupled admissible
        # mechanical/electric residual still exceeds tolerance.
        if projected<=tolerance:
            status="PROJECTED_ONLY_FULL_RESIDUAL_NOT_CONVERGED"
    if not converged and status=="MAX_SWEEPS":
        status="MAX_SWEEPS_FULL_RESIDUAL_NOT_CONVERGED"
    defect=float(np.max(abs(full.defect(v)),initial=0))
    if defect>1.e-9:converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    certificate=certify_mvi_solution(
        x_profiles=x,y_profiles=y,lift_scaled=lift.reshape(shape),
        scales=scales.reshape(shape),mechanical=qm,electrical=pm,
        full_basis=full.basis,reduced_residual=g,
        projected_residual_scaled=projected,
        full_tolerance_scaled=tolerance,constraint_defect_scaled=defect,
        solver_converged=converged)
    if converged and not certificate["accepted"]:
        converged=False
        status=("MVI_PROFILE_RECONSTRUCTION_DRIFT"
                if not certificate["profile_reconstruction_satisfied"]
                else "MVI_FULL_RESIDUAL_NOT_CONVERGED"
                if not certificate["full_equilibrium_satisfied"]
                else "BOUNDARY_CONSTRAINT_DRIFT")
    scaled_r=scales*r;reaction=(scaled_r-full.basis@(full.basis.T@scaled_r))/scales
    norm=certificate["full_admissible_residual_scaled"]
    return DynamicSpatialResult(converged,status,qm,pm,velocity,acceleration,r.copy(),reaction,
        norm,norm/initial if initial else 0.,defect,tuple(history),iterations,sweeps,
        projected,projected<=tolerance,x.copy(),y.copy(),lift.reshape(shape),
        scales.reshape(shape),certificate)


__all__=("DynamicSpatialResult","solve_dynamic_subspace","solve_dynamic_fdm",
         "solve_dynamic_mvi")
