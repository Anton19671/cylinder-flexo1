#!/usr/bin/env python3
"""Small-grid coupled Newmark solves in full nodal or separated MVI spaces."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
from cylinder_variational_fdm import StaticFDMSettings, state_scales
from cylinder_variational_mvi import (
    StaticMVISettings, reconstruct_profiles, _initial_profiles, _normalize_partner,
)


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
    initial=float(np.max(abs(g),initial=0))
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    history=[];iterations=0;status="MAX_ITERATIONS";converged=False
    for iteration in range(settings.max_iterations+1):
        norm=float(np.max(abs(g),initial=0))
        record={"iteration":iteration,"residual_scaled":norm,
                "residual_relative_to_initial":norm/initial if initial else 0.}
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
        record["tangent_condition"]=float(np.linalg.cond(tangent))
        if not np.isfinite(record["tangent_condition"]) or record["tangent_condition"]>1/np.finfo(float).eps:
            status="SINGULAR_TANGENT_CHECK_SUPPORTS_AND_GAUGE";break
        try:correction=np.linalg.solve(tangent,-g)
        except np.linalg.LinAlgError:
            status="SINGULAR_TANGENT_CHECK_SUPPORTS_AND_GAUGE";break
        merit=float(np.max(abs(g),initial=0))
        for halving in range(settings.max_halvings+1):
            alpha=2.**(-halving);trial=v+b@(alpha*correction)
            try:data=evaluate(trial)
            except ValueError as exc:
                if "INVALID_" not in str(exc):raise
                continue
            trial_g=data[2]
            if (np.max(abs(trial_g),initial=0)<=(1.-1.e-4*alpha)*merit or
                    np.max(abs(trial_g),initial=0)<=tolerance):
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
    norm=float(np.max(abs(g),initial=0))
    return DynamicSpatialResult(converged,status,q,p,velocity,acceleration,r.copy(),reaction,
        norm,norm/initial if initial else 0.,defect,tuple(history),iterations)


def solve_dynamic_fdm(problem, mechanical, electrical=None, **kwargs):
    return solve_dynamic_subspace(problem,mechanical,electrical,**kwargs)


def solve_dynamic_mvi(problem, mechanical, electrical=None, *, q_predictor,
                      velocity_predictor, time_s, dt_s, beta, gamma,
                      density_kg_m3, dead_load=None, free_charge=None,
                      settings=None, mvi_settings=None, on_iteration=None,
                      mass_damping_rate_per_s=0.0, eta_s=0.0):
    """Complete frozen-partner X/Y cycle for one effective Newmark step."""
    settings=StaticFDMSettings() if settings is None else settings
    mvi_settings=StaticMVISettings() if mvi_settings is None else mvi_settings
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
            values.append(float(np.max(abs(space.basis.T@(scales*r)/energy_scale),initial=0)))
        return max(values)

    history=[];iterations=0;sweeps=0;status="MAX_SWEEPS";converged=False
    v,assembled,r,g,qm,pm,velocity,acceleration=full_data()
    initial=float(np.max(abs(g),initial=0));tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial
    projected=projected_norm(v,r)
    def record(sweep):
        row={"event":"sweep","sweep":sweep,"iteration":iterations,
             "residual_scaled":float(np.max(abs(g),initial=0)),
             "residual_relative_to_initial":float(np.max(abs(g),initial=0))/initial if initial else 0.,
             "projected_residual_scaled":projected,
             "constraint_defect_scaled":float(np.max(abs(full.defect(v)),initial=0))}
        history.append(row)
        if on_iteration is not None:on_iteration(dict(row))
    record(0)
    if initial<=tolerance:converged=True;status="CONVERGED"
    for sweep in range(1,mvi_settings.max_sweeps+1):
        if converged:break
        failed=False
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
        v,assembled,r,g,qm,pm,velocity,acceleration=full_data();projected=projected_norm(v,r);record(sweep)
        if failed:break
        if np.max(abs(g),initial=0)<=tolerance:converged=True;status="CONVERGED";break
        if projected<=tolerance:status="PROJECTED_ONLY_FULL_RESIDUAL_NOT_CONVERGED";break
    defect=float(np.max(abs(full.defect(v)),initial=0))
    if defect>1.e-9:converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    scaled_r=scales*r;reaction=(scaled_r-full.basis@(full.basis.T@scaled_r))/scales
    norm=float(np.max(abs(g),initial=0))
    return DynamicSpatialResult(converged,status,qm,pm,velocity,acceleration,r.copy(),reaction,
        norm,norm/initial if initial else 0.,defect,tuple(history),iterations,sweeps,
        projected,projected<=tolerance,x.copy(),y.copy(),lift.reshape(shape),scales.reshape(shape))


__all__=("DynamicSpatialResult","solve_dynamic_subspace","solve_dynamic_fdm",
         "solve_dynamic_mvi")
