"""Own dynamic Newmark X/Y subspace Newton for coupled G3D/Gz MVI.

This file does not import variational FDM, its Newton inverse,
or an MKR spatial operator. A full nodal solve is forbidden:
the caller MUST provide separated frozen-partner directions.
Constitutive enthalpy and geometric boundary traces remain shared
with the corresponding continuous physical model; this is a numerical
kernel separation, not a claim of independent coupled spatial assembly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from cylinder_boundary_spatial import affine_trace_space
from cylinder_residual_norms import admissible_nodal_inf_norm
from cylinder_mvi_subspace_newton import (
    MVIHalfstepSettings, mvi_settings_from_options,
    mvi_state_scales, mvi_checked_reduced_tangent,
)


@dataclass(frozen=True)
class MVIDynamicHalfstepResult:
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


def mvi_solve_dynamic_subspace(problem, mechanical, electrical, *, q_predictor,
                           velocity_predictor, time_s, dt_s, beta, gamma,
                           density_kg_m3, dead_load=None, free_charge=None,
                           settings=None, directions=None, on_iteration=None,
                           mass_damping_rate_per_s=0.0, eta_s=0.0,
                           krylov_settings=None):
    """Newton solve in the selected exact admissible spatial subspace.

    krylov_settings is opt-in and applies only to MVI frozen-partner
    directions when the reduced dimension exceeds the dense safety cap.
    """
    if directions is None:
        raise ValueError("MVI_OWN_DYNAMIC_REQUIRES_XY_DIRECTIONS")
    directions=np.asarray(directions,float)
    if (directions.ndim!=2 or not directions.shape[1] or
            not np.isfinite(directions).all()):
        raise ValueError("MVI_OWN_DYNAMIC_BAD_XY_DIRECTIONS")
    if problem.kind!="DYNAMIC":raise ValueError("dynamic solve requires DYNAMIC problem")
    settings=mvi_settings_from_options(settings)

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
    scales=mvi_state_scales(f,q,p,settings)
    zero_acceleration=np.zeros_like(q)
    first=problem.assemble(q,p,dead_load=dead_load,free_charge=free_charge,time_s=time_s,
        acceleration=zero_acceleration,density_kg_m3=density_kg_m3)
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    # A frozen MVI direction can satisfy the clamped trace exactly.
    # Its computed C*T then contains only roundoff. Do not mistake those
    # roundoff entries for additional independent prescribed constraints.
    # Retain the strict physical trace-defect check inside affine_trace_space.
    trace_rank_floor = (0.0 if directions is None else
        1.e-10 * float(np.max(np.linalg.norm(directions, axis=0), initial=0.0)))
    space=affine_trace_space(first.space.matrix*scales[None,:],first.space.target,
                             first.state/scales,directions=directions,
                             constraint_rank_floor=trace_rank_floor)
    b=space.basis
    use_krylov=b.shape[1]>settings.max_free_dofs and krylov_settings is not None
    if b.shape[1]>settings.max_free_dofs and not use_krylov:
        raise ValueError(f"reference dense solve has {b.shape[1]} free DOFs; limit={settings.max_free_dofs}")
    if krylov_settings is not None and directions is None:
        raise ValueError("MVI_KRYLOV_REQUIRES_FROZEN_PARTNER_DIRECTIONS")
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
        if use_krylov:
            # The action stays strictly in B: no full-nodal FDM search
            # direction, no enlarged rank, no artificial stiffness.
            from scipy.sparse.linalg import LinearOperator, gmres

            count=[0]
            class _BudgetExceeded(RuntimeError):
                pass

            def action(delta):
                if count[0]>=krylov_settings.dynamic_krylov_max_matvecs:
                    raise _BudgetExceeded()
                count[0]+=1
                nodal=b@delta
                mag=float(np.linalg.norm(nodal))
                if mag<1.e-30:
                    return np.zeros_like(delta)
                h=settings.tangent_step_scaled/mag
                for _ in range(settings.max_halvings+1):
                    try:
                        gp=evaluate(v+h*nodal)[2]
                        gm=evaluate(v-h*nodal)[2]
                        return (gp-gm)/(2*h)
                    except ValueError as exc:
                        if "INVALID_" not in str(exc):raise
                        h*=.5
                raise ValueError("INVALID_TANGENT_PERTURBATION")

            operator=LinearOperator((len(g),len(g)),matvec=action,dtype=float)
            try:
                correction,info=gmres(
                    operator,-g,rtol=krylov_settings.dynamic_krylov_rtol,
                    atol=0.,restart=krylov_settings.dynamic_krylov_restart,
                    maxiter=max(1,(krylov_settings.dynamic_krylov_max_matvecs //
                                    krylov_settings.dynamic_krylov_restart)),
                    callback_type="pr_norm")
                if info!=0 or not np.isfinite(correction).all():
                    status="MVI_KRYLOV_LINEAR_NOT_CONVERGED";break
                # Fail closed: GMRES claims are checked using this actual
                # projected physical Jacobian-vector action.
                true_linear=float(np.linalg.norm(action(correction)+g))
                if true_linear>max(
                    1.e-11,2*krylov_settings.dynamic_krylov_rtol*
                    float(np.linalg.norm(g))):
                    status="MVI_KRYLOV_LINEAR_NOT_CONVERGED";break
            except _BudgetExceeded:
                status="MVI_KRYLOV_MATVEC_BUDGET_EXCEEDED";break
            record["linear_solver"]="MVI_FROZEN_PARTNER_GMRES"
            record["krylov_matvecs"]=count[0]
        else:
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
                correction,estimate=mvi_checked_reduced_tangent(tangent,g)
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
    return MVIDynamicHalfstepResult(converged,status,q,p,velocity,acceleration,r.copy(),reaction,
        norm,norm/initial if initial else 0.,defect,tuple(history),iterations)



__all__=("MVIDynamicHalfstepResult","mvi_solve_dynamic_subspace")
