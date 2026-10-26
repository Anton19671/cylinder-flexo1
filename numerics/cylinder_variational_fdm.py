#!/usr/bin/env python3
"""Dense reference Newton solve of constrained shell enthalpy stationarity.

Finite differences form the tangent of the assembled first variation, not a
different strong PDE. Electrical unknowns are saddle variables: solve the
coupled indefinite system and globalize on residual, never energy decrease.
This small-grid backend is explicit and never substitutes for MVI.
"""
from dataclasses import dataclass
import numpy as np

from cylinder_boundary_spatial import affine_trace_space
from cylinder_variational_boundary import combine_functional_and_boundary


@dataclass(frozen=True)
class StaticFDMSettings:
    relative_tolerance: float = 1.e-7
    absolute_tolerance_scaled: float = 1.e-12
    tangent_step_scaled: float = 2.e-6
    max_iterations: int = 15
    max_halvings: int = 16
    potential_scale_v: float = 1.
    max_free_dofs: int = 256

    def __post_init__(self):
        for key in ("relative_tolerance", "absolute_tolerance_scaled",
                    "tangent_step_scaled", "potential_scale_v"):
            value=getattr(self,key)
            if isinstance(value,bool):raise ValueError(key+": boolean is not numeric")
            try:value=float(value)
            except (TypeError,ValueError) as exc:raise ValueError(key+": finite number required") from exc
            if not np.isfinite(value) or value<0 or (
                    key!="absolute_tolerance_scaled" and value==0):
                raise ValueError(key+" must be finite and positive (absolute tolerance may be zero)")
            object.__setattr__(self,key,value)
        for key in ("max_iterations","max_halvings","max_free_dofs"):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(key+" must be a positive integer")
        if self.max_free_dofs>512:
            raise ValueError("dense reference solver is limited to 512 free DOFs")


@dataclass(frozen=True)
class StaticFDMResult:
    converged: bool
    status: str
    mechanical: np.ndarray
    electrical: np.ndarray | None
    total_potential_j: float
    full_residual: np.ndarray
    constraint_reaction: np.ndarray
    admissible_residual_scaled: float
    residual_relative_to_initial: float
    constraint_defect_scaled: float
    history: tuple
    iterations: int


def state_scales(functional, mechanical, electrical, settings):
    """Physical units of one dimensionless nodal coefficient."""
    q=np.asarray(mechanical)
    scales=np.full(q.size+(0 if electrical is None else np.size(electrical)),functional.h)
    if functional.theory in ("FSDT","REDDY"):
        scales[3*np.prod(functional.ops.shape):q.size]=1.
    if electrical is not None:scales[q.size:]=settings.potential_scale_v
    return scales


def solve_static_subspace(problem, mechanical, electrical=None, *, dead_load=None,
                          free_charge=None, settings=None, on_iteration=None,
                          directions=None):
    """Solve selected admissible nodal equations with fixed STATIC boundary data.

    Translation/CUF coefficients are scaled by h, director rotations by 1 rad,
    electric coefficients by potential_scale_v. Residual is the derivative of
    Pi/(E*h*area) in those scaled coordinates. Reported absolute tolerance has
    this dimensionless meaning. No eigenvalue clipping or tangent regularizing.
    """
    if problem.kind!="STATIC":
        raise ValueError("static subspace solve requires STATIC; no dynamic substitution")
    settings=StaticFDMSettings() if settings is None else settings
    if not isinstance(settings,StaticFDMSettings):
        raise TypeError("settings must be StaticFDMSettings")
    f=problem.functional
    q=np.asarray(mechanical,float)
    p=None if electrical is None else np.asarray(electrical,float)
    first=problem.assemble(q,p,dead_load=dead_load,free_charge=free_charge)
    scales=state_scales(f,q,p,settings)
    energy_scale=float(f.material.young_pa*f.h*np.sum(f.W))
    if not np.isfinite(energy_scale) or energy_scale<=0:
        raise ValueError("invalid mechanical energy scale")
    # directions are dimensionless nodal variations. None selects the full
    # space; an MVI caller supplies only its frozen-partner profile directions.
    space=affine_trace_space(first.space.matrix*scales[None,:],
                             first.space.target,first.state/scales,directions=directions)
    b=space.basis
    if b.shape[1]>settings.max_free_dofs:
        raise ValueError(f"reference dense solve has {b.shape[1]} free DOFs; limit={settings.max_free_dofs}")

    def evaluate(v):
        u=scales*v
        qm=u[:q.size].reshape(q.shape)
        pm=None if p is None else u[q.size:].reshape(p.shape)
        energy,rm,re,_=combine_functional_and_boundary(
            f,qm,pm,bc_spec=problem.spec,dead_load=dead_load,free_charge=free_charge)
        r=rm.ravel() if re is None else np.concatenate((rm.ravel(),re.ravel()))
        g=b.T@(scales*r)/energy_scale
        if not np.isfinite(energy) or not np.isfinite(r).all():
            raise ValueError("nonfinite stationary functional/residual")
        return energy,r,g

    v=space.origin.copy()
    energy,r,g=evaluate(v)
    initial_norm=float(np.max(abs(g),initial=0))
    tolerance=settings.absolute_tolerance_scaled+settings.relative_tolerance*initial_norm
    history=[]
    status="MAX_ITERATIONS"
    converged=False
    iterations=0
    for iteration in range(settings.max_iterations+1):
        norm=float(np.max(abs(g),initial=0))
        record={"iteration":iteration,"residual_scaled":norm,
                "residual_relative_to_initial":norm/initial_norm if initial_norm else 0.,
                "potential_j":float(energy)}
        history.append(record)
        if on_iteration is not None:on_iteration(dict(record))
        if norm<=tolerance:
            converged=True;status="CONVERGED";break
        if iteration==settings.max_iterations:break
        tangent=np.empty((len(g),len(g)))
        for j,direction in enumerate(b.T):
            step=settings.tangent_step_scaled
            for attempt in range(settings.max_halvings+1):
                try:
                    gp=evaluate(v+step*direction)[2]
                    gm=evaluate(v-step*direction)[2]
                    break
                except ValueError as exc:
                    if "INVALID_" not in str(exc):raise
                    step*=.5
            else:
                status="INVALID_TANGENT_PERTURBATION";break
            tangent[:,j]=(gp-gm)/(2*step)
        if status=="INVALID_TANGENT_PERTURBATION":break
        record["tangent_symmetry_relative"]=float(
            np.linalg.norm(tangent-tangent.T)/max(np.linalg.norm(tangent),1.e-300))
        record["tangent_condition"]=float(np.linalg.cond(tangent))
        if (not np.isfinite(record["tangent_condition"]) or
                record["tangent_condition"]>1./np.finfo(float).eps):
            status="SINGULAR_TANGENT_CHECK_SUPPORTS_AND_GAUGE";break
        try:correction=np.linalg.solve(tangent,-g)
        except np.linalg.LinAlgError:
            status="SINGULAR_TANGENT_CHECK_SUPPORTS_AND_GAUGE";break
        merit=float(np.max(abs(g),initial=0))
        for halving in range(settings.max_halvings+1):
            alpha=2.**(-halving)
            trial=v+b@(alpha*correction)
            try:trial_energy,trial_r,trial_g=evaluate(trial)
            except ValueError as exc:
                if "INVALID_" not in str(exc):raise
                continue
            if (np.max(abs(trial_g),initial=0)<=(1.-1.e-4*alpha)*merit or
                    np.max(abs(trial_g),initial=0)<=tolerance):
                record["step_factor"]=alpha
                record["increment_scaled"]=float(np.max(abs(alpha*correction),initial=0))
                v,energy,r,g=trial,trial_energy,trial_r,trial_g
                iterations+=1
                break
        else:
            status="RESIDUAL_LINE_SEARCH_FAILED";break
    # R = applied generalized reaction + admissible imbalance. A raw zero
    # residual at fixed displacement/electrode coefficients is not required.
    scaled_r=scales*r
    admissible=b@(b.T@scaled_r)
    reaction=(scaled_r-admissible)/scales
    defect=float(np.max(abs(space.defect(v)),initial=0))
    if defect>1.e-9:
        converged=False;status="BOUNDARY_CONSTRAINT_DRIFT"
    u=scales*v
    return StaticFDMResult(converged,status,u[:q.size].reshape(q.shape),
        None if p is None else u[q.size:].reshape(p.shape),float(energy),r.copy(),reaction,
        float(np.max(abs(g),initial=0)),
        float(np.max(abs(g),initial=0))/initial_norm if initial_norm else 0.,
        defect,tuple(history),iterations)


def solve_static_fdm(problem, mechanical, electrical=None, *, dead_load=None,
                     free_charge=None, settings=None, on_iteration=None):
    """Full admissible nodal Newton solve; never selects an MVI subspace."""
    return solve_static_subspace(problem,mechanical,electrical,dead_load=dead_load,
        free_charge=free_charge,settings=settings,on_iteration=on_iteration)


__all__=("StaticFDMSettings","StaticFDMResult","solve_static_fdm",
         "solve_static_subspace","state_scales")
