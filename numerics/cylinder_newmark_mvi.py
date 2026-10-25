#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Newmark time stepping with explicit MVI-or-FDM spatial solver selection.

This is a *method-independent temporal integration core*, not an already
wired physical cylinder solver. The caller MUST supply the actual shell
internal residual, mass/damping actions, forcing and a convergent spatial
solver. In the MVI route that callback must perform its complete X/Y cycle
for EACH time step. No method fallback and no use of static load increments
as physical time is permitted. No artificial time-history data are made.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping
import numpy as np


CALCULATION_KINDS=("STATIC","DYNAMIC")
SPATIAL_METHODS=("MVI","FDM")


class DynamicInputError(ValueError):
    pass


class DynamicStepError(RuntimeError):
    pass


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise DynamicInputError(f"{name}: boolean is not a number")
    try:
        v=float(value)
    except (TypeError, ValueError) as exc:
        raise DynamicInputError(f"{name}: finite numeric value required") from exc
    if not isfinite(v):
        raise DynamicInputError(f"{name}: finite numeric value required")
    if positive and v<=0 or nonnegative and v<0:
        raise DynamicInputError(f"{name}: invalid sign")
    return v


@dataclass(frozen=True)
class NewmarkSettings:
    dt_s: float
    n_steps: int
    beta: float=0.25
    gamma: float=0.5
    t_start_s: float=0.0
    residual_tolerance_relative: float=1.e-6

    def __post_init__(self):
        for key in ("dt_s","beta","gamma","t_start_s","residual_tolerance_relative"):
            v=_real(getattr(self,key),key,
                    positive=key in ("dt_s","beta","residual_tolerance_relative"))
            object.__setattr__(self,key,v)
        if isinstance(self.n_steps,bool) or not isinstance(self.n_steps,(int,np.integer)) or self.n_steps<1:
            raise DynamicInputError("n_steps must be a positive integer")
        if not (0.5<=self.gamma<=1.0 and self.beta>=0.25*(self.gamma+0.5)**2):
            raise DynamicInputError(
                "Use an unconditionally stable Newmark pair: 0.5 <= gamma <= 1, "
                "beta >= 0.25*(gamma+0.5)^2"
            )


def calculation_kind(case):
    kind=str(case.get("calculation_kind","STATIC")).strip().upper()
    if kind not in CALCULATION_KINDS:
        raise DynamicInputError("calculation_kind must be STATIC or DYNAMIC")
    return kind


def validate_dynamic_case(case):
    """Validate dynamics configuration without claiming the shell PDE is wired.

    Density is always specified in SI kg/m3 independent of the GUI geometry
    system. Initial conditions, forcing time law and actual spatial operators
    still belong to the concrete physical shell backend.
    """
    if calculation_kind(case)!="DYNAMIC":
        raise DynamicInputError("DYNAMIC config requested for a STATIC case")
    method=str(case.get("method","")).strip().upper()
    if method not in SPATIAL_METHODS:
        raise DynamicInputError("Choose MVI or FDM as the spatial solver")
    d=case.get("dynamics")
    if not isinstance(d,dict):
        raise DynamicInputError("DYNAMIC requires dynamics object")
    if "time_step_s" not in d or "n_steps" not in d or "density_kg_m3" not in d:
        raise DynamicInputError("Specify time_step_s, n_steps and density_kg_m3")
    value=d["n_steps"]
    if isinstance(value,bool) or str(value).strip()=="":
        raise DynamicInputError("n_steps must be a positive integer")
    try:
        steps=int(value)
    except (TypeError,ValueError) as exc:
        raise DynamicInputError("n_steps must be a positive integer") from exc
    if str(steps)!=str(value).strip():
        raise DynamicInputError("n_steps must be an integer")
    settings=NewmarkSettings(
        dt_s=_real(d["time_step_s"],"time_step_s",positive=True),
        n_steps=steps,
        beta=_real(d.get("newmark_beta",0.25),"newmark_beta",positive=True),
        gamma=_real(d.get("newmark_gamma",0.5),"newmark_gamma",positive=True),
        t_start_s=_real(d.get("start_time_s",0.),"start_time_s"),
        residual_tolerance_relative=_real(
            d.get("residual_tolerance_relative",1.e-6),"residual_tolerance_relative",
            positive=True
        )
    )
    density=_real(d["density_kg_m3"],"density_kg_m3",positive=True)
    damping=str(d.get("damping","NONE")).strip().upper()
    if damping not in ("NONE","MASS_PROPORTIONAL","EXTERNAL_OPERATOR"):
        raise DynamicInputError(
            "damping must be NONE, MASS_PROPORTIONAL or EXTERNAL_OPERATOR")
    # MASS_PROPORTIONAL represents the mechanically consistent Rayleigh term
    # C = alpha_M * M, where alpha_M has dimensions 1/s.  This is NOT
    # micro.eta_s (material gradient viscosity), nor electrical damping.
    damping_key="mass_damping_rate_per_s"
    if damping=="MASS_PROPORTIONAL":
        if damping_key not in d:
            raise DynamicInputError(
                "MASS_PROPORTIONAL requires mass_damping_rate_per_s")
        alpha_m=_real(d[damping_key],damping_key,positive=True)
    else:
        if damping_key in d:
            raise DynamicInputError(
                damping_key+" is only allowed with MASS_PROPORTIONAL")
        alpha_m=0.0
    return {
        "kind":"DYNAMIC","method":method,
        "time_step_s":settings.dt_s,"n_steps":settings.n_steps,
        "newmark_beta":settings.beta,"newmark_gamma":settings.gamma,
        "start_time_s":settings.t_start_s,
        "density_kg_m3":density,"damping":damping,
        "mass_damping_rate_per_s":alpha_m,
        "residual_tolerance_relative":settings.residual_tolerance_relative,
        "spatial_solver_status":"NOT_CONNECTED_TO_CYLINDER_PDE",
    }


def _state_vector(obj,label,shape=None):
    a=np.asarray(obj,float)
    if a.ndim!=1 or a.size==0 or not np.isfinite(a).all():
        raise DynamicInputError(f"{label} must be a finite nonempty 1D array")
    if shape is not None and a.shape!=shape:
        raise DynamicInputError(f"{label} shape mismatch")
    return a.copy()


@dataclass(frozen=True)
class TransientState:
    time_s: float
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        t=_real(self.time_s,"time_s")
        u=_state_vector(self.displacement,"displacement")
        v=_state_vector(self.velocity,"velocity",u.shape)
        a=_state_vector(self.acceleration,"acceleration",u.shape)
        object.__setattr__(self,"time_s",t)
        object.__setattr__(self,"displacement",u)
        object.__setattr__(self,"velocity",v)
        object.__setattr__(self,"acceleration",a)


@dataclass(frozen=True)
class SpatialStep:
    displacement: np.ndarray
    converged: bool
    xy_cycles: int=0
    label: str=""


@dataclass(frozen=True)
class AcceptedStep:
    index: int
    state: TransientState
    method: str
    full_residual_relative: float
    xy_cycles: int


@dataclass(frozen=True)
class EffectiveNewmarkProblem:
    """Mechanical residual R(u)=M*a(u)+C*v(u)+R_int(u)-F(t_new)."""

    time_s: float
    method: str
    predicted_displacement: np.ndarray
    predicted_velocity: np.ndarray
    dt_s: float
    beta: float
    gamma: float
    mass_action: Callable
    internal_force: Callable
    external_force: np.ndarray
    damping_action: Callable | None = None

    def kinematics(self,u):
        u=_state_vector(u,"trial displacement",self.predicted_displacement.shape)
        a=(u-self.predicted_displacement)/(self.beta*self.dt_s**2)
        v=self.predicted_velocity+self.gamma*self.dt_s*a
        return v,a

    def residual_terms(self,u):
        v,a=self.kinematics(u)
        mass=_state_vector(self.mass_action(a),"inertia force",a.shape)
        internal=_state_vector(self.internal_force(u,self.time_s),
                               "internal force",a.shape)
        damp=np.zeros_like(a) if self.damping_action is None else _state_vector(
            self.damping_action(v),"damping force",a.shape)
        return mass,internal,damp

    def residual(self,u):
        inertia,internal,damping=self.residual_terms(u)
        return inertia+internal+damping-self.external_force

    def full_residual_relative(self,u):
        inertia,internal,damping=self.residual_terms(u)
        remainder=inertia+internal+damping-self.external_force
        scale=max(1.0,*[float(np.max(np.abs(x))) for x in
            (inertia,internal,damping,self.external_force)])
        return float(np.max(np.abs(remainder))/scale)


def integrate_newmark(initial_state: TransientState, settings: NewmarkSettings, *,
                      method: str, spatial_solvers: Mapping[str,Callable],
                      mass_action: Callable, internal_force: Callable,
                      external_force: Callable, damping_action: Callable | None=None,
                      on_accepted_step: Callable | None=None):
    """Drive one selected spatial solver ON EVERY Newmark time step.

    spatial_solvers["MVI"] must actually solve the supplied effective residual
    by MVI directional cycles. Its callback signature is:
        solver(problem, initial_guess, step_index) -> SpatialStep
    A claimed MVI step must report >=1 XY cycle and a converged full residual.
    On an invalid step, raise immediately; do not publish a fabricated state.
    """
    if not isinstance(initial_state,TransientState):
        raise TypeError("initial_state must be TransientState")
    if not isinstance(settings,NewmarkSettings):
        raise TypeError("settings must be NewmarkSettings")
    if method not in SPATIAL_METHODS:
        raise DynamicInputError("method must be MVI or FDM")
    solver=spatial_solvers.get(method)
    if not callable(solver):
        raise DynamicInputError(
            f"{method} spatial solver is not connected; "
            "cannot substitute STATIC or another method"
        )
    if not all(callable(f) for f in (mass_action,internal_force,external_force)):
        raise DynamicInputError("physical mass, internal and forcing operators required")
    if damping_action is not None and not callable(damping_action):
        raise DynamicInputError("damping_action must be callable or None")
    state=initial_state
    dt,beta,gamma=settings.dt_s,settings.beta,settings.gamma
    for i in range(1,settings.n_steps+1):
        tp=state.time_s+dt
        u_pred=state.displacement+dt*state.velocity+dt**2*(.5-beta)*state.acceleration
        v_pred=state.velocity+dt*(1-gamma)*state.acceleration
        force=_state_vector(external_force(tp),"external force",state.displacement.shape)
        problem=EffectiveNewmarkProblem(
            tp,method,u_pred,v_pred,dt,beta,gamma,
            mass_action,internal_force,force,damping_action
        )
        response=solver(problem,u_pred.copy(),i)
        if not isinstance(response,SpatialStep):
            raise DynamicStepError(f"{method} step {i}: expected SpatialStep result")
        if response.converged is not True:
            raise DynamicStepError(f"{method} step {i}: spatial solve did not converge")
        if isinstance(response.xy_cycles,bool) or not isinstance(
            response.xy_cycles,(int,np.integer)
        ) or (method=="MVI" and response.xy_cycles<1):
            raise DynamicStepError(f"MVI step {i}: missing X/Y cycles")
        u=_state_vector(response.displacement,"accepted displacement",u_pred.shape)
        rel=problem.full_residual_relative(u)
        if rel>settings.residual_tolerance_relative:
            raise DynamicStepError(
                f"{method} step {i}: FULL effective residual {rel:.3e} "
                f"> {settings.residual_tolerance_relative:.3e}; step rejected"
            )
        v,a=problem.kinematics(u)
        state=TransientState(tp,u,v,a)
        accepted=AcceptedStep(i,state,method,rel,int(response.xy_cycles))
        if on_accepted_step is not None:
            on_accepted_step(accepted)
        yield accepted


__all__=("CALCULATION_KINDS","SPATIAL_METHODS","NewmarkSettings",
         "TransientState","SpatialStep","AcceptedStep",
         "EffectiveNewmarkProblem","DynamicInputError","DynamicStepError",
         "calculation_kind","validate_dynamic_case","integrate_newmark")
