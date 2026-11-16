"""Consistent initial acceleration and electric equilibrium for native strong MKR.

This module never invokes the MVI, reference variational FDM, or an automatic
time simulation. It solves the acceleration field at a GIVEN displacement
and velocity, together with the quasi-static electric potential.

Fixed-in-time essential support traces are required: their displacement
values must hold at t0, their initial velocity traces must vanish and their
initial acceleration traces are zero. Moving supports require a distinct
time-dependent boundary implementation, and are deliberately rejected here.
Physical differential half-orders and natural ends are caller-certified.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from cylinder_direct_strong_boundary import (
    _condition, assemble_physical_end_rows,
)
from cylinder_direct_strong_fdm import direct_coefficient_derivative
from cylinder_direct_strong_newton import (
    StrongNewtonResult, StrongNewtonSettings, solve_strong_nonlinear_equations,
)
from cylinder_native_mkr_newmark import (
    NativeNewmarkState, NativeStrongNewmark, _field, _finite_number,
)


@dataclass(frozen=True)
class NativeInitialStateResult:
    """A failed initialization cannot be mistaken for an accepted state."""
    state: NativeNewmarkState | None
    nonlinear: StrongNewtonResult


def _normal_trace(field, grid, component, normal_order, side):
    node=0 if side=="left" else -1
    if normal_order==0:
        return field[component,node,:]
    derivative=direct_coefficient_derivative(field[component],
                                             normal_order,0,grid)
    sign = -1. if side=="left" and normal_order%2 else 1.
    return sign*derivative[node,:]


def _verify_fixed_initial_supports(engine, q, v, *, absolute_tolerance, acceleration=None):
    """Check prescribed q traces and time-independent support velocities."""
    for side in ("left","right"):
        for comp,conditions in enumerate(engine.mechanical_ends[side]):
            for order,item in enumerate(conditions):
                kind,value=_condition(item,engine.model.grid.ns,
                                      "initial "+side+" support trace")
                if kind!="ESSENTIAL":
                    continue
                qtrace=_normal_trace(q,engine.model.grid,comp,order,side)
                vtrace=_normal_trace(v,engine.model.grid,comp,order,side)
                if np.max(np.abs(qtrace-value)) > absolute_tolerance:
                    raise ValueError("INITIAL_DISPLACEMENT_ESSENTIAL_TRACE_MISMATCH")
                if np.max(np.abs(vtrace)) > absolute_tolerance:
                    raise ValueError("INITIAL_MOVING_SUPPORT_NOT_SUPPORTED")
                if acceleration is not None:
                    atrace=_normal_trace(acceleration,engine.model.grid,
                                         comp,order,side)
                    if np.max(np.abs(atrace)) > absolute_tolerance:
                        raise ValueError("INITIAL_ACCELERATION_ESSENTIAL_TRACE_MISMATCH")


def _acceleration_ends(ends):
    transformed=deepcopy(ends)
    for side in ("left","right"):
        for items in transformed[side]:
            for entry in items:
                if entry["kind"]=="ESSENTIAL":
                    entry["value"]=0.
    return transformed


def initialize_consistent_native_mkr(
        engine: NativeStrongNewmark, mechanical, velocity,
        electrical_guess=None, *, time_s=0., acceleration_guess=None,
        mechanical_state_scale, mechanical_equation_scale,
        electrical_state_scale=None, electrical_equation_scale=None,
        mechanical_load=None, electrical_charge=None,
        support_absolute_tolerance=1.e-9,
        settings=StrongNewtonSettings()):
    """Find a0 and phi0 at (q0,v0) using native MKR equations.

    The nonlinear unknown is acceleration, NOT displacement. Essential
    acceleration boundary traces are zero for stationary supports. Natural
    mechanical end actions are evaluated at the actual prescribed q0,v0
    and trial a0. The potential uses its original, simultaneous electric
    bulk/end equations and optional ONE global gauge.

    Higher natural displacement traces are compatibility conditions on
    q0/v0/phi0, not acceleration boundary equations. Keep mass balance
    at those adjacent nodes and check their original physical actions
    after closing phi0 and a0. Otherwise a local mass law acquires zero
    Jacobian rows from static slope/curvature actions.

    Mechanical load and electrical charge are generalized surface
    densities at t0. Return state=None unless the coupled solve converges.
    """
    if not isinstance(engine,NativeStrongNewmark):
        raise TypeError("NativeStrongNewmark configuration required")
    q=_field(mechanical,engine.model.shape,"initial displacement").copy()
    v=_field(velocity,engine.model.shape,"initial velocity").copy()
    tol=_finite_number(support_absolute_tolerance,"support tolerance",positive=True)
    t=_finite_number(time_s,"initial time")
    _verify_fixed_initial_supports(engine,q,v,absolute_tolerance=tol)

    if engine.model.electrical_on:
        p0=engine._electric_field(electrical_guess,"initial electric guess")
    else:
        if electrical_guess is not None:
            raise ValueError("potential guess given when electricity OFF")
        p0=None

    a0=(np.zeros_like(q) if acceleration_guess is None else
        _field(acceleration_guess,engine.model.shape,"initial acceleration guess").copy())
    load_m=(engine.mechanical_load if mechanical_load is None else
            _field(mechanical_load,engine.model.shape,"initial mechanical load").copy())
    load_e=(engine.electrical_charge if electrical_charge is None else
            engine._electric_field(electrical_charge,"initial electric charge"))
    engine._check_insulated_charge(load_e)
    acceleration_ends=_acceleration_ends(engine.mechanical_ends)

    def equations(a,p):
        balance=engine.model.balance(
            q,p,mechanical_load=load_m,electrical_charge=load_e,
            velocity=v,acceleration=a,density_kg_m3=engine.density_kg_m3,
            eta_s=engine.eta_s,
            mass_damping_rate_per_s=engine.mass_damping_rate_per_s)
        # For the initial MASS equations, essential rows must constrain
        # ACCELERATION, not reuse displacement q and create zero Jacobian rows.
        rows=assemble_physical_end_rows(
            balance,a,engine.model.grid,
            axial_orders=engine.axial_orders,mechanical_ends=acceleration_ends,
            electrical=p,electrical_ends=engine.electrical_ends,
            gauge_closed_elsewhere=engine.electric_gauge is not None)
        # Only prescribed kinematic traces have prescribed time
        # derivatives. Natural higher actions constrain the given q0/v0
        # and phi0; they cannot replace equations for unknown a0.
        for side in ("left","right"):
            for c,conditions in enumerate(engine.mechanical_ends[side]):
                for order,item in enumerate(conditions):
                    kind,_=_condition(item,engine.model.grid.ns,"initial natural trace")
                    if kind=="NATURAL" and order>0:
                        node=order if side=="left" else engine.model.grid.nx-1-order
                        rows.mechanical[c,node]=balance.mechanical_bulk[c,node]
        return rows.mechanical,rows.electrical

    nonlinear=solve_strong_nonlinear_equations(
        equations,a0,p0,
        mechanical_state_scale=mechanical_state_scale,
        mechanical_equation_scale=mechanical_equation_scale,
        electrical_state_scale=electrical_state_scale,
        electrical_equation_scale=electrical_equation_scale,
        electric_gauge=engine.electric_gauge,settings=settings)
    if not nonlinear.converged:
        return NativeInitialStateResult(None,nonlinear)
    # Retaining mass equations must never silently discard the original
    # end conditions. Reject incompatible initial displacement/velocity
    # and electric fields without publishing an accepted initial state.
    balance=engine.model.balance(
        q,nonlinear.electrical,mechanical_load=load_m,electrical_charge=load_e,
        velocity=v,acceleration=nonlinear.mechanical,density_kg_m3=engine.density_kg_m3,
        eta_s=engine.eta_s,mass_damping_rate_per_s=engine.mass_damping_rate_per_s)
    physical=assemble_physical_end_rows(
        balance,q,engine.model.grid,axial_orders=engine.axial_orders,
        mechanical_ends=engine.mechanical_ends,electrical=nonlinear.electrical,
        electrical_ends=engine.electrical_ends,
        gauge_closed_elsewhere=engine.electric_gauge is not None)
    scale=np.broadcast_to(np.asarray(mechanical_equation_scale,float),q.shape)
    defect=0.
    for side in ("left","right"):
        for c,conditions in enumerate(engine.mechanical_ends[side]):
            for order,item in enumerate(conditions):
                kind,_=_condition(item,engine.model.grid.ns,"initial natural trace")
                if kind=="NATURAL":
                    node=order if side=="left" else engine.model.grid.nx-1-order
                    defect=max(defect,float(np.max(np.abs(
                        physical.mechanical[c,node]/scale[c,node]))))
    if defect>settings.scaled_absolute_tolerance:
        return NativeInitialStateResult(None,replace(
            nonlinear,converged=False,status="INITIAL_NATURAL_TRACE_INCOMPATIBLE",
            scaled_equation_norm=max(defect,nonlinear.scaled_equation_norm)))
    _verify_fixed_initial_supports(engine,q,v,absolute_tolerance=tol,
                                   acceleration=nonlinear.mechanical)
    state=NativeNewmarkState(t,q,v,nonlinear.mechanical,nonlinear.electrical)
    return NativeInitialStateResult(state,nonlinear)


__all__=("NativeInitialStateResult","initialize_consistent_native_mkr")
