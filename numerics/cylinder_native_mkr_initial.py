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
    _condition, assemble_physical_end_rows, native_kinematic_normal_trace,
)
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
    # Must match essential rows in assemble_physical_end_rows and native
    # strain jets. A separate seven-point coefficient-flux stencil is NOT
    # the displacement/acceleration slope enforced by this integrator.
    return native_kinematic_normal_trace(
        field,grid,component,normal_order,side)


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
                # A failed dynamic support is a physical trace defect.
                # Report its end, field, derivative order and magnitude;
                # the runner retains this detail in RUN_SUMMARY.json.
                qdef=float(np.max(np.abs(qtrace-value)))
                if qdef>absolute_tolerance:
                    raise ValueError(
                        "INITIAL_DISPLACEMENT_ESSENTIAL_TRACE_MISMATCH: "
                        f"side={side} component={comp} normal_order={order} "
                        f"max_abs_defect={qdef:.6e} tolerance={absolute_tolerance:.6e}")
                vdef=float(np.max(np.abs(vtrace)))
                if vdef>absolute_tolerance:
                    raise ValueError(
                        "INITIAL_MOVING_SUPPORT_NOT_SUPPORTED: "
                        f"side={side} component={comp} normal_order={order} "
                        f"max_abs_defect={vdef:.6e} tolerance={absolute_tolerance:.6e}")
                if acceleration is not None:
                    atrace=_normal_trace(acceleration,engine.model.grid,
                                         comp,order,side)
                    adef=float(np.max(np.abs(atrace)))
                    if adef>absolute_tolerance:
                        raise ValueError(
                            "INITIAL_ACCELERATION_ESSENTIAL_TRACE_MISMATCH: "
                            f"side={side} component={comp} normal_order={order} "
                            f"max_abs_defect={adef:.6e} tolerance={absolute_tolerance:.6e}")


def _acceleration_ends(ends):
    transformed=deepcopy(ends)
    for side in ("left","right"):
        for items in transformed[side]:
            for entry in items:
                if entry["kind"]=="ESSENTIAL":
                    entry["value"]=0.
    return transformed


def _acceleration_dependent_natural_tractions(
        engine,q,v,a0,p0,load_m,load_e):
    """Identify zeroth-order natural rows involving UNKNOWN acceleration.

    KL and Reddy director inertia depends on acceleration *gradients*;
    its true free-end traction must therefore be solved at t0, not
    dismissed as compatibility of given q0/v0. In contrast, a
    natural action independent of acceleration cannot replace an
    acceleration mass equation: its tangent row is identically zero.

    Compare actual physical end actions for several independent finite
    acceleration probes. This is a strict detection diagnostic, NOT a
    general rank/well-posedness certificate or an excuse to weaken the
    original post-solve natural end check.
    """
    candidates=[]
    for side_index,side in enumerate(("left","right")):
        for c,conditions in enumerate(engine.mechanical_ends[side]):
            kind,_=_condition(conditions[0],engine.model.grid.ns,
                              "initial zeroth natural trace")
            if kind=="NATURAL":
                candidates.append((side_index,side,c))
    if not candidates:
        return frozenset()
    grid=engine.model.grid

    def traction(acceleration):
        balance=engine.model.balance(
            q,p0,mechanical_load=load_m,electrical_charge=load_e,
            velocity=v,acceleration=acceleration,density_kg_m3=engine.density_kg_m3,
            eta_s=engine.eta_s,mass_damping_rate_per_s=engine.mass_damping_rate_per_s)
        array=np.asarray(balance.mechanical_end_displacement,float)
        expected=(2,engine.model.n_mechanical,grid.ns)
        if array.shape!=expected or not np.isfinite(array).all():
            raise ValueError("INITIAL_NATURAL_TRACTION_SHAPE_INVALID")
        return array

    base=traction(a0)
    # Linear acceleration metric with spatially varying probes. Uniform
    # acceleration alone cannot detect the Reddy w_x inertial trace.
    axis=np.linspace(-1.,1.,grid.nx)[None,:,None]
    angle=(2.*np.pi*np.arange(grid.ns)/grid.ns)[None,None,:]
    fields=(1.+np.arange(engine.model.n_mechanical))[:,None,None]
    magnitude=max(1.,float(np.max(np.abs(a0))))
    shapes=(
        fields*(axis+.29*np.cos(angle)),
        (fields+.37)*(axis**2+.19*np.sin(angle)),
        (fields+.73)*(axis**3+.23*np.cos(2.*angle)),
    )
    responses=[]
    for shape in shapes:
        perturbation=np.broadcast_to(shape,q.shape)*magnitude
        responses.append(traction(a0+perturbation))
    active=set()
    for side_index,side,c in candidates:
        b=base[side_index,c]
        values=[v[side_index,c] for v in responses]
        signal=max(float(np.max(np.abs(v-b))) for v in values)
        scale=max(1.,float(np.max(np.abs(b))),
                  *(float(np.max(np.abs(v))) for v in values))
        if signal>128.*np.finfo(float).eps*scale:
            active.add((side,c))
    return frozenset(active)


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

    Higher natural traces and acceleration-independent zeroth actions
    are compatibility conditions on prescribed q0/v0/phi0, not
    acceleration equations: keep the mass balance at those rows, then
    check original natural actions. Acceleration-dependent zeroth natural
    actions (possible with KL/Reddy director-gradient inertia) ARE
    equations for unknown a0 and remain in the coupled solve.
    Otherwise the initializer either acquires zero Jacobian rows or
    discards real dynamic free-end conditions.

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
    dynamic_natural=_acceleration_dependent_natural_tractions(
        engine,q,v,a0,p0,load_m,load_e)

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
        # Higher natural actions have no acceleration-gradient coefficient
        # beyond q_x in the implemented KL/Reddy inertia. They are checked
        # as compatibility of the prescribed initial q/v after the solve.
        # Zeroth natural traction is different: KL/Reddy gradient inertia
        # can make it an equation for UNKNOWN acceleration. Retain such a
        # physical equation, rather than silently replacing it by mass.
        # Zeroth actions insensitive to acceleration still need mass rows,
        # or the Newton tangent acquires an exactly zero row. Every
        # physical natural action is checked again after the solve.
        for side in ("left","right"):
            for c,conditions in enumerate(engine.mechanical_ends[side]):
                for order,item in enumerate(conditions):
                    kind,_=_condition(item,engine.model.grid.ns,"initial natural trace")
                    if kind=="NATURAL" and not (
                            order==0 and (side,c) in dynamic_natural):
                        node=order if side=="left" else engine.model.grid.nx-1-order
                        rows.mechanical[c,node]=balance.mechanical_bulk[c,node]
        return rows.mechanical,rows.electrical

    # The unknown is acceleration: essential rows carry acceleration traces,
    # not force residuals. Enforce the same absolute support tolerance used
    # for post-solve acceptance, without scaling the interior mass equations.
    from cylinder_native_mkr_newmark import _time_step_mechanical_equation_scale
    initial_equation_scale=_time_step_mechanical_equation_scale(
        engine,mechanical_equation_scale,1.,settings,
        support_absolute_tolerance=tol)
    nonlinear=solve_strong_nonlinear_equations(
        equations,a0,p0,
        mechanical_state_scale=mechanical_state_scale,
        mechanical_equation_scale=initial_equation_scale,
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
