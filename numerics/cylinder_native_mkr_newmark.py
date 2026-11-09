"""Opt-in native MKR Newmark step for the seven cylindrical-shell theories.

The spatial balance is DirectStrongShell's own strong-equation route, never
CylinderFunctional, variational FDM, MVI, or a shared spatial projection.
Mechanical inertia/Kelvin-Voigt damping are evaluated at t[n+1]. The
electrical variables are quasi-static and solved in the SAME Newton system.
Construction and importing this module do not advance a solution.

RESEARCH-ONLY: coupled face fluxes, physical differential orders and higher
end closures are not yet certified. No GUI/CASE dispatch is changed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from cylinder_direct_strong_boundary import assemble_physical_end_rows
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import (
    StrongElectricGauge, StrongNewtonResult, StrongNewtonSettings,
    solve_strong_nonlinear_equations,
)


def _finite_number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ": a finite numeric value is required")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ": a finite numeric value is required") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(name + ": invalid physical parameter")
    return number


def _field(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + ": finite field with shape " + str(shape) + " required")
    return a


def _frozen_field(value, shape, name):
    out = _field(value, shape, name).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class NativeNewmarkState:
    """One complete mechanical state and instantaneous electric potential."""
    time_s: float
    mechanical: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    electrical: np.ndarray | None = None

    def __post_init__(self):
        t = _finite_number(self.time_s, "time")
        q = np.asarray(self.mechanical, dtype=float)
        if q.ndim != 3 or min(q.shape) < 1 or not np.isfinite(q).all():
            raise ValueError("finite 3D mechanical field required")
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "mechanical", _frozen_field(q, q.shape, "mechanical"))
        object.__setattr__(self, "velocity", _frozen_field(self.velocity, q.shape, "velocity"))
        object.__setattr__(self, "acceleration", _frozen_field(self.acceleration, q.shape, "acceleration"))
        if self.electrical is not None:
            p = np.asarray(self.electrical, dtype=float)
            if p.ndim != 3 or p.shape[1:] != q.shape[1:] or p.shape[0] < 1:
                raise ValueError("potential field must match the mechanical grid")
            object.__setattr__(self, "electrical", _frozen_field(p, p.shape, "electrical"))


@dataclass(frozen=True)
class NativeNewmarkStep:
    """An unsuccessful Newton solve MUST NOT produce an accepted time state."""
    state: NativeNewmarkState | None
    nonlinear: StrongNewtonResult
    attempted_time_s: float


class NativeStrongNewmark:
    """Own Newmark kinematics around one native strong MKR shell.

    Axial differential half-orders and end programs are external physical
    inputs, not deduced from kinematic jet counts. The caller must supply a
    consistent initial acceleration and every essential/natural end trace.
    Default beta=1/4, gamma=1/2 is the average-acceleration method.
    """

    def __init__(self, model: DirectStrongShell, *, axial_orders,
                 mechanical_ends, density_kg_m3, electrical_ends=None,
                 electric_gauge: StrongElectricGauge | None = None,
                 mechanical_load=None, electrical_charge=None, eta_s=0.,
                 mass_damping_rate_per_s=0., beta=0.25, gamma=0.5):
        if not isinstance(model, DirectStrongShell):
            raise TypeError("NativeStrongNewmark requires DirectStrongShell")
        orders = tuple(axial_orders)
        if (len(orders) != model.n_mechanical or
                any(type(x) is not int or x not in (1, 2, 3) for x in orders)):
            raise ValueError("explicit per-component axial half-orders 1..3 required")
        if not isinstance(mechanical_ends, dict) or set(mechanical_ends) != {"left", "right"}:
            raise ValueError("explicit physical left/right mechanical ends required")
        if model.electrical_on and (
                not isinstance(electrical_ends, dict) or
                set(electrical_ends) != {"left", "right"}):
            raise ValueError("active electricity requires physical electric ends")
        if not model.electrical_on and (electrical_ends is not None or electric_gauge is not None or
                                        electrical_charge is not None):
            raise ValueError("electricity is OFF; electric ends, gauge and charge are forbidden")
        if electric_gauge is not None and not isinstance(electric_gauge, StrongElectricGauge):
            raise TypeError("StrongElectricGauge required")
        self.model = model
        self.axial_orders = orders
        self.mechanical_ends = deepcopy(mechanical_ends)
        self.electrical_ends = deepcopy(electrical_ends)
        self.electric_gauge = electric_gauge
        self.density_kg_m3 = _finite_number(density_kg_m3, "density", positive=True)
        self.eta_s = _finite_number(eta_s, "Kelvin-Voigt time")
        self.mass_damping_rate_per_s = _finite_number(
            mass_damping_rate_per_s, "mass damping")
        self.beta = _finite_number(beta, "Newmark beta", positive=True)
        self.gamma = _finite_number(gamma, "Newmark gamma", positive=True)
        # Standard unconditional-stability region for linear elastodynamics.
        # This is a parameter preflight, NOT a stability proof for nonlinear
        # gradient-flexoelectric time integration.
        if not (0.5 <= self.gamma <= 1.0 and
                self.beta >= 0.25 * (self.gamma + 0.5)**2):
            raise ValueError("NEWMARK_UNSTABLE_PARAMETER_PAIR")
        self.mechanical_load = (None if mechanical_load is None else
            _frozen_field(mechanical_load, model.shape, "mechanical load"))
        self.electrical_charge = (None if electrical_charge is None else
            self._electric_field(electrical_charge, "electric charge"))
        self._check_insulated_charge(self.electrical_charge)

    @classmethod
    def from_cuf_parallel_problem(cls, problem, *, density_kg_m3,
                                  eta_s=0., mass_damping_rate_per_s=0.,
                                  beta=0.25, gamma=0.5):
        """Keep the Krysko-Zhong CUF parallel-end program and its one gauge."""
        from cylinder_cuf_parallel_ends import CUFParallelEndProblem
        if not isinstance(problem, CUFParallelEndProblem):
            raise TypeError("CUFParallelEndProblem required")
        return cls(problem.model, axial_orders=problem.ends.axial_orders,
                   mechanical_ends=problem.ends.mechanical_ends,
                   electrical_ends=problem.ends.electrical_ends,
                   electric_gauge=problem.electric_gauge,
                   mechanical_load=problem.mechanical_load,
                   electrical_charge=problem.electrical_charge,
                   density_kg_m3=density_kg_m3, eta_s=eta_s,
                   mass_damping_rate_per_s=mass_damping_rate_per_s,
                   beta=beta, gamma=gamma)

    def _electric_field(self, value, name):
        a = np.asarray(value, dtype=float)
        if not self.model.electrical_on:
            raise ValueError("electricity is OFF")
        if a.ndim != 3 or a.shape[1:] != self.model.shape[1:] or not 1 <= a.shape[0] <= 5:
            raise ValueError(name + ": potential modes 0..4 and model grid required")
        if self.electric_gauge is not None and a.shape != self.electric_gauge.weights.shape:
            raise ValueError(name + ": field must match electric gauge modes")
        return _frozen_field(a, a.shape, name)

    def _check_insulated_charge(self, charge):
        if charge is None or self.electric_gauge is None:
            return
        try:
            insulated = all(self.electrical_ends[side][0]["kind"] == "NATURAL"
                            for side in ("left", "right"))
        except (KeyError, IndexError, TypeError):
            return  # The boundary assembler will reject malformed end data.
        if not insulated:
            return
        weighted = self.model.area_weights * charge[0]
        net = float(np.sum(weighted))
        total = float(np.sum(np.abs(weighted)))
        if abs(net) > 1.e-10 * max(1.e-30, total):
            raise ValueError("incompatible fully insulated constant-mode free charge")

    def advance(self, state: NativeNewmarkState, dt_s, *,
                mechanical_state_scale, mechanical_equation_scale,
                electrical_state_scale=None, electrical_equation_scale=None,
                mechanical_load=None, electrical_charge=None,
                mechanical_guess=None, electrical_guess=None,
                settings=StrongNewtonSettings()):
        """Attempt exactly one step; return no new state on Newton failure.

        Optional generalized loads are values AT t[n+1], not raw pressure in
        pascals. No time series is silently generated or advanced on failure.
        """
        if not isinstance(state, NativeNewmarkState):
            raise TypeError("NativeNewmarkState required")
        q0 = _field(state.mechanical, self.model.shape, "initial displacement")
        v0 = _field(state.velocity, self.model.shape, "initial velocity")
        a0 = _field(state.acceleration, self.model.shape, "initial acceleration")
        if self.model.electrical_on:
            p0 = self._electric_field(state.electrical, "initial potential")
        elif state.electrical is not None or electrical_guess is not None:
            raise ValueError("electric state supplied when OFF")
        else:
            p0 = None
        dt = _finite_number(dt_s, "time step", positive=True)
        time_next = state.time_s + dt
        if not math.isfinite(time_next) or time_next <= state.time_s:
            raise ValueError("time step does not advance representable time")
        load_m = (self.mechanical_load if mechanical_load is None else
                  _frozen_field(mechanical_load, self.model.shape, "step mechanical load"))
        load_e = (self.electrical_charge if electrical_charge is None else
                  self._electric_field(electrical_charge, "step electric charge"))
        self._check_insulated_charge(load_e)
        predictor = q0 + dt*v0 + (0.5-self.beta)*dt**2*a0
        velocity_predictor = v0 + (1.-self.gamma)*dt*a0
        if not np.isfinite(predictor).all() or not np.isfinite(velocity_predictor).all():
            raise ValueError("nonfinite Newmark predictor")
        qguess = predictor if mechanical_guess is None else _field(
            mechanical_guess, self.model.shape, "step mechanical guess")
        if self.model.electrical_on:
            pguess = p0 if electrical_guess is None else self._electric_field(
                electrical_guess, "step electric guess")
        else:
            pguess = None

        def kinematics(q):
            a = (q-predictor)/(self.beta*dt**2)
            v = velocity_predictor + self.gamma*dt*a
            if not np.isfinite(a).all() or not np.isfinite(v).all():
                raise ValueError("nonfinite Newmark velocity/acceleration")
            return v, a

        def equations(q, p):
            v, a = kinematics(q)
            balance = self.model.balance(
                q, p, mechanical_load=load_m, electrical_charge=load_e,
                velocity=v, acceleration=a, density_kg_m3=self.density_kg_m3,
                eta_s=self.eta_s,
                mass_damping_rate_per_s=self.mass_damping_rate_per_s)
            rows = assemble_physical_end_rows(
                balance, q, self.model.grid, axial_orders=self.axial_orders,
                mechanical_ends=self.mechanical_ends, electrical=p,
                electrical_ends=self.electrical_ends,
                gauge_closed_elsewhere=self.electric_gauge is not None)
            return rows.mechanical, rows.electrical

        nonlinear = solve_strong_nonlinear_equations(
            equations, qguess, pguess,
            mechanical_state_scale=mechanical_state_scale,
            mechanical_equation_scale=mechanical_equation_scale,
            electrical_state_scale=electrical_state_scale,
            electrical_equation_scale=electrical_equation_scale,
            electric_gauge=self.electric_gauge, settings=settings)
        if not nonlinear.converged:
            return NativeNewmarkStep(None, nonlinear, time_next)
        velocity, acceleration = kinematics(nonlinear.mechanical)
        accepted = NativeNewmarkState(time_next, nonlinear.mechanical,
                                      velocity, acceleration, nonlinear.electrical)
        return NativeNewmarkStep(accepted, nonlinear, time_next)


__all__ = ("NativeNewmarkState", "NativeNewmarkStep", "NativeStrongNewmark")
