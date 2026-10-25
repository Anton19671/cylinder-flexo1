#!/usr/bin/env python3
"""Mechanically admissible Newmark steps, including moving essential traces.

A trace Cq=d(t) is enforced in the spatial variational solve. This module
checks its *discrete Newmark* velocity and acceleration, never substituting
continuous derivatives of a prescribed time law for those state variables.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _vector(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite shape {shape}, got {a.shape}")
    return a.copy()


def mechanical_trace_matrix(rows, count):
    """Return row-normalized C_mech and d, excluding electrical constraints."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("mechanical trace requires positive integer state size")
    rows = [r for r in rows if r.field == "mechanical"]
    c = np.zeros((len(rows), int(count)), dtype=float)
    d = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        if len(row.indices) != len(row.weights):
            raise ValueError("mechanical trace index/weight count mismatch")
        for idx, weight in zip(row.indices, row.weights):
            if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)) or not 0 <= idx < count:
                raise ValueError("mechanical trace index out of range")
            c[i, idx] += float(weight)
        d[i] = float(row.prescribed)
    if not (np.isfinite(c).all() and np.isfinite(d).all()):
        raise ValueError("nonfinite mechanical trace")
    norm = np.linalg.norm(c, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("empty mechanical BC trace")
    # Empty constraints must not involve a division by zero or a zero-sized max.
    return c / norm[:, None], d / norm


@dataclass(frozen=True)
class TraceStep:
    time_s: float
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def effective_step_acceptance(converged, relative_residual, tolerance):
    """Use the actual full admissible Newmark residual, not an XY projection.

    Return the boolean decision and a stable status for persisted output.
    """
    if not isinstance(converged, (bool, np.bool_)):
        raise TypeError("converged must be boolean")
    residual = float(relative_residual)
    limit = float(tolerance)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("relative Newmark tolerance must be finite positive")
    if not np.isfinite(residual) or residual < 0:
        return False, "INVALID_FULL_EFFECTIVE_RESIDUAL"
    if not converged:
        return False, "SPATIAL_SOLVE_NOT_CONVERGED"
    if residual > limit:
        return False, "FULL_EFFECTIVE_RESIDUAL_ABOVE_TOLERANCE"
    return True, "ACCEPTED"


class MechanicalTraceNewmark:
    """Acceptance gate for a fixed mechanical trace matrix C.

    The target sequence d(t) can move. At every step the constrained state
    must satisfy all three: Cq=d, Cv=d_vel_Newmark, Ca=d_acc_Newmark.
    A failed attempt never advances the monitor's accepted state.
    """
    def __init__(self, c, d, q, v, a, *, t, dt, beta, gamma,
                 atol=1.e-10, rtol=1.e-8):
        self.c = np.asarray(c, dtype=float).copy()
        if self.c.ndim != 2 or self.c.shape[1] < 1 or not np.isfinite(self.c).all():
            raise ValueError("invalid initial mechanical trace matrix")
        self.d = _vector(d, (self.c.shape[0],), "initial trace target")
        self.t, self.dt, self.beta, self.gamma = map(float, (t, dt, beta, gamma))
        self.atol, self.rtol = float(atol), float(rtol)
        if (not np.isfinite([self.t, self.dt, self.beta, self.gamma,
                             self.atol, self.rtol]).all() or
            self.dt <= 0 or self.beta <= 0 or self.gamma < 0 or
            self.atol <= 0 or self.rtol < 0 or self.t + self.dt <= self.t):
            raise ValueError("invalid Newmark mechanical trace settings")
        self.v = self.c @ _vector(v, (self.c.shape[1],), "initial velocity")
        self.a = self.c @ _vector(a, (self.c.shape[1],), "initial acceleration")
        self._verify("initial displacement",
                     self.c @ _vector(q, (self.c.shape[1],), "initial displacement"), self.d)

    def _verify(self, name, actual, want):
        err = np.abs(actual - want)
        tol = self.atol + self.rtol * np.maximum(np.abs(actual), np.abs(want))
        if np.any(~np.isfinite(err)) or np.any(err > tol):
            raise ValueError(f"Newmark mechanical BC {name} drift: "
                             f"{float(np.max(err, initial=0)):.3e}")
        return float(np.max(err, initial=0))

    def _verify_next_time(self, new_time):
        tt = float(new_time)
        expected = self.t + self.dt
        # Compare to a few representable ULPs, not to the *absolute* current
        # time via np.isclose(default rtol). Large t must not mask stale steps.
        gap = abs(tt - expected)
        time_tol = 16 * np.finfo(float).eps * max(abs(tt), abs(expected), 1.)
        if not np.isfinite(tt) or gap > time_tol:
            raise ValueError("moving BC time step changed or stale preview")
        return tt

    def preview(self, c, d, t):
        c = _vector(c, self.c.shape, "moving trace matrix")
        target = _vector(d, self.d.shape, "moving trace target")
        if np.max(np.abs(c-self.c), initial=0) > 1.e-12:
            raise ValueError("moving BC constraint matrix changed")
        tt = self._verify_next_time(t)
        predicted = self.d + self.dt*self.v + self.dt**2*(.5-self.beta)*self.a
        acceleration = (target-predicted)/(self.beta*self.dt**2)
        velocity = self.v + self.dt*((1-self.gamma)*self.a+self.gamma*acceleration)
        return TraceStep(tt, target, velocity, acceleration)

    def accept(self, expected, q, v, a):
        if not isinstance(expected, TraceStep):
            raise TypeError("expected TraceStep from preview")
        self._verify_next_time(expected.time_s)
        d = _vector(expected.displacement, self.d.shape, "preview displacement")
        vel = _vector(expected.velocity, self.v.shape, "preview velocity")
        acc = _vector(expected.acceleration, self.a.shape, "preview acceleration")
        delta = {
            "displacement": self._verify("displacement", self.c @ _vector(
                q, (self.c.shape[1],), "displacement"), d),
            "velocity": self._verify("velocity", self.c @ _vector(
                v, (self.c.shape[1],), "velocity"), vel),
            "acceleration": self._verify("acceleration", self.c @ _vector(
                a, (self.c.shape[1],), "acceleration"), acc),
        }
        # Commit only when all checks above have succeeded.
        self.t, self.d, self.v, self.a = expected.time_s, d, vel, acc
        return delta


__all__ = ("TraceStep", "MechanicalTraceNewmark", "mechanical_trace_matrix",
           "effective_step_acceptance")
