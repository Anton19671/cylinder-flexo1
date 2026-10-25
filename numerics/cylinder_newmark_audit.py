#!/usr/bin/env python3
"""Independent discrete Newmark kinematics audit of an accepted mechanical state.

For a step q_(n+1), v_(n+1), a_(n+1), check the *actual stored*
fields against predictors made from the previous accepted state. No field is
silently reconstructed or corrected: an inconsistent step is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class NewmarkKinematicDefects:
    displacement_abs_max: float
    velocity_abs_max: float
    displacement_tolerance: float
    velocity_tolerance: float

    def as_dict(self):
        return {
            "displacement_abs_max": self.displacement_abs_max,
            "velocity_abs_max": self.velocity_abs_max,
            "displacement_tolerance": self.displacement_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
        }


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite mechanical field shape {shape} required")
    return a


def audit_newmark_step(q0, v0, a0, q1, v1, a1, *, dt_s, beta, gamma,
                       relative_tolerance=2.e-10, absolute_tolerance=2.e-14):
    """Verify q1=q_pred+beta*dt^2*a1 and v1=v_pred+gamma*dt*a1.

    Error bounds use the scales of the *actual predictor and state*, rather
    than dividing by tiny motion amplitudes. The absolute floor protects
    against ordinary floating point rounding in near-zero rigid fields.
    This is an independent kinematic gate, not an equilibrium test.
    """
    q0 = np.asarray(q0, dtype=float)
    if q0.ndim < 1 or q0.size == 0 or not np.isfinite(q0).all():
        raise ValueError("q0: finite nonempty mechanical field required")
    v0 = _array(v0, q0.shape, "v0")
    a0 = _array(a0, q0.shape, "a0")
    q1 = _array(q1, q0.shape, "q1")
    v1 = _array(v1, q0.shape, "v1")
    a1 = _array(a1, q0.shape, "a1")
    values = (dt_s, beta, gamma, relative_tolerance, absolute_tolerance)
    if any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("Newmark audit parameters must be numbers")
    dt_s, beta, gamma, relative_tolerance, absolute_tolerance = map(float, values)
    if (not all(math.isfinite(v) for v in (dt_s,beta,gamma,relative_tolerance,absolute_tolerance))
        or dt_s <= 0 or beta <= 0 or gamma < 0
        or relative_tolerance < 0 or absolute_tolerance < 0
        or relative_tolerance == absolute_tolerance == 0):
        raise ValueError("invalid Newmark audit parameters")
    q_predictor = q0 + dt_s*v0 + dt_s**2*(.5-beta)*a0
    v_predictor = v0 + dt_s*(1-gamma)*a0
    q_expected = q_predictor + beta*dt_s**2*a1
    v_expected = v_predictor + gamma*dt_s*a1
    if not np.isfinite(q_expected).all() or not np.isfinite(v_expected).all():
        raise ValueError("Newmark predictor overflow/nonfinite")
    qerr = float(np.max(np.abs(q1-q_expected)))
    verr = float(np.max(np.abs(v1-v_expected)))
    q_scale = max(float(np.max(np.abs(q1))), float(np.max(np.abs(q_predictor))),
                  float(np.max(np.abs(beta*dt_s**2*a1))))
    v_scale = max(float(np.max(np.abs(v1))), float(np.max(np.abs(v_predictor))),
                  float(np.max(np.abs(gamma*dt_s*a1))))
    qtol = absolute_tolerance + relative_tolerance*q_scale
    vtol = absolute_tolerance + relative_tolerance*v_scale
    if qerr > qtol or verr > vtol:
        raise ValueError(
            "NEWMARK_KINEMATICS_DRIFT: "
            f"q defect {qerr:.3e} > {qtol:.3e} or "
            f"v defect {verr:.3e} > {vtol:.3e}")
    return NewmarkKinematicDefects(qerr, verr, qtol, vtol)


__all__ = ("NewmarkKinematicDefects", "audit_newmark_step")
