#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only diagnostics for the frozen-history plane-stress J2 tangent.

This module deliberately does not change the constitutive update or the MVI
solver.  It compares the tangent currently returned by
plane_stress_algorithmic_tangent() with central directional differences of the
actual condensed mapping

    (eps_xx, eps_yy, engineering_gamma_xy) -> (sig_xx, sig_yy, tau_xy)

while every perturbed evaluation uses the same committed material state.

Run manually:
    python J2_PLANE_STRESS_TANGENT_SELF_CHECK.py

The step sweep is intentional: a single very small FD step cannot distinguish
an incorrect tangent from noise in the nested sigma_zz=0 iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from final_j2_incremental import (
    J2Hardening,
    J2Parameters,
    J2PlaneStressResponse,
    J2State,
    plane_stress_algorithmic_tangent,
    update_plane_stress,
)


DEFAULT_STEPS = (1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8)


@dataclass(frozen=True)
class DirectionalRow:
    step: float
    relative_error: float
    absolute_error: float
    masks_equal: bool
    max_sigma_zz: float


def _stress_vector(response: J2PlaneStressResponse) -> np.ndarray:
    return np.stack(
        (
            response.stress[..., 0, 0],
            response.stress[..., 1, 1],
            response.stress[..., 0, 1],
        ),
        axis=-1,
    )


def _same_branch(
    base: J2PlaneStressResponse,
    plus: J2PlaneStressResponse,
    minus: J2PlaneStressResponse,
) -> bool:
    """Require elementwise equality of every constitutive branch mask."""

    names = ("plastic_mask", "unloading_mask", "reloading_mask")
    return all(
        np.array_equal(getattr(base, name), getattr(plus, name))
        and np.array_equal(getattr(base, name), getattr(minus, name))
        for name in names
    )


def diagnose_plane_stress_point(
    label: str,
    strain_xx: np.ndarray,
    strain_yy: np.ndarray,
    engineering_gamma_xy: np.ndarray,
    committed: J2State,
    parameters: J2Parameters,
    *,
    direction: Iterable[float] = (0.37, -0.21, 0.51),
    steps: Iterable[float] = DEFAULT_STEPS,
) -> tuple[J2PlaneStressResponse, np.ndarray, tuple[DirectionalRow, ...]]:
    """Print a local consistency report and return all numerical results."""

    ex = np.asarray(strain_xx, float)
    ey = np.asarray(strain_yy, float)
    gam = np.asarray(engineering_gamma_xy, float)
    d = np.asarray(tuple(direction), float)
    if d.shape != (3,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0.0:
        raise ValueError("direction must contain three finite nonzero components")
    d /= np.linalg.norm(d)

    base = update_plane_stress(ex, ey, gam, committed, parameters)
    tangent = plane_stress_algorithmic_tangent(
        ex, ey, gam, committed, parameters, response=base
    )
    predicted = np.einsum("...ij,j->...i", tangent, d, optimize=True)

    sigma_zz = np.asarray(base.stress[..., 2, 2], float)
    stress_scale = np.maximum(
        float(parameters.E), np.maximum(np.asarray(base.yield_stress, float), 1.0)
    )
    # This reproduces the current fallback acceptance bound in
    # update_plane_stress(); it is a diagnostic, not a new tolerance.
    current_sigma_zz_limit = (
        128.0 * float(parameters.local_tolerance) * stress_scale
    )

    print(f"\n=== {label} ===")
    print(f"plastic points       : {int(np.count_nonzero(base.plastic_mask))}")
    print(f"unloading points     : {int(np.count_nonzero(base.unloading_mask))}")
    print(f"reloading points     : {int(np.count_nonzero(base.reloading_mask))}")
    print(f"max |sigma_zz|       : {float(np.max(np.abs(sigma_zz))):.6e}")
    print(
        "current zz limit     : "
        f"{float(np.max(current_sigma_zz_limit)):.6e}"
    )
    print(
        "max |sigma_zz|/scale : "
        f"{float(np.max(np.abs(sigma_zz) / stress_scale)):.6e}"
    )
    print(
        "tangent asymmetry    : "
        f"{float(np.max(np.abs(tangent - np.swapaxes(tangent, -1, -2)))):.6e}"
    )
    print("step          rel_error      abs_error      same_masks  max|sigma_zz|")

    rows = []
    for raw_step in steps:
        h = float(raw_step)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("all diagnostic steps must be finite and positive")
        plus = update_plane_stress(
            ex + h * d[0],
            ey + h * d[1],
            gam + h * d[2],
            committed,
            parameters,
        )
        minus = update_plane_stress(
            ex - h * d[0],
            ey - h * d[1],
            gam - h * d[2],
            committed,
            parameters,
        )
        actual = (_stress_vector(plus) - _stress_vector(minus)) / (2.0 * h)
        absolute_error = float(np.max(np.abs(predicted - actual)))
        derivative_scale = max(float(np.max(np.abs(actual))), 1.0)
        relative_error = absolute_error / derivative_scale
        masks_equal = _same_branch(base, plus, minus)
        max_zz = float(
            max(
                np.max(np.abs(plus.stress[..., 2, 2])),
                np.max(np.abs(minus.stress[..., 2, 2])),
            )
        )
        row = DirectionalRow(
            h, relative_error, absolute_error, masks_equal, max_zz
        )
        rows.append(row)
        print(
            f"{h:11.3e}  {relative_error:12.5e}  "
            f"{absolute_error:12.5e}  {str(masks_equal):>10s}  {max_zz:13.6e}"
        )

    stable = [row.relative_error for row in rows if row.masks_equal]
    if stable:
        best = min(stable)
        if best < 1.0e-4:
            verdict = "tangent agrees with a central-FD window"
        elif best > 1.0e-2:
            verdict = "structural tangent mismatch"
        else:
            verdict = "borderline; inspect the step trend and sigma_zz noise"
        print(f"best stable rel_error : {best:.6e} -> {verdict}")
    else:
        print("best stable rel_error : unavailable; every step changed a branch mask")

    return base, tangent, tuple(rows)


def check_engineering_shear(parameters: J2Parameters) -> float:
    """Elastic identity: tau_xy = G * engineering_gamma_xy."""

    elastic_parameters = J2Parameters(
        E=parameters.E,
        nu=parameters.nu,
        hardening=J2Hardening(
            sigma_y0=max(parameters.hardening.sigma_y0, 1.0e12),
            kind="PERFECT",
        ),
        yield_tolerance=parameters.yield_tolerance,
        local_tolerance=parameters.local_tolerance,
        max_local_iterations=parameters.max_local_iterations,
    )
    gamma = np.asarray([1.0e-6])
    response = update_plane_stress(
        np.asarray([0.0]),
        np.asarray([0.0]),
        gamma,
        J2State.zeros((1,)),
        elastic_parameters,
    )
    actual = float(response.stress[0, 0, 1])
    expected = float(parameters.G) * float(gamma[0])
    relative_error = abs(actual - expected) / max(abs(expected), np.finfo(float).tiny)
    print("\n=== engineering shear convention ===")
    print(f"tau_xy actual        : {actual:.12e}")
    print(f"G * gamma_xy         : {expected:.12e}")
    print(f"relative error       : {relative_error:.6e}")
    return relative_error


def directional_residual_check(
    residual: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    tangent: np.ndarray,
    direction: np.ndarray,
    *,
    steps: Iterable[float] = DEFAULT_STEPS,
) -> tuple[DirectionalRow, ...]:
    """Reusable central-FD check for the actual global projected residual.

    The caller must close residual() over one frozen committed material state,
    one load level, and one unchanged set/order of projected variations.
    """

    x = np.asarray(point, float)
    matrix = np.asarray(tangent, float)
    d = np.asarray(direction, float)
    if x.ndim != 1 or d.shape != x.shape or matrix.shape != (x.size, x.size):
        raise ValueError("point, direction and tangent dimensions are inconsistent")
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("global diagnostic direction must be finite and nonzero")
    d = d / norm
    predicted = matrix @ d

    rows = []
    for raw_step in steps:
        h = float(raw_step)
        actual = (
            np.asarray(residual(x + h * d), float)
            - np.asarray(residual(x - h * d), float)
        ) / (2.0 * h)
        absolute_error = float(np.max(np.abs(predicted - actual)))
        scale = max(float(np.max(np.abs(actual))), 1.0)
        rows.append(
            DirectionalRow(
                step=h,
                relative_error=absolute_error / scale,
                absolute_error=absolute_error,
                masks_equal=True,
                max_sigma_zz=float("nan"),
            )
        )
    return tuple(rows)


def main() -> None:
    parameters = J2Parameters(
        E=210000.0,
        nu=0.3,
        hardening=J2Hardening(sigma_y0=250.0, kind="LINEAR", H=1000.0),
        yield_tolerance=1.0e-10,
        local_tolerance=1.0e-12,
        max_local_iterations=64,
    )

    shear_error = check_engineering_shear(parameters)

    zero = J2State.zeros((1,))
    fresh, _, _ = diagnose_plane_stress_point(
        "deep plastic trial from virgin state",
        np.asarray([4.0e-3]),
        np.asarray([-1.0e-3]),
        np.asarray([2.0e-3]),
        zero,
        parameters,
    )

    # A second check with genuine committed plastic history.  The diagnostic
    # trial still uses the same frozen committed state for base, +h and -h.
    diagnose_plane_stress_point(
        "plastic trial from committed plastic history",
        np.asarray([4.3e-3]),
        np.asarray([-0.9e-3]),
        np.asarray([2.1e-3]),
        fresh.state.copy(),
        parameters,
    )

    if shear_error > 1.0e-10:
        raise SystemExit("FAILED: engineering shear convention is inconsistent")
    print("\nDiagnostic completed. No constitutive or solver state was modified.")


if __name__ == "__main__":
    main()
