#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent incremental associative J2 material-point return mapping.

Birger is not imported or modified.  This module owns committed plastic
history at every integration point.  Adapters must evaluate each global trial
from the last accepted state and commit the returned state only after the load
step is accepted.

This is the small-strain material-frame kernel for LINEAR and von-Karman.
A full Green-Lagrange path must use the separate finite-strain adapter and
must never silently fall back to this additive small-strain law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class J2MaterialError(ValueError):
    pass


@dataclass(frozen=True)
class J2Hardening:
    """Isotropic law sigma_y(kappa): PERFECT, LINEAR, or monotone TABLE."""

    sigma_y0: float
    kind: str = "LINEAR"
    H: float = 0.0
    kappa_table: Optional[Sequence[float]] = None
    sigma_y_table: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma_y0) or self.sigma_y0 <= 0.0:
            raise J2MaterialError("J2 sigma_y0 must be finite and positive")
        kind = str(self.kind).upper()
        if kind not in ("PERFECT", "LINEAR", "TABLE"):
            raise J2MaterialError("J2 hardening.kind must be PERFECT, LINEAR or TABLE")
        object.__setattr__(self, "kind", kind)
        if not np.isfinite(self.H) or self.H < 0.0:
            raise J2MaterialError("J2 H must be finite and non-negative")
        if kind == "TABLE":
            if self.kappa_table is None or self.sigma_y_table is None:
                raise J2MaterialError("J2 TABLE requires kappa_table and sigma_y_table")
            k = np.asarray(self.kappa_table, float)
            y = np.asarray(self.sigma_y_table, float)
            if k.ndim != 1 or y.ndim != 1 or k.size < 2 or y.size != k.size:
                raise J2MaterialError("J2 TABLE arrays must be 1-D, equal-sized and have at least two entries")
            if not np.isfinite(k).all() or not np.isfinite(y).all() or k[0] != 0.0:
                raise J2MaterialError("J2 TABLE must be finite and start at kappa=0")
            if np.any(np.diff(k) <= 0.0) or np.any(np.diff(y) < 0.0) or np.any(y <= 0.0):
                raise J2MaterialError("J2 TABLE kappa must increase and sigma_y must be positive/non-decreasing")
            if not np.isclose(float(y[0]), float(self.sigma_y0), rtol=1.0e-12, atol=0.0):
                raise J2MaterialError("J2 TABLE first sigma_y must equal sigma_y0")

    def yield_stress(self, kappa: np.ndarray) -> np.ndarray:
        k = np.asarray(kappa, float)
        if not np.isfinite(k).all() or np.any(k < -1.0e-14):
            raise J2MaterialError("J2 kappa must be finite and non-negative")
        k = np.maximum(k, 0.0)
        if self.kind == "PERFECT":
            return np.full_like(k, float(self.sigma_y0), dtype=float)
        if self.kind == "LINEAR":
            return float(self.sigma_y0) + float(self.H) * k
        x = np.asarray(self.kappa_table, float)
        y = np.asarray(self.sigma_y_table, float)
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return np.interp(k, x, y) + slope * np.maximum(k - x[-1], 0.0)


@dataclass(frozen=True)
class J2Parameters:
    E: float
    nu: float
    hardening: J2Hardening
    yield_tolerance: float = 1.0e-10
    local_tolerance: float = 1.0e-12
    max_local_iterations: int = 48

    def __post_init__(self) -> None:
        if not np.isfinite(self.E) or self.E <= 0.0:
            raise J2MaterialError("J2 E must be finite and positive")
        if not np.isfinite(self.nu) or not (-1.0 < self.nu < 0.5):
            raise J2MaterialError("J2 nu must satisfy -1 < nu < 0.5")
        if self.yield_tolerance < 0.0 or self.local_tolerance <= 0.0:
            raise J2MaterialError("J2 tolerances are invalid")
        if int(self.max_local_iterations) < 4:
            raise J2MaterialError("J2 max_local_iterations must be at least 4")

    @property
    def G(self) -> float:
        return float(self.E) / (2.0 * (1.0 + float(self.nu)))

    @property
    def K(self) -> float:
        return float(self.E) / (3.0 * (1.0 - 2.0 * float(self.nu)))


@dataclass
class J2State:
    """Committed history: symmetric plastic strain, kappa, equivalent stress."""

    plastic_strain: np.ndarray
    kappa: np.ndarray
    equivalent_stress: np.ndarray

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "J2State":
        return cls(
            plastic_strain=np.zeros(tuple(shape) + (3, 3), float),
            kappa=np.zeros(tuple(shape), float),
            equivalent_stress=np.zeros(tuple(shape), float),
        )

    def copy(self) -> "J2State":
        return J2State(self.plastic_strain.copy(), self.kappa.copy(), self.equivalent_stress.copy())

    def validate(self) -> Tuple[int, ...]:
        ep = np.asarray(self.plastic_strain, float)
        k = np.asarray(self.kappa, float)
        q = np.asarray(self.equivalent_stress, float)
        if ep.ndim < 2 or ep.shape[-2:] != (3, 3):
            raise J2MaterialError("J2 plastic_strain shape must end with (3,3)")
        shape = ep.shape[:-2]
        if k.shape != shape or q.shape != shape:
            raise J2MaterialError("J2 scalar histories must match plastic_strain point shape")
        if not np.isfinite(ep).all() or not np.isfinite(k).all() or not np.isfinite(q).all():
            raise J2MaterialError("J2 state must be finite")
        if np.any(k < -1.0e-14) or np.any(q < -1.0e-14):
            raise J2MaterialError("J2 scalar history must be non-negative")
        if ep.size and np.max(np.abs(ep - np.swapaxes(ep, -1, -2))) > 1.0e-10:
            raise J2MaterialError("J2 plastic_strain must be symmetric")
        return shape


@dataclass
class J2Response:
    """Uncommitted local response; caller commits state only after step accept."""

    stress: np.ndarray
    elastic_strain: np.ndarray
    state: J2State
    equivalent_stress_trial: np.ndarray
    equivalent_stress: np.ndarray
    yield_stress: np.ndarray
    yield_function_trial: np.ndarray
    plastic_multiplier: np.ndarray
    plastic_mask: np.ndarray
    unloading_mask: np.ndarray
    reloading_mask: np.ndarray


@dataclass
class J2PlaneStressResponse(J2Response):
    """J2 response satisfying sigma_zz=0 with the solved total strain tensor."""

    total_strain: np.ndarray


_I3 = np.eye(3, dtype=float)


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def _trace(a: np.ndarray) -> np.ndarray:
    return np.trace(a, axis1=-2, axis2=-1)


def _dev(a: np.ndarray) -> np.ndarray:
    return a - _trace(a)[..., None, None] * _I3 / 3.0


def _ddot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...ij->...", a, b)


def _equivalent(s: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(1.5 * _ddot(s, s), 0.0))


def _elastic_stress(eps_e: np.ndarray, par: J2Parameters) -> np.ndarray:
    return float(par.K) * _trace(eps_e)[..., None, None] * _I3 + 2.0 * float(par.G) * _dev(eps_e)


def _plastic_increment(q_trial: np.ndarray, k_old: np.ndarray, active: np.ndarray, par: J2Parameters) -> np.ndarray:
    """Monotone local solve of qtr - 3G*dgamma - sigma_y(kappa+dgamma)=0."""

    dg = np.zeros_like(q_trial, float)
    if not np.any(active):
        return dg
    lo = np.zeros_like(q_trial, float)
    hi = np.where(active, q_trial / max(3.0 * float(par.G), np.finfo(float).tiny), 0.0)
    # Use extra safeguarded bisection iterations for difficult plastic points.
    # The local return map is cheap compared with the global MVI tangent and
    # must not fail merely because the user-facing iteration limit is modest.
    n_iter = max(64, int(par.max_local_iterations))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fmid = q_trial - 3.0 * float(par.G) * mid - par.hardening.yield_stress(k_old + mid)
        lo = np.where(active & (fmid > 0.0), mid, lo)
        hi = np.where(active & (fmid <= 0.0), mid, hi)
        if np.max(np.where(active, np.abs(hi - lo), 0.0)) <= float(par.local_tolerance):
            break
    # Select the best evaluated endpoint/midpoint.  This avoids losing a
    # valid root through cancellation in the arithmetic midpoint.
    mid = 0.5 * (lo + hi)
    candidates = np.stack((lo, mid, hi), axis=0)
    residuals = np.stack(
        tuple(
            np.abs(q_trial - 3.0 * float(par.G) * candidate
                   - par.hardening.yield_stress(k_old + candidate))
            for candidate in (lo, mid, hi)
        ),
        axis=0,
    )
    best = np.argmin(residuals, axis=0)
    dg = np.take_along_axis(candidates, best[None, ...], axis=0)[0]
    dg = np.where(active, dg, 0.0)
    gap = q_trial - 3.0 * float(par.G) * dg - par.hardening.yield_stress(k_old + dg)
    scale = np.maximum(q_trial, np.maximum(par.hardening.yield_stress(k_old), 1.0))
    # Stress residuals are formed by subtracting terms of comparable
    # magnitude.  Allow a conservative floating-point floor so an O(eps)
    # cancellation residual cannot reject an otherwise valid local return.
    roundoff_floor = 2048.0 * np.finfo(float).eps * np.maximum(
        q_trial + 3.0 * float(par.G) * dg + scale, 1.0
    )
    allowed = np.maximum(32.0 * float(par.local_tolerance) * scale, roundoff_floor)
    failed = active & (np.abs(gap) > allowed)
    if np.any(failed):
        # The user tolerance may terminate the first bracket early even when
        # stress cancellation leaves a tiny consistency gap.  Refine only
        # those exceptional trials before rejecting the material point.
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            fmid = q_trial - 3.0 * float(par.G) * mid - par.hardening.yield_stress(k_old + mid)
            lo = np.where(active & (fmid > 0.0), mid, lo)
            hi = np.where(active & (fmid <= 0.0), mid, hi)
        mid = 0.5 * (lo + hi)
        candidates = np.stack((lo, mid, hi), axis=0)
        residuals = np.stack(
            tuple(
                np.abs(q_trial - 3.0 * float(par.G) * candidate
                       - par.hardening.yield_stress(k_old + candidate))
                for candidate in (lo, mid, hi)
            ),
            axis=0,
        )
        best = np.argmin(residuals, axis=0)
        dg = np.where(
            active,
            np.take_along_axis(candidates, best[None, ...], axis=0)[0],
            0.0,
        )
        gap = q_trial - 3.0 * float(par.G) * dg - par.hardening.yield_stress(k_old + dg)
        scale = np.maximum(q_trial, np.maximum(par.hardening.yield_stress(k_old), 1.0))
        roundoff_floor = 2048.0 * np.finfo(float).eps * np.maximum(
            q_trial + 3.0 * float(par.G) * dg + scale, 1.0
        )
        allowed = np.maximum(32.0 * float(par.local_tolerance) * scale, roundoff_floor)
        failed = active & (np.abs(gap) > allowed)
    if np.any(failed):
        bad = np.flatnonzero(failed)[0]
        raise J2MaterialError(
            "J2 return-mapping local consistency solve did not converge: "
            f"point={int(bad)}, |gap|={abs(float(gap.flat[bad])):.6g}, "
            f"allowed={float(allowed.flat[bad]):.6g}"
        )
    return dg


def update_small_strain(total_strain: np.ndarray, committed: J2State, par: J2Parameters) -> J2Response:
    """Evaluate one J2 trial from the last accepted state.

    total_strain shape is surface_shape + (3,3).  This routine never mutates
    committed state, which makes it safe inside FDM and MVI residual trials.
    """

    shape = committed.validate()
    eps = np.asarray(total_strain, float)
    if eps.shape != shape + (3, 3):
        raise J2MaterialError(f"J2 total_strain shape must be {shape + (3, 3)}, got {eps.shape}")
    if not np.isfinite(eps).all():
        raise J2MaterialError("J2 total_strain must be finite")
    eps = _sym(eps)
    ep_old = np.asarray(committed.plastic_strain, float)
    k_old = np.maximum(np.asarray(committed.kappa, float), 0.0)
    sig_trial = _elastic_stress(eps - ep_old, par)
    s_trial = _dev(sig_trial)
    q_trial = _equivalent(s_trial)
    sy_old = par.hardening.yield_stress(k_old)
    f_trial = q_trial - sy_old
    tol = float(par.yield_tolerance) * np.maximum(sy_old, 1.0)
    plastic = f_trial > tol
    dg = _plastic_increment(q_trial, k_old, plastic, par)

    flow_scale = np.zeros_like(q_trial, dtype=float)
    np.divide(1.5 * dg, q_trial, out=flow_scale, where=q_trial > np.finfo(float).tiny)
    dep = flow_scale[..., None, None] * s_trial
    ep_new = _sym(ep_old + dep)
    k_new = k_old + dg
    sy_new = par.hardening.yield_stress(k_new)

    # The scalar return solve supplies the target radius on the yield surface.
    # Rebuild the deviator by this radius instead of subtracting nearly equal
    # tensors.  That avoids loss of significance at the first plastic point
    # while retaining the same radial direction and frozen accepted history.
    radial_scale = np.ones_like(q_trial, dtype=float)
    np.divide(
        sy_new, q_trial, out=radial_scale,
        where=plastic & (q_trial > np.finfo(float).tiny),
    )
    s_new = np.where(
        plastic[..., None, None],
        radial_scale[..., None, None] * s_trial,
        s_trial,
    )
    stress = s_new + _trace(sig_trial)[..., None, None] * _I3 / 3.0
    q_new = _equivalent(s_new)
    yield_gap = np.abs(q_new - sy_new)
    yield_allowed = (
        64.0 * float(par.local_tolerance) * np.maximum(sy_new, 1.0)
    )
    if np.any(plastic & (yield_gap > yield_allowed)):
        bad = int(np.flatnonzero(plastic & (yield_gap > yield_allowed))[0])
        raise J2MaterialError(
            "J2 plastic correction did not return to yield surface: "
            f"point={bad}, |gap|={float(yield_gap.flat[bad]):.6g}, "
            f"allowed={float(yield_allowed.flat[bad]):.6g}, "
            f"q_trial={float(q_trial.flat[bad]):.6g}, "
            f"q_new={float(q_new.flat[bad]):.6g}, "
            f"sigma_y={float(sy_new.flat[bad]):.6g}, "
            f"dgamma={float(dg.flat[bad]):.6g}"
        )

    q_old = np.asarray(committed.equivalent_stress, float)
    had_history = k_old > float(par.local_tolerance)
    elastic = ~plastic
    unloading = elastic & had_history & (q_trial < q_old - tol)
    reloading = elastic & had_history & ~unloading & (q_trial >= q_old - tol)
    return J2Response(
        stress=stress,
        elastic_strain=eps - ep_new,
        state=J2State(ep_new, k_new, q_new),
        equivalent_stress_trial=q_trial,
        equivalent_stress=q_new,
        yield_stress=sy_new,
        yield_function_trial=f_trial,
        plastic_multiplier=dg,
        plastic_mask=plastic,
        unloading_mask=unloading,
        reloading_mask=reloading,
    )


def update_plane_stress(
    strain_xx: np.ndarray,
    strain_yy: np.ndarray,
    engineering_gamma_xy: np.ndarray,
    committed: J2State,
    par: J2Parameters,
) -> J2PlaneStressResponse:
    """Evaluate the plane-stress J2 update for all material points in a batch.

    update_small_strain is already array-native.  The thickness strain solve
    must therefore also operate on whole fields: scalar point-by-point solves
    made an MVI tangent cost one Python return-map loop per grid point.  The
    safeguarded secant/bracket algorithm below preserves the former local
    acceptance rules while solving every independent sigma_zz = 0 root in
    parallel from the same committed history.
    """

    shape = committed.validate()
    ex = np.asarray(strain_xx, float)
    ey = np.asarray(strain_yy, float)
    gam = np.asarray(engineering_gamma_xy, float)
    if ex.shape != shape or ey.shape != shape or gam.shape != shape:
        raise J2MaterialError(
            "J2 plane-stress in-plane strain shapes must match committed history"
        )
    if not np.isfinite(ex).all() or not np.isfinite(ey).all() or not np.isfinite(gam).all():
        raise J2MaterialError("J2 plane-stress strains must be finite")

    def local(ezz: np.ndarray) -> J2Response:
        total = np.zeros(shape + (3, 3), float)
        total[..., 0, 0] = ex
        total[..., 1, 1] = ey
        total[..., 0, 1] = total[..., 1, 0] = 0.5 * gam
        total[..., 2, 2] = ezz
        return update_small_strain(total, committed, par)

    # The elastic plane-stress value centres a deterministic bracket for each
    # point.  Plastic history only affects the bracket width, never the
    # committed state used by an evaluation.
    centre = -float(par.nu) * (ex + ey) / (1.0 - float(par.nu))
    history_scale = np.max(np.abs(np.asarray(committed.plastic_strain, float)), axis=(-2, -1))
    width = np.maximum(1.0e-12, np.abs(ex) + np.abs(ey) + np.abs(gam) + history_scale)
    lo = centre - width
    hi = centre + width
    rlo = np.asarray(local(lo).stress[..., 2, 2], float)
    rhi = np.asarray(local(hi).stress[..., 2, 2], float)

    bracketed = (rlo <= 0.0) & (rhi >= 0.0)
    for _ in range(int(par.max_local_iterations)):
        if np.all(bracketed):
            break
        width = np.where(bracketed, width, 2.0 * width)
        lo = np.where(bracketed, lo, centre - width)
        hi = np.where(bracketed, hi, centre + width)
        trial_lo = local(lo)
        trial_hi = local(hi)
        rlo = np.asarray(trial_lo.stress[..., 2, 2], float)
        rhi = np.asarray(trial_hi.stress[..., 2, 2], float)
        bracketed = (rlo <= 0.0) & (rhi >= 0.0)
    if not np.all(bracketed):
        bad = int(np.flatnonzero(~bracketed)[0])
        raise J2MaterialError(
            "J2 plane-stress local bracket does not contain sigma_zz=0: "
            f"point={bad}"
        )

    best_ezz = centre.copy() if shape else np.asarray(centre, float)
    best_abs_residual = np.full(shape, np.inf, float)
    solved_ezz = np.zeros(shape, float)
    converged = np.zeros(shape, bool)

    for _ in range(int(par.max_local_iterations)):
        active = ~converged
        if not np.any(active):
            break
        span = hi - lo
        denom = rhi - rlo
        midpoint = 0.5 * (lo + hi)
        secant = np.divide(
            lo * rhi - hi * rlo,
            denom,
            out=midpoint.copy(),
            where=np.abs(denom) > np.finfo(float).tiny,
        )
        margin = 0.05 * span
        valid_secant = (
            np.isfinite(secant)
            & (span > 0.0)
            & (secant > lo + margin)
            & (secant < hi - margin)
        )
        candidate_ezz = np.where(valid_secant, secant, midpoint)
        # A converged root is retained exactly; only active points are moved.
        candidate_ezz = np.where(active, candidate_ezz, solved_ezz)
        candidate = local(candidate_ezz)
        residual = np.asarray(candidate.stress[..., 2, 2], float)
        abs_residual = np.abs(residual)

        better = active & (abs_residual < best_abs_residual)
        best_ezz = np.where(better, candidate_ezz, best_ezz)
        best_abs_residual = np.where(better, abs_residual, best_abs_residual)

        stress_scale = np.maximum(
            float(par.E), np.maximum(np.asarray(candidate.yield_stress, float), 1.0)
        )
        just_converged = active & (
            abs_residual <= 32.0 * float(par.local_tolerance) * stress_scale
        )
        solved_ezz = np.where(just_converged, candidate_ezz, solved_ezz)
        converged = converged | just_converged

        below = active & ~just_converged & (residual < 0.0)
        above = active & ~just_converged & ~below
        lo = np.where(below, candidate_ezz, lo)
        rlo = np.where(below, residual, rlo)
        hi = np.where(above, candidate_ezz, hi)
        rhi = np.where(above, residual, rhi)

    # For a point that did not meet the strict tolerance, retain the best
    # state actually evaluated instead of replacing it by a bracket midpoint.
    ezz = np.where(converged, solved_ezz, best_ezz)
    response = local(ezz)
    residual = np.abs(np.asarray(response.stress[..., 2, 2], float))
    stress_scale = np.maximum(
        float(par.E), np.maximum(np.asarray(response.yield_stress, float), 1.0)
    )
    accepted = converged | (
        residual <= 128.0 * float(par.local_tolerance) * stress_scale
    )
    if not np.all(accepted):
        bad = int(np.flatnonzero(~accepted)[0])
        raise J2MaterialError(
            "J2 plane-stress local solve did not converge: "
            f"point={bad}, |sigma_zz|={float(residual.flat[bad]):.6g}, "
            f"tolerance={float((128.0 * float(par.local_tolerance) * stress_scale).flat[bad]):.6g}"
        )

    total = np.zeros(shape + (3, 3), float)
    total[..., 0, 0] = ex
    total[..., 1, 1] = ey
    total[..., 0, 1] = total[..., 1, 0] = 0.5 * gam
    total[..., 2, 2] = ezz
    return J2PlaneStressResponse(
        stress=response.stress,
        elastic_strain=response.elastic_strain,
        state=response.state,
        equivalent_stress_trial=response.equivalent_stress_trial,
        equivalent_stress=response.equivalent_stress,
        yield_stress=response.yield_stress,
        yield_function_trial=response.yield_function_trial,
        plastic_multiplier=response.plastic_multiplier,
        plastic_mask=response.plastic_mask,
        unloading_mask=response.unloading_mask,
        reloading_mask=response.reloading_mask,
        total_strain=total,
    )

def plane_stress_algorithmic_tangent(
    strain_xx: np.ndarray,
    strain_yy: np.ndarray,
    engineering_gamma_xy: np.ndarray,
    committed: J2State,
    par: J2Parameters,
    *,
    response: J2PlaneStressResponse | None = None,
) -> np.ndarray:
    """Algorithmic plane-stress tangent from the frozen-history J2 update.

    The three in-plane strain directions are perturbed once for the complete
    material-point array.  This is a local constitutive linearization, not a
    global MVI finite difference; the returned tensor maps
    [d eps_xx, d eps_yy, d gamma_xy] to [d sigma_xx, d sigma_yy, d tau_xy].
    """
    ex = np.asarray(strain_xx, float)
    ey = np.asarray(strain_yy, float)
    gam = np.asarray(engineering_gamma_xy, float)
    if response is None:
        response = update_plane_stress(ex, ey, gam, committed, par)
    s0 = np.stack((
        response.stress[..., 0, 0],
        response.stress[..., 1, 1],
        response.stress[..., 0, 1],
    ), axis=-1)
    scale = np.maximum.reduce((
        np.abs(ex), np.abs(ey), np.abs(gam),
        np.full_like(ex, 1.0e-5, dtype=float),
    ))
    step = np.maximum(1.0e-8, 1.0e-5 * scale)
    tangent = np.empty(ex.shape + (3, 3), float)
    for col in range(3):
        xp, yp, gp = ex.copy(), ey.copy(), gam.copy()
        if col == 0:
            xp += step
        elif col == 1:
            yp += step
        else:
            gp += step
        trial = update_plane_stress(xp, yp, gp, committed, par)
        sp = np.stack((
            trial.stress[..., 0, 0],
            trial.stress[..., 1, 1],
            trial.stress[..., 0, 1],
        ), axis=-1)
        tangent[..., :, col] = (sp - s0) / step[..., None]
    if not np.all(np.isfinite(tangent)):
        raise J2MaterialError("J2 plane-stress algorithmic tangent is non-finite")
    return tangent


__all__ = [
    "J2MaterialError",
    "J2Hardening",
    "J2Parameters",
    "J2State",
    "J2Response",
    "J2PlaneStressResponse",
    "update_small_strain",
    "update_plane_stress",
    "plane_stress_algorithmic_tangent",
]
