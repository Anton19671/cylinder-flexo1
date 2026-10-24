#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite-strain multiplicative J2 return mapping for the GL geometry branch.

The state is a plastic deformation gradient Fp and accumulated equivalent
plastic strain kappa at every integration point.  The update uses an elastic
Hencky strain and Kirchhoff stress, isochoric associative flow, and radial
return in principal elastic-logarithmic strain space.  It returns physical
Cauchy and second Piola stresses, so this path does not mix Cauchy and
Piola--Kirchhoff measures.

This module is independent of Birger and of the small-strain J2 module except
for shared elastic/hardening parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from final_j2_incremental import J2MaterialError, J2Parameters


_I3 = np.eye(3, dtype=float)


@dataclass
class FiniteJ2State:
    """Committed finite-plasticity history at one or many integration points."""

    Fp: np.ndarray
    kappa: np.ndarray
    equivalent_kirchhoff_stress: np.ndarray

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "FiniteJ2State":
        fp = np.broadcast_to(_I3, tuple(shape) + (3, 3)).copy()
        return cls(fp, np.zeros(tuple(shape), float), np.zeros(tuple(shape), float))

    def copy(self) -> "FiniteJ2State":
        return FiniteJ2State(self.Fp.copy(), self.kappa.copy(), self.equivalent_kirchhoff_stress.copy())

    def validate(self) -> Tuple[int, ...]:
        fp = np.asarray(self.Fp, float)
        k = np.asarray(self.kappa, float)
        q = np.asarray(self.equivalent_kirchhoff_stress, float)
        if fp.ndim < 2 or fp.shape[-2:] != (3, 3):
            raise J2MaterialError("finite J2 Fp shape must end with (3,3)")
        shape = fp.shape[:-2]
        if k.shape != shape or q.shape != shape:
            raise J2MaterialError("finite J2 scalar histories must match Fp point shape")
        if not np.isfinite(fp).all() or not np.isfinite(k).all() or not np.isfinite(q).all():
            raise J2MaterialError("finite J2 state must be finite")
        det = np.linalg.det(fp.reshape((-1, 3, 3)))
        if np.any(det <= 0.0) or np.any(np.abs(det - 1.0) > 1.0e-8):
            raise J2MaterialError("finite J2 Fp must be positive and isochoric")
        if np.any(k < -1.0e-14) or np.any(q < -1.0e-14):
            raise J2MaterialError("finite J2 scalar history must be non-negative")
        return shape


@dataclass
class FiniteJ2Response:
    """Uncommitted finite J2 local response."""

    cauchy: np.ndarray
    second_piola: np.ndarray
    kirchhoff: np.ndarray
    elastic_log_strain: np.ndarray
    state: FiniteJ2State
    equivalent_trial: np.ndarray
    equivalent_stress: np.ndarray
    yield_stress: np.ndarray
    plastic_multiplier: np.ndarray
    plastic_mask: np.ndarray
    unloading_mask: np.ndarray
    reloading_mask: np.ndarray


def _dev(a: np.ndarray) -> np.ndarray:
    return a - np.trace(a) * _I3 / 3.0


def _q(a: np.ndarray) -> float:
    return float(np.sqrt(max(1.5 * float(np.einsum("ij,ij->", a, a)), 0.0)))


def _sym_log_spd(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
    if np.any(vals <= 0.0) or not np.isfinite(vals).all():
        raise J2MaterialError("finite J2 requires positive definite elastic left Cauchy-Green tensor")
    return (vecs * np.log(vals)) @ vecs.T


def _sym_exp(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
    return (vecs * np.exp(vals)) @ vecs.T


def _solve_increment(q_trial: float, kappa: float, par: J2Parameters) -> float:
    sy0 = float(par.hardening.yield_stress(np.array(kappa)))
    if q_trial - sy0 <= float(par.yield_tolerance) * max(sy0, 1.0):
        return 0.0
    lo = 0.0
    hi = q_trial / max(3.0 * float(par.G), np.finfo(float).tiny)
    for _ in range(int(par.max_local_iterations)):
        mid = 0.5 * (lo + hi)
        f = q_trial - 3.0 * float(par.G) * mid - float(par.hardening.yield_stress(np.array(kappa + mid)))
        if f > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= float(par.local_tolerance):
            break
    dg = 0.5 * (lo + hi)
    gap = q_trial - 3.0 * float(par.G) * dg - float(par.hardening.yield_stress(np.array(kappa + dg)))
    if abs(gap) > 32.0 * float(par.local_tolerance) * max(q_trial, sy0, 1.0):
        raise J2MaterialError("finite J2 local return mapping did not converge")
    return dg


def update_finite_strain(F: np.ndarray, committed: FiniteJ2State, par: J2Parameters) -> FiniteJ2Response:
    """Return-map a deformation-gradient field from accepted finite J2 history.

    The returned state is trial-only.  It is safe to recompute inside every
    global Newton or MVI residual evaluation because committed is never mutated.
    """

    shape = committed.validate()
    Fa = np.asarray(F, float)
    if Fa.shape != shape + (3, 3):
        raise J2MaterialError(f"finite J2 F shape must be {shape + (3, 3)}, got {Fa.shape}")
    if not np.isfinite(Fa).all():
        raise J2MaterialError("finite J2 F must be finite")

    cauchy = np.empty_like(Fa)
    second_piola = np.empty_like(Fa)
    kirchhoff = np.empty_like(Fa)
    elog = np.empty_like(Fa)
    fp_new = np.empty_like(Fa)
    k_new = np.empty(shape, float)
    qtrial = np.empty(shape, float)
    qnew = np.empty(shape, float)
    synew = np.empty(shape, float)
    dgout = np.empty(shape, float)
    pmask = np.zeros(shape, bool)
    unloading = np.zeros(shape, bool)
    reloading = np.zeros(shape, bool)

    flatF = Fa.reshape((-1, 3, 3))
    flatFp = np.asarray(committed.Fp, float).reshape((-1, 3, 3))
    flatK = np.asarray(committed.kappa, float).reshape(-1)
    flatQold = np.asarray(committed.equivalent_kirchhoff_stress, float).reshape(-1)
    for n, (Fn, Fpn, kn, qold) in enumerate(zip(flatF, flatFp, flatK, flatQold)):
        J = float(np.linalg.det(Fn))
        if not np.isfinite(J) or J <= 0.0:
            raise J2MaterialError("finite J2 requires det(F)>0")
        Fe_trial = Fn @ np.linalg.inv(Fpn)
        be_trial = Fe_trial @ Fe_trial.T
        e_trial = 0.5 * _sym_log_spd(be_trial)
        dev_e_trial = _dev(e_trial)
        tau_trial = float(par.K) * float(np.trace(e_trial)) * _I3 + 2.0 * float(par.G) * dev_e_trial
        s_trial = _dev(tau_trial)
        qt = _q(s_trial)
        dg = _solve_increment(qt, float(kn), par)
        if dg > 0.0:
            dev_e = dev_e_trial - 1.5 * dg * s_trial / max(qt, np.finfo(float).tiny)
        else:
            dev_e = dev_e_trial
        e_new = dev_e + float(np.trace(e_trial)) * _I3 / 3.0
        tau = float(par.K) * float(np.trace(e_new)) * _I3 + 2.0 * float(par.G) * _dev(e_new)
        qn = _q(_dev(tau))
        knew = float(kn) + dg
        syn = float(par.hardening.yield_stress(np.array(knew)))
        if dg > 0.0 and abs(qn - syn) > 64.0 * float(par.local_tolerance) * max(syn, 1.0):
            raise J2MaterialError("finite J2 plastic state is off the yield surface")

        # Preserve the trial elastic rotation while returning only the elastic
        # stretch.  Fp = Fe^{-1} F remains isochoric up to round-off.
        U, _singular, Vt = np.linalg.svd(Fe_trial)
        Re = U @ Vt
        Fe_new = _sym_exp(e_new) @ Re
        Fpnext = np.linalg.solve(Fe_new, Fn)
        det_fp = float(np.linalg.det(Fpnext))
        if det_fp <= 0.0 or not np.isfinite(det_fp):
            raise J2MaterialError("finite J2 produced invalid plastic deformation gradient")
        Fpnext /= det_fp ** (1.0 / 3.0)

        sigma = tau / J
        Finv = np.linalg.inv(Fn)
        S = J * Finv @ sigma @ Finv.T
        i = np.unravel_index(n, shape) if shape else ()
        cauchy[i] = sigma
        second_piola[i] = 0.5 * (S + S.T)
        kirchhoff[i] = tau
        elog[i] = e_new
        fp_new[i] = Fpnext
        k_new[i] = knew
        qtrial[i] = qt
        qnew[i] = qn
        synew[i] = syn
        dgout[i] = dg
        pmask[i] = dg > 0.0
        tol = float(par.yield_tolerance) * max(syn, 1.0)
        unloading[i] = (dg == 0.0 and kn > float(par.local_tolerance) and qt < qold - tol)
        reloading[i] = (dg == 0.0 and kn > float(par.local_tolerance) and not unloading[i] and qt >= qold - tol)

    return FiniteJ2Response(
        cauchy=cauchy,
        second_piola=second_piola,
        kirchhoff=kirchhoff,
        elastic_log_strain=elog,
        state=FiniteJ2State(fp_new, k_new, qnew),
        equivalent_trial=qtrial,
        equivalent_stress=qnew,
        yield_stress=synew,
        plastic_multiplier=dgout,
        plastic_mask=pmask,
        unloading_mask=unloading,
        reloading_mask=reloading,
    )


def _stretch_from_right_cauchy_green(c11: float, c22: float, c12: float, lambda3: float) -> np.ndarray:
    """Choose the positive stretch U with U.T U=C.

    The constitutive law is isotropic, hence replacing F by its right stretch
    does not discard a physical stress component: it removes only a superposed
    rigid rotation.  The returned second Piola stress is the measure conjugate
    to the Green strain used by the KL GL weak form.
    """
    C = np.array(((c11, c12, 0.0), (c12, c22, 0.0), (0.0, 0.0, lambda3 * lambda3)), float)
    vals, vecs = np.linalg.eigh(C)
    if np.any(vals <= 0.0) or not np.isfinite(vals).all():
        raise J2MaterialError("finite J2 plane-stress in-plane C must be positive definite")
    return (vecs * np.sqrt(vals)) @ vecs.T


def _plane_stress_point(
    c11: float, c22: float, c12: float, old: FiniteJ2State, par: J2Parameters,
) -> FiniteJ2Response:
    """Solve S_33=0 at one point, keeping the accepted J2 history fixed."""
    if not np.isfinite((c11, c22, c12)).all():
        raise J2MaterialError("finite J2 plane-stress C components must be finite")
    if c11 <= 0.0 or c22 <= 0.0 or c11 * c22 - c12 * c12 <= 0.0:
        raise J2MaterialError("finite J2 plane-stress requires positive definite in-plane C")

    def response_at(log_lambda3: float) -> FiniteJ2Response:
        lam = float(np.exp(log_lambda3))
        F = _stretch_from_right_cauchy_green(c11, c22, c12, lam)
        return update_finite_strain(F, old, par)

    center = 0.0
    center_response = response_at(center)
    scale = max(float(par.E), 1.0)
    tol = 64.0 * float(par.local_tolerance) * scale
    rc = float(center_response.second_piola[2, 2])
    if abs(rc) <= tol:
        return center_response

    lower = center - 1.0
    upper = center + 1.0
    left = response_at(lower)
    right = response_at(upper)
    rl = float(left.second_piola[2, 2])
    rr = float(right.second_piola[2, 2])
    for _ in range(16):
        if rl == 0.0:
            return left
        if rr == 0.0:
            return right
        if rl * rr < 0.0:
            break
        lower -= 1.0
        upper += 1.0
        left = response_at(lower)
        right = response_at(upper)
        rl = float(left.second_piola[2, 2])
        rr = float(right.second_piola[2, 2])
    else:
        raise J2MaterialError("finite J2 plane-stress thickness-stretch bracket not found")

    result = center_response
    for _ in range(max(64, int(par.max_local_iterations))):
        mid = 0.5 * (lower + upper)
        result = response_at(mid)
        rm = float(result.second_piola[2, 2])
        if abs(rm) <= tol or abs(upper - lower) <= float(par.local_tolerance):
            return result
        if rl * rm <= 0.0:
            upper = mid
            right = result
            rr = rm
        else:
            lower = mid
            left = result
            rl = rm
    if abs(float(result.second_piola[2, 2])) > tol:
        raise J2MaterialError("finite J2 plane-stress solve did not converge")
    return result


def update_finite_plane_stress(
    C11: np.ndarray, C22: np.ndarray, C12: np.ndarray,
    committed: FiniteJ2State, par: J2Parameters,
) -> FiniteJ2Response:
    """Finite J2 radial return with material-plane condition S_33=0.

    C11, C22 and C12 are the material in-plane right Cauchy--Green
    components.  This function is non-mutating: it may be called for every
    global Newton or MVI residual probe with one fixed accepted history.
    """
    shape = committed.validate()
    arrays = [np.asarray(a, float) for a in (C11, C22, C12)]
    if any(a.shape != shape for a in arrays):
        raise J2MaterialError(f"finite J2 plane-stress C fields must have shape {shape}")
    c11, c22, c12 = arrays

    cauchy = np.empty(shape + (3, 3), float)
    second_piola = np.empty_like(cauchy)
    kirchhoff = np.empty_like(cauchy)
    elog = np.empty_like(cauchy)
    fp = np.empty_like(cauchy)
    kappa = np.empty(shape, float)
    qtrial = np.empty(shape, float)
    qnew = np.empty(shape, float)
    sy = np.empty(shape, float)
    dg = np.empty(shape, float)
    plastic = np.zeros(shape, bool)
    unloading = np.zeros(shape, bool)
    reloading = np.zeros(shape, bool)

    for flat, index in enumerate(np.ndindex(shape)):
        old = FiniteJ2State(
            np.asarray(committed.Fp[index], float),
            np.asarray(committed.kappa[index], float),
            np.asarray(committed.equivalent_kirchhoff_stress[index], float),
        )
        response = _plane_stress_point(
            float(c11[index]), float(c22[index]), float(c12[index]), old, par
        )
        cauchy[index] = response.cauchy
        second_piola[index] = response.second_piola
        kirchhoff[index] = response.kirchhoff
        elog[index] = response.elastic_log_strain
        fp[index] = response.state.Fp
        kappa[index] = response.state.kappa
        qtrial[index] = response.equivalent_trial
        qnew[index] = response.equivalent_stress
        sy[index] = response.yield_stress
        dg[index] = response.plastic_multiplier
        plastic[index] = response.plastic_mask
        unloading[index] = response.unloading_mask
        reloading[index] = response.reloading_mask

    return FiniteJ2Response(
        cauchy=cauchy,
        second_piola=second_piola,
        kirchhoff=kirchhoff,
        elastic_log_strain=elog,
        state=FiniteJ2State(fp, kappa, qnew),
        equivalent_trial=qtrial,
        equivalent_stress=qnew,
        yield_stress=sy,
        plastic_multiplier=dg,
        plastic_mask=plastic,
        unloading_mask=unloading,
        reloading_mask=reloading,
    )


__all__ = ["FiniteJ2State", "FiniteJ2Response", "update_finite_strain", "update_finite_plane_stress"]