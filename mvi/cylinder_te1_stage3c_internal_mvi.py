#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 closed-cylinder Stage 3C:
strict internal MVI X/Y solver at frozen material field GB.

The algorithm follows the already accepted refined Kirichenko-Krysko MVI path:
- all retained terms and all six TE1 fields of one direction are corrected jointly;
- partner profiles are frozen during the entire half-cycle;
- each half-cycle converges to its own projected equations before switching;
- r = T^T W R;
- A = T^T W (K_T T);
- A dc = -r;
- direct np.linalg.solve only;
- no 2-D full-grid Newton correction;
- no SVD/pinv/lstsq/QR/Cholesky/regularization.
- correction-step control is enabled for the strict MVI Newton update
  (ported from the KL strict solver): the trial full correction is
  checked and alpha is reduced if the projected residual increases.

K_T*T is evaluated by the same residual-perturbation principle already present
in the accepted working MVI code: for a normalized profile direction dU,
    K_T dU ~= [R(U+h dU)-R(U)]/h.
The finite-difference step is an EXPLICIT input. No hidden universal default
is assigned for physical calculations.

No literature benchmark data are embedded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage3b_boundary as b3
import final_v3_fd as v3fd


class MVIInternalError(RuntimeError):
    pass


class SingularReducedSystem(MVIInternalError):
    pass


class HalfstepDidNotConverge(MVIInternalError):
    pass


@dataclass(frozen=True)
class InternalMVISettings:
    projected_tolerance: float
    increment_tolerance: float
    tangent_fd_step: float
    max_halfstep_iterations: int
    max_xy_cycles: int
    max_correction_halvings: int = 5

    def __post_init__(self):
        if self.projected_tolerance <= 0.0:
            raise ValueError("projected_tolerance must be positive")
        if self.increment_tolerance <= 0.0:
            raise ValueError("increment_tolerance must be positive")
        if self.tangent_fd_step <= 0.0:
            raise ValueError("tangent_fd_step must be positive")
        if self.max_halfstep_iterations < 1:
            raise ValueError("max_halfstep_iterations must be >=1")
        if self.max_xy_cycles < 1:
            raise ValueError("max_xy_cycles must be >=1")
        if self.max_correction_halvings < 0:
            raise ValueError("max_correction_halvings must be >=0")


@dataclass(frozen=True)
class HalfstepRecord:
    direction: str
    iterations: int
    projected_inf: float
    correction_norm: float
    status: str


@dataclass
class InternalMVIResult:
    state: mvi.ProfileStateTE1
    converged: bool
    status: str
    cycles: int
    history: list
    final_projected_inf: float
    final_increment_rel: float


ResidualFunction = Callable[[mvi.ProfileStateTE1], np.ndarray]


def _field_vector(state: mvi.ProfileStateTE1, grid: mvi.CylinderGrid):
    return mvi.reconstruct_vector(state, grid)


def _relative_field_change(old, new, grid):
    a = _field_vector(old, grid)
    b = _field_vector(new, grid)
    return float(np.linalg.norm(b-a) / max(np.linalg.norm(b), 1.0e-300))


def _active_mask_for_direction(state, grid, bc, direction):
    d = direction.upper()
    if d == "X":
        # Exact homogeneous essential end constraints from verified Stage 3B.
        return b3.homogeneous_geometric_x_active_mask(state, grid, bc)
    if d == "Y":
        return np.ones_like(state.Y, dtype=bool)
    raise ValueError("direction must be X or Y")


def _weighted_projected(Tn, residual, weights):
    return Tn.T @ (weights * residual)


def _direction_data(state, grid, bc, direction):
    mask = _active_mask_for_direction(state, grid, bc, direction)
    T, ids = mvi.build_direction_tangent(state, grid, direction, mask)
    norms = np.linalg.norm(T, axis=0)
    active = norms > 1.0e-14
    if not np.any(active):
        return mask, T, ids, norms, active, np.zeros((T.shape[0], 0))
    Tn = T[:, active] / norms[active][None, :]
    return mask, T, ids, norms, active, Tn


def _residual_vector(residual_fn, state, grid):
    r = np.asarray(residual_fn(state), dtype=float)
    if r.shape == (6, grid.ns, grid.nx):
        r = r.reshape(-1)
    if r.shape != (grid.n_full_dofs,):
        raise MVIInternalError(
            f"residual must have shape {(grid.n_full_dofs,)} or "
            f"{(6,grid.ns,grid.nx)}, got {r.shape}"
        )
    if not np.all(np.isfinite(r)):
        raise MVIInternalError("NONFINITE_FULL_RESIDUAL")
    return r


def _mvi_characteristic_scale(state, grid, quadrature, direction, ids):
    """J-12 physical field scale for each active TE1 profile parameter.

    For a normalized MVI tangent direction, a_j=c_j*||T_j|| has the dimension
    of the reconstructed generalized displacement field.  The reviewer requires
    the same V3 scale as for the full nodal TE1 vector: h for tau=0 and h/L
    for tau=1.
    """
    h_shell = v3fd.thickness_from_quadrature(quadrature)
    L = float(grid.length_x)
    n_dir = grid.nx if str(direction).upper() == "X" else grid.ns
    out = np.empty(len(ids), dtype=float)
    for k, flat_idx in enumerate(ids):
        block = int(flat_idx) // int(n_dir)
        field = block // int(state.rank)
        tau = field % 2
        out[k] = h_shell / (L ** tau)
    return out


def directional_tangent_action_fd(
    state,
    grid,
    bc,
    direction,
    residual_fn,
    *,
    tangent_fd_step,
    quadrature,
):
    """Return K_T*Tn by J-12 V3-scaled central residual differences."""
    mask, _T, ids, norms, active, Tn = _direction_data(
        state, grid, bc, direction
    )
    if Tn.shape[1] == 0:
        raise MVIInternalError(
            f"{direction.upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )

    active_positions = np.flatnonzero(active)
    out = np.empty((grid.n_full_dofs, active_positions.size), dtype=float)

    current, ids2 = mvi.get_active_parameters(state, direction, mask)
    if not np.array_equal(ids, ids2):
        raise MVIInternalError("active parameter ordering changed")
    char = _mvi_characteristic_scale(state, grid, quadrature, direction, ids)
    amplitude = current * norms
    hj = v3fd.relative_steps(amplitude, char, tangent_fd_step)

    for jcol, pos in enumerate(active_positions):
        plus = state.copy()
        cp, ids3 = mvi.get_active_parameters(plus, direction, mask)
        if not np.array_equal(ids, ids3):
            raise MVIInternalError("active parameter ordering changed")
        cp[pos] += hj[pos] / norms[pos]
        mvi.set_active_parameters(plus, direction, ids, cp)

        minus = state.copy()
        cm, ids4 = mvi.get_active_parameters(minus, direction, mask)
        if not np.array_equal(ids, ids4):
            raise MVIInternalError("active parameter ordering changed")
        cm[pos] -= hj[pos] / norms[pos]
        mvi.set_active_parameters(minus, direction, ids, cm)

        Rp = _residual_vector(residual_fn, plus, grid)
        Rm = _residual_vector(residual_fn, minus, grid)
        out[:, jcol] = (Rp - Rm) / (2.0 * hj[pos])
    return out, Tn, mask, ids, norms, active


def _direct_solve(A, projected, direction):
    try:
        delta = np.linalg.solve(
            np.asarray(A, dtype=float),
            -np.asarray(projected, dtype=float)
        )
    except np.linalg.LinAlgError as exc:
        raise SingularReducedSystem(
            f"SINGULAR_REDUCED_MVI_SYSTEM / {direction.upper()}_HALFSTEP"
        ) from exc
    if not np.all(np.isfinite(delta)):
        raise SingularReducedSystem(
            f"NONFINITE_REDUCED_MVI_CORRECTION / {direction.upper()}_HALFSTEP"
        )
    return delta


def projected_inf_for_direction(state, grid, bc, direction, residual_fn, weights):
    R = _residual_vector(residual_fn, state, grid)
    _, _, _, _, _active, Tn = _direction_data(state, grid, bc, direction)
    if Tn.shape[1] == 0:
        # If the actual full residual is already zero, a zero-direction state is
        # a valid exact equilibrium (e.g. zero load and zero field).
        if float(np.max(np.abs(R))) == 0.0:
            return 0.0
        raise MVIInternalError(
            f"{direction.upper()}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
        )
    projected = _weighted_projected(Tn, R, weights)
    return float(np.max(np.abs(projected))) if projected.size else 0.0


def solve_strict_halfstep(
    state,
    grid,
    bc,
    direction,
    residual_fn,
    settings,
    *,
    quadrature,
    weights=None,
):
    """
    Frozen-partner nonlinear half-cycle.

    No balancing, orthogonalization, relaxation or line-search is performed.
    """
    d = direction.upper()
    w = (
        grid.residual_projection_weights()
        if weights is None else np.asarray(weights, dtype=float)
    )
    if w.shape != (grid.n_full_dofs,):
        raise ValueError("weights shape mismatch")

    last_corr = 0.0
    for it in range(1, settings.max_halfstep_iterations + 1):
        R = _residual_vector(residual_fn, state, grid)

        mask, _T, ids, norms, active, Tn = _direction_data(
            state, grid, bc, d
        )
        if Tn.shape[1] == 0:
            if float(np.max(np.abs(R))) == 0.0:
                return HalfstepRecord(
                    d, it-1, 0.0, 0.0, "CONVERGED_REDUCED_MVI"
                )
            raise MVIInternalError(
                f"{d}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS"
            )

        projected = _weighted_projected(Tn, R, w)
        pinf = float(np.max(np.abs(projected))) if projected.size else 0.0
        if pinf <= settings.projected_tolerance:
            return HalfstepRecord(
                d, it-1, pinf, last_corr, "CONVERGED_REDUCED_MVI"
            )

        KT_Tn, Tn2, _mask2, ids2, _norms2, active2 = (
            directional_tangent_action_fd(
                state, grid, bc, d, residual_fn,
                tangent_fd_step=settings.tangent_fd_step,
                quadrature=quadrature,
            )
        )
        if not np.array_equal(ids, ids2):
            raise MVIInternalError("direction data changed inside halfstep")
        if not np.array_equal(active, active2):
            raise MVIInternalError("active directions changed inside halfstep")
        if np.max(np.abs(Tn-Tn2)) > 1.0e-12:
            raise MVIInternalError("normalized tangent changed inside halfstep")

        A = Tn.T @ (w[:, None] * KT_Tn)
        delta_normalized = _direct_solve(A, projected, d)

        current, ids3 = mvi.get_active_parameters(state, d, mask)
        if not np.array_equal(ids, ids3):
            raise MVIInternalError("active parameter ordering changed")

        delta_parameters = np.zeros_like(current)
        positions = np.flatnonzero(active)
        delta_parameters[positions] = (
            delta_normalized / norms[active]
        )

        # Correction-step control (ported from the KL strict-MVI solver,
        # cylinder_kl_stage8b5_strict_mvi.py): try the full Newton
        # correction first, accept only if the projected residual does not
        # increase; otherwise halve alpha and retry. Without this guard an
        # overshooting step (e.g. at large deflection, or a spatially rough
        # frozen material field such as a spatially modulated stiffness) has
        # no way to recover, and the halfstep can run away geometrically
        # until it exhausts max_halfstep_iterations or overflows.
        alpha = 1.0
        max_halvings = int(getattr(settings, "max_correction_halvings", 5))
        min_alpha = 1.0 / (2.0 ** max_halvings)
        accepted = False
        for _ih in range(max_halvings + 1):
            trial_state = state.copy()
            mvi.set_active_parameters(
                trial_state, d, ids, current + alpha * delta_parameters
            )
            trial_pinf = projected_inf_for_direction(
                trial_state, grid, bc, d, residual_fn, w
            )
            if trial_pinf <= pinf:
                mvi.set_active_parameters(
                    state, d, ids, current + alpha * delta_parameters
                )
                last_corr = float(alpha * np.linalg.norm(delta_parameters))
                accepted = True
                break
            alpha *= 0.5
            if alpha < min_alpha:
                break
        if not accepted:
            raise MVIInternalError("CORRECTION_STEP_FAILED")

    pinf = projected_inf_for_direction(
        state, grid, bc, d, residual_fn, w
    )
    if pinf <= settings.projected_tolerance:
        return HalfstepRecord(
            d, settings.max_halfstep_iterations, pinf, last_corr,
            "CONVERGED_REDUCED_MVI"
        )
    raise HalfstepDidNotConverge(
        f"MAX_HALFSTEP_ITERATIONS / {d}_HALFSTEP / projected_inf={pinf:.17g}"
    )


def solve_internal_mvi(
    initial_state,
    grid,
    bc,
    residual_fn,
    settings,
    *,
    quadrature,
    weights=None,
):
    """
    Strict Y -> X cycles at frozen material state.

    Convergence is accepted only after complete Y and X half-cycles and requires:
      - both current projected directional residuals <= projected_tolerance;
      - relative change of the reconstructed six-field vector
        <= increment_tolerance.
    """
    # Stage 3B explicitly forbids silently choosing a rigid-mode removal rule.
    bc.check_rigid_mode_requirement()

    state = initial_state.copy()
    state.validate_grid(grid)
    # Exact homogeneous geometric endpoint enforcement before the first cycle.
    b3.enforce_homogeneous_geometric_x_profiles(state, grid, bc)

    w = (
        grid.residual_projection_weights()
        if weights is None else np.asarray(weights, dtype=float)
    )
    history = []
    final_p = float("inf")
    final_inc = float("inf")

    for cyc in range(1, settings.max_xy_cycles + 1):
        before = state.copy()

        hy = solve_strict_halfstep(
            state, grid, bc, "Y", residual_fn, settings,
            quadrature=quadrature, weights=w
        )
        hx = solve_strict_halfstep(
            state, grid, bc, "X", residual_fn, settings,
            quadrature=quadrature, weights=w
        )

        # Homogeneous geometric constraints remain exact after X correction.
        b3.enforce_homogeneous_geometric_x_profiles(state, grid, bc)

        py = projected_inf_for_direction(
            state, grid, bc, "Y", residual_fn, w
        )
        px = projected_inf_for_direction(
            state, grid, bc, "X", residual_fn, w
        )
        final_p = max(px, py)
        final_inc = _relative_field_change(before, state, grid)

        history.append({
            "cycle": cyc,
            "Y_half_iterations": hy.iterations,
            "X_half_iterations": hx.iterations,
            "Y_projected_inf_after_cycle": py,
            "X_projected_inf_after_cycle": px,
            "projected_inf": final_p,
            "field_increment_rel": final_inc,
        })

        if (
            final_p <= settings.projected_tolerance
            and final_inc <= settings.increment_tolerance
        ):
            return InternalMVIResult(
                state=state,
                converged=True,
                status="CONVERGED_REDUCED_MVI",
                cycles=cyc,
                history=history,
                final_projected_inf=final_p,
                final_increment_rel=final_inc,
            )

    return InternalMVIResult(
        state=state,
        converged=False,
        status="MAX_XY_CYCLES",
        cycles=settings.max_xy_cycles,
        history=history,
        final_projected_inf=final_p,
        final_increment_rel=final_inc,
    )


def make_physical_residual_function(
    ops: c3.CylinderSpatialOperators,
    *,
    K0: float,
    GB,
    quadrature: s2.ThicknessQuadrature,
    generalized_load,
    bc: b3.CylinderEndBC,
):
    """
    Connect verified Stage 3A raw physical residual with verified Stage 3B BCs.
    GB is frozen by this closure for the entire internal MVI solve.
    """
    p = np.asarray(generalized_load, dtype=float)
    expected = (6, ops.grid.ns, ops.grid.nx)
    if p.shape != expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def residual(state):
        raw, cache, (q, _qx, _qs) = c3.raw_strong_residual_from_profiles(
            state, ops,
            K0=K0,
            GB=GB,
            quadrature=quadrature,
            generalized_load=p,
        )
        full = b3.assemble_strong_residual_with_end_bcs(
            raw, q, cache, bc
        )
        return full.reshape(-1)

    return residual
