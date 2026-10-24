#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math
import numpy as np

FIELD_NAMES = ("u0", "u1", "v0", "v1", "w0", "w1")
N_FIELDS = 6

class ReducedSystemError(RuntimeError):
    pass

class KernelContractError(RuntimeError):
    pass

@dataclass(frozen=True)
class CylinderGrid:
    length_x: float
    radius: float
    nx: int
    ns: int

    def __post_init__(self):
        if self.length_x <= 0 or self.radius <= 0:
            raise ValueError("length_x and radius must be positive")
        if self.nx < 2 or self.ns < 3:
            raise ValueError("nx>=2 and ns>=3 required")

    @property
    def circumference(self):
        return 2.0 * math.pi * self.radius

    @property
    def x(self):
        return np.linspace(0.0, self.length_x, self.nx)

    @property
    def s(self):
        # unique periodic nodes; 2*pi*R is not duplicated
        return np.linspace(0.0, self.circumference, self.ns, endpoint=False)

    @property
    def dx(self):
        return self.length_x / (self.nx - 1)

    @property
    def ds(self):
        return self.circumference / self.ns

    @property
    def n_surface_nodes(self):
        return self.nx * self.ns

    @property
    def n_full_dofs(self):
        return N_FIELDS * self.n_surface_nodes

    def residual_projection_weights(self):
        wx = np.full(self.nx, self.dx)
        wx[0] *= 0.5
        wx[-1] *= 0.5
        ws = np.full(self.ns, self.ds)
        w2 = np.outer(ws, wx)
        return np.tile(w2.ravel(), N_FIELDS)

@dataclass
class ProfileStateTE1:
    # X: (6, rank, nx), Y: (6, rank, ns)
    X: np.ndarray
    Y: np.ndarray

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        self.Y = np.asarray(self.Y, float)
        if self.X.ndim != 3 or self.Y.ndim != 3:
            raise ValueError("X/Y must be 3D")
        if self.X.shape[0] != 6 or self.Y.shape[0] != 6:
            raise ValueError("six TE1 fields required")
        if self.X.shape[1] != self.Y.shape[1] or self.X.shape[1] < 1:
            raise ValueError("invalid rank")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.Y)):
            raise ValueError("non-finite profiles")

    @property
    def rank(self):
        return self.X.shape[1]

    def copy(self):
        return ProfileStateTE1(self.X.copy(), self.Y.copy())

    def validate_grid(self, grid):
        if self.X.shape[2] != grid.nx or self.Y.shape[2] != grid.ns:
            raise ValueError("profile/grid mismatch")

def reconstruct_fields(state: ProfileStateTE1):
    # field-major, arrays shaped (6, ns, nx)
    U = np.empty((6, state.Y.shape[2], state.X.shape[2]))
    for f in range(6):
        U[f] = state.Y[f].T @ state.X[f]
    return U

def reconstruct_vector(state: ProfileStateTE1, grid: CylinderGrid):
    state.validate_grid(grid)
    return reconstruct_fields(state).reshape(-1)

def _active_ids(state, direction, active_mask=None):
    d = direction.upper()
    arr = state.X if d == "X" else state.Y if d == "Y" else None
    if arr is None:
        raise ValueError("direction must be X or Y")
    if active_mask is None:
        mask = np.ones_like(arr, bool)
    else:
        mask = np.asarray(active_mask, bool)
        if mask.shape != arr.shape:
            raise ValueError("active_mask shape mismatch")
    ids = np.flatnonzero(mask.ravel())
    if ids.size == 0:
        raise ReducedSystemError(f"{d}_HALFSTEP_HAS_NO_ACTIVE_PROFILE_DIRECTIONS")
    return ids

def get_active_parameters(state, direction, active_mask=None):
    ids = _active_ids(state, direction, active_mask)
    arr = state.X if direction.upper() == "X" else state.Y
    return arr.ravel()[ids].copy(), ids

def set_active_parameters(state, direction, ids, values):
    arr = state.X if direction.upper() == "X" else state.Y
    ids = np.asarray(ids, int)
    values = np.asarray(values, float)
    if values.shape != ids.shape:
        raise ValueError("values/ids shape mismatch")
    # ndarray.flat writes through to the original array even when the
    # profile storage is a non-C-contiguous view.  ravel() may return a
    # copy in that case and silently lose the MVI update.
    arr.flat[ids] = values

def _decode(flat_idx, rank, n_dir):
    fr, node = divmod(int(flat_idx), int(n_dir))
    field, r = divmod(fr, int(rank))
    return field, r, node

def build_direction_tangent(state, grid, direction, active_mask=None):
    """
    Exact algebraic T=dU/dc for the separated representation.
    No profile transformations are performed.
    """
    state.validate_grid(grid)
    d = direction.upper()
    ids = _active_ids(state, d, active_mask)
    T = np.zeros((grid.n_full_dofs, len(ids)))
    n2 = grid.n_surface_nodes
    n_dir = grid.nx if d == "X" else grid.ns

    for col, idx in enumerate(ids):
        f, r, node = _decode(idx, state.rank, n_dir)
        base = f * n2
        if d == "Y":
            is0 = node
            row0 = base + is0 * grid.nx
            T[row0:row0 + grid.nx, col] = state.X[f, r]
        else:
            ix0 = node
            rows = base + np.arange(grid.ns) * grid.nx + ix0
            T[rows, col] = state.Y[f, r]
    return T, ids

def project_reduced_system(residual, tangent, T, row_weights):
    """
    r = T^T W R
    A = T^T W K_T T
    W is applied exactly once here.
    """
    R = np.asarray(residual, float)
    K = np.asarray(tangent, float)
    T = np.asarray(T, float)
    w = np.asarray(row_weights, float)
    n = T.shape[0]
    if R.shape != (n,) or K.shape != (n, n) or w.shape != (n,):
        raise KernelContractError("R/K/W shape mismatch")
    r = T.T @ (w * R)
    A = T.T @ (w[:, None] * (K @ T))
    return r, A

def direct_reduced_solve(A, r):
    try:
        dc = np.linalg.solve(np.asarray(A, float), -np.asarray(r, float))
    except np.linalg.LinAlgError as exc:
        raise ReducedSystemError("SINGULAR_REDUCED_MVI_SYSTEM") from exc
    if not np.all(np.isfinite(dc)):
        raise ReducedSystemError("NONFINITE_REDUCED_MVI_CORRECTION")
    return dc

Kernel = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

@dataclass(frozen=True)
class HalfstepResult:
    direction: str
    converged: bool
    iterations: int
    projected_norm: float
    correction_norm: float
    status: str

def solve_halfstep(state: ProfileStateTE1, grid: CylinderGrid, direction: str,
                   kernel: Kernel, *, active_mask: Optional[np.ndarray]=None,
                   row_weights: Optional[np.ndarray]=None,
                   projected_tol=1e-10, correction_tol=1e-12,
                   max_iterations=50):
    """
    Frozen-partner halfstep. The supplied physical kernel must return
    an UNWEIGHTED full residual R(U) and full tangent K_T(U).

    There is no hidden full-grid correction and no fallback linear solver.
    """
    d = direction.upper()
    weights = grid.residual_projection_weights() if row_weights is None else np.asarray(row_weights, float)
    last_r = float("inf")
    last_dc = float("inf")

    for it in range(1, max_iterations + 1):
        U = reconstruct_vector(state, grid)
        R, K = kernel(U)
        T, ids = build_direction_tangent(state, grid, d, active_mask)
        r, A = project_reduced_system(R, K, T, weights)
        last_r = float(np.linalg.norm(r))

        if last_r <= projected_tol:
            return HalfstepResult(d, True, it-1, last_r, 0.0, "CONVERGED_REDUCED_MVI")

        dc = direct_reduced_solve(A, r)
        last_dc = float(np.linalg.norm(dc))
        c, ids2 = get_active_parameters(state, d, active_mask)
        if not np.array_equal(ids, ids2):
            raise RuntimeError("active parameter ordering changed")
        set_active_parameters(state, d, ids, c + dc)

        if last_dc <= correction_tol:
            U = reconstruct_vector(state, grid)
            R, K = kernel(U)
            T, _ = build_direction_tangent(state, grid, d, active_mask)
            r, _ = project_reduced_system(R, K, T, weights)
            last_r = float(np.linalg.norm(r))
            if last_r <= projected_tol:
                return HalfstepResult(d, True, it, last_r, last_dc, "CONVERGED_REDUCED_MVI")

    # The last permitted Newton correction has changed ``state``.  Check that
    # corrected state once before reporting exhaustion of the iteration limit.
    U = reconstruct_vector(state, grid)
    R, K = kernel(U)
    T, _ = build_direction_tangent(state, grid, d, active_mask)
    r, _ = project_reduced_system(R, K, T, weights)
    last_r = float(np.linalg.norm(r))
    if last_r <= projected_tol:
        return HalfstepResult(
            d, True, max_iterations, last_r, last_dc,
            "CONVERGED_REDUCED_MVI"
        )
    return HalfstepResult(d, False, max_iterations, last_r, last_dc, "MAX_HALFSTEP_ITERATIONS")
