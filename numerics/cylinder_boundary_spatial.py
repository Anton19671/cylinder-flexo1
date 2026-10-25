#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constrained reference spatial assembly of the common shell functional.

FDM uses all nodal directions. An MVI X/Y half-step uses only directions of
the frozen-partner separated representation (ranks 1..8). In both cases the
equations are B.T R = 0, where C B = 0. Prescribed traces are imposed through
an affine lift, never through residual-row replacement or a penalty. The
raw residual retains support/electrode reactions.

This dense verification API connects boundary work to spatial equations;
it is NOT the production CASE dispatcher or a complete MVI/Newmark driver.
Callers must solve the projected equations and check the full admissible
residual separately from an individual MVI half-step residual. SI units and
field ordering are inherited from CylinderFunctional.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from cylinder_functional_assembly import _basis_moments
from cylinder_variational_boundary import combine_functional_and_boundary


def _array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected finite shape {shape}")
    return a


@dataclass(frozen=True)
class AffineTraceSpace:
    """u = origin + basis*z; columns of basis are admissible variations."""
    origin: np.ndarray
    basis: np.ndarray
    matrix: np.ndarray
    target: np.ndarray
    row_scale: np.ndarray
    labels: tuple

    def defect(self, state):
        u = _array(state, self.origin.shape, "state")
        # Row-normalized traces have displacement/potential scale, avoiding
        # arbitrary dependence on the units of D_x versus D_xx rows.
        return self.matrix @ u - self.target

    def project(self, state):
        """Closest state in this affine space in Euclidean nodal norm."""
        u = _array(state, self.origin.shape, "state")
        # basis need not be orthonormal for arbitrary MVI directions.
        if not self.basis.shape[1]:
            return self.origin.copy()
        z = np.linalg.solve(self.basis.T @ self.basis,
                            self.basis.T @ (u-self.origin))
        return self.origin + self.basis @ z

    def residual(self, full_residual):
        return self.basis.T @ _array(full_residual, self.origin.shape, "residual")

    def tangent(self, tangent_action):
        """Exact pullback B.T K B for a caller-supplied tangent action."""
        if not callable(tangent_action):
            raise TypeError("tangent_action must be callable")
        if not self.basis.shape[1]:
            return np.empty((0, 0))
        kb = np.column_stack([
            _array(tangent_action(b), self.origin.shape, "tangent action")
            for b in self.basis.T])
        return self.basis.T @ kb


def affine_trace_space(matrix, target, base, *, directions=None, labels=(),
                       rank_tolerance=1.e-12):
    """Eliminate general (including dependent) linear trace equations C u=d.

    With directions T this imposes C(base+T*c)=d in the selected half-step.
    Inconsistent prescriptions and unreachable inhomogeneous traces raise;
    no fallback to a full-grid correction is performed for MVI.
    """
    u = np.asarray(base, dtype=float)
    if u.ndim != 1 or not u.size or not np.isfinite(u).all():
        raise ValueError("base must be a finite nonempty vector")
    c = np.asarray(matrix, dtype=float)
    if c.ndim != 2 or c.shape[1] != u.size or not np.isfinite(c).all():
        raise ValueError("constraint matrix shape/nonfinite")
    d = _array(target, (c.shape[0],), "constraint target")
    if labels and len(labels) != c.shape[0]:
        raise ValueError("constraint labels shape")
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError("rank_tolerance must be between zero and one")
    if max(u.size, c.shape[0]) > 4096:
        raise ValueError("dense reference trace assembly limited to 4096 unknowns/rows")
    t = np.eye(u.size) if directions is None else np.asarray(directions, dtype=float)
    if t.ndim != 2 or t.shape[0] != u.size or not np.isfinite(t).all():
        raise ValueError("spatial directions shape/nonfinite")
    # This is deliberately a dense reference implementation, not suitable
    # for silently replacing large-grid production MVI/FDM operators.
    if max(u.size, t.shape[1], c.shape[0]) > 4096:
        raise ValueError("dense reference trace assembly limited to 4096 unknowns/rows")
    if t.shape[1]:
        sv = np.linalg.svd(t, compute_uv=False)
        if t.shape[1] > t.shape[0] or sv[-1] <= rank_tolerance*sv[0]:
            raise ValueError("dependent spatial directions; normalize frozen MVI partners")
    scale = np.linalg.norm(c, axis=1)
    if np.any(scale == 0):
        raise ValueError("zero boundary constraint row")
    c = c / scale[:, None]
    d = d / scale
    if not len(d):
        return AffineTraceSpace(u.copy(), t.copy(), c, d, scale, tuple(labels))
    a = c @ t
    rhs = d-c@u
    if not t.shape[1]:
        lift = np.empty(0)
        z = np.empty((0, 0))
    else:
        # SVD here only eliminates linear BC constraints; it never truncates
        # a physical MVI state, adds regularization or solves shell equilibrium.
        left, singular, right = np.linalg.svd(a, full_matrices=True)
        cutoff = rank_tolerance*sv[0]
        rank = int(np.count_nonzero(singular > cutoff))
        lift = right[:rank].T @ ((left[:, :rank].T @ rhs)/singular[:rank])
        z = right[rank:].T
    mismatch = a@lift-rhs
    # Use each row's uncancelled arithmetic scale. An exactly admissible
    # electrode can have C*u ~ roundoff although its individual potentials
    # are nonzero; scaling by |C*u| incorrectly rejects such MVI half-steps.
    tol = 1.e-10*np.maximum(1.e-12, abs(d)+abs(c)@abs(u)+abs(a)@abs(lift))
    # A dense SVD may mix independent zero-target blocks at roundoff level.
    # Include its backward-error floor as well as the physical row scale.
    tol += 20*np.finfo(float).eps*max(a.shape)*float(np.max(abs(rhs), initial=0))
    # A previous constrained solve or an exact profile QR coordinate change
    # can leave roundoff at a zero prescribed trace. Its backward-error scale
    # is the whole represented state, not only the tiny constrained entries.
    tol += 20*np.finfo(float).eps*max(a.shape)*float(np.max(abs(u), initial=0))
    if np.any(abs(mismatch) > tol):
        raise ValueError("inconsistent/unreachable prescribed traces in spatial subspace")
    return AffineTraceSpace(u+t@lift, t@z, c, d, scale, tuple(labels))


def separated_profile_directions(x, y, direction):
    """Nodal variations of sum_r X[c,r,x]*Y[c,r,s], c = q followed by phi.

    The partner family stays frozen throughout one half-step. A lift for
    inhomogeneous traces should be carried separately from these profiles.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not 1 <= x.shape[1] <= 8 or x.shape[0] < 1
            or min(x.shape[2], y.shape[2]) < 1):
        raise ValueError("profiles must be finite (components, rank=1..8, nodes)")
    direction = str(direction).upper()
    if direction not in ("X", "Y"):
        raise ValueError("MVI half-step must be X or Y")
    nc, rank, nx = x.shape
    ns = y.shape[2]
    count = nx if direction == "X" else ns
    t = np.zeros((nc*nx*ns, nc*rank*count))
    for c in range(nc):
        for r in range(rank):
            for j in range(count):
                field = np.zeros((nc, nx, ns))
                if direction == "X":
                    field[c, j, :] = y[c, r]
                else:
                    field[c, :, j] = x[c, r]
                t[:, (c*rank+r)*count+j] = field.ravel()
    return t


@dataclass(frozen=True)
class SpatialBoundaryAssembly:
    potential_j: float
    full_residual: np.ndarray
    state: np.ndarray
    edge: object
    space: AffineTraceSpace
    method: str
    time_s: float | None

    @property
    def projected_residual(self):
        return self.space.residual(self.full_residual)

    @property
    def constraint_max_defect(self):
        return float(np.max(abs(self.space.defect(self.state)), initial=0))


class BoundarySpatialProblem:
    """Bulk + end/face work + admissible FDM/MVI directions at a physical time."""
    def __init__(self, functional, *, bc_spec=None, calculation_kind="STATIC"):
        self.functional = functional
        self.spec = copy.deepcopy({} if bc_spec is None else bc_spec)
        self.kind = str(calculation_kind).strip().upper()
        if self.kind not in ("STATIC", "DYNAMIC"):
            raise ValueError("calculation_kind must be STATIC/DYNAMIC")

    def assemble(self, mechanical, electrical=None, *, method="FDM", directions=None,
                 dead_load=None, free_charge=None, time_s=None,
                 acceleration=None, density_kg_m3=None):
        method = str(method).upper()
        if method not in ("FDM", "MVI"):
            raise ValueError("method must be FDM/MVI")
        if (method == "MVI") != (directions is not None):
            raise ValueError("MVI requires frozen-partner directions; FDM uses all nodes")
        if self.kind == "STATIC" and acceleration is not None:
            raise ValueError("STATIC cannot include inertia")
        if (acceleration is None) != (density_kg_m3 is None):
            raise ValueError("acceleration and density_kg_m3 must be supplied together")
        energy, rm, re, edge = combine_functional_and_boundary(
            self.functional, mechanical, electrical, bc_spec=self.spec,
            dead_load=dead_load, free_charge=free_charge,
            calculation_kind=self.kind, time_s=time_s)
        if acceleration is not None:
            # Electric coefficients remain quasi-static; they acquire no mass.
            # The functional explicitly rejects unsupported nonlinear KL mass.
            rm = rm+self.functional.mass_action(acceleration, density_kg_m3=density_kg_m3)
        q = np.asarray(mechanical, float)
        state = q.ravel() if re is None else np.concatenate((q.ravel(), np.asarray(electrical).ravel()))
        residual = rm.ravel() if re is None else np.concatenate((rm.ravel(), re.ravel()))
        constraints = list(edge.constraints)
        rows, values, labels = [], [], []
        for row in constraints:
            c = np.zeros(state.size)
            offset = 0 if row.field == "mechanical" else q.size
            c[np.asarray(row.indices)+offset] = row.weights
            rows.append(c)
            values.append(row.prescribed)
            labels.append(row.label)
        if edge.gauge_required:
            p = np.asarray(electrical)
            moments = _basis_moments(len(p)-1, self.functional.R, self.functional.h)
            gauge = moments[:, None, None]*self.functional.W[None, ...]
            rows.append(np.concatenate((np.zeros(q.size), gauge.ravel())))
            values.append(0.)
            labels.append("electric volume-mean gauge")
        c = np.asarray(rows).reshape((-1, state.size))
        space = affine_trace_space(c, np.asarray(values), state,
                                   directions=directions, labels=labels)
        return SpatialBoundaryAssembly(energy, residual, state, edge, space, method, time_s)


__all__ = ("AffineTraceSpace", "affine_trace_space", "separated_profile_directions",
           "SpatialBoundaryAssembly", "BoundarySpatialProblem")
