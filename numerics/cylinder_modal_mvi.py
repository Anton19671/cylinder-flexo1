#!/usr/bin/env python3
"""Fixed-rank alternating variational eigenmodes, not an FDM eigensolve/SVD fit.

Each physical field u_c(x,s) is sum_r X[c,r,x]*Y[c,r,s].
X and Y are updated in turn by a generalized Rayleigh--Ritz eigenproblem
in the frozen-partner constrained variation space. Previously calculated
modes are mass-orthogonalized within each half-step. Electrics, when present,
have no artificial inertia and are eliminated by the reduced stationary
Schur complement. All shell kinematics/inertia are the existing ones.
This is a small-grid reference, not a large-grid production solver.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, eigh, qr, solve

from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
from cylinder_variational_fdm import state_scales
from cylinder_variational_mvi import _normalize_partner, reconstruct_profiles

MAX_PROJECTED_DIRECTIONS = 1600


def _mass_orthogonal_null(matrix):
    """QR null basis of preceding modal mass constraints, without SVD."""
    c = np.asarray(matrix, float)
    if c.ndim != 2:
        raise ValueError("invalid modal orthogonality matrix")
    if not c.shape[0]:
        return np.eye(c.shape[1])
    q, r, _ = qr(c.T, mode="full", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    cutoff = 1.e-11 * max(float(np.max(diagonal, initial=0.)), 1.e-30)
    rank = int(np.sum(diagonal > cutoff))
    return q[:, rank:]


def _initial_factors(components, rank, nx, ns, mode):
    """Reproducible factors, not precomputed FDM mode shapes."""
    rng = np.random.default_rng(94031 + 101 * mode + 37 * rank)
    x = np.empty((components, rank, nx), float)
    y = np.empty((components, rank, ns), float)
    for c in range(components):
        x[c] = np.linalg.qr(rng.normal(size=(nx, rank)), mode="reduced")[0].T
        y[c] = np.linalg.qr(rng.normal(size=(ns, rank)), mode="reduced")[0].T
    return x, y


class _SeparatedModalProblem:
    def __init__(self, prepared, density, rank):
        self.prepared = prepared
        self.f = prepared.problem.functional
        self.q0, self.p0 = prepared.mechanical, prepared.electrical
        self.nm = self.q0.size
        self.ne = 0 if self.p0 is None else self.p0.size
        self.shape = self.q0.shape
        self.n_components = self.shape[0] + (0 if self.p0 is None else self.p0.shape[0])
        self.nx, self.ns = self.shape[1:]
        self.rank = rank
        if not 1 <= rank <= 8 or rank > min(self.nx, self.ns):
            raise ValueError("MODAL MVI rank must be 1..8 and <= min(nx, ns)")
        count = self.n_components * rank * max(self.nx, self.ns)
        if count > MAX_PROJECTED_DIRECTIONS:
            raise ValueError("MODAL MVI small-grid limit: projected X/Y family has "
                             f"{count} directions, maximum {MAX_PROJECTED_DIRECTIONS}")
        first = prepared.problem.assemble(self.q0, self.p0)
        if np.max(np.abs(first.space.origin), initial=0.) > 1.e-10:
            raise ValueError("MODAL MVI needs homogeneous essential boundary conditions")
        c, target = first.space.matrix, first.space.target
        if np.max(np.abs(target), initial=0.) > 1.e-12:
            raise ValueError("MODAL MVI cannot prescribe nonzero supports")
        has_m = np.any(c[:, :self.nm] != 0., axis=1)
        has_e = np.any(c[:, self.nm:] != 0., axis=1)
        if np.any(has_m & has_e) or np.any(~(has_m | has_e)):
            raise ValueError("MODAL MVI cannot separate mixed mechanical/electric constraints")
        self.cm = c[has_m, :self.nm]
        self.ce = c[has_e, self.nm:]
        self.step = prepared.settings.tangent_step_scaled
        self.scales = state_scales(self.f, self.q0, self.p0, prepared.settings)
        self.velocity0 = np.zeros_like(self.q0)
        self.density = density

    def mass(self, direction):
        q = np.asarray(direction, float).reshape(self.shape)
        return self.f.mass_action(q, density_kg_m3=self.density,
                                  mechanical=self.q0, velocity=self.velocity0).ravel()

    def stiffness(self, direction):
        """Tangent action via the unchanged discrete first variation."""
        direction = np.asarray(direction, float)
        if direction.shape != (self.nm + self.ne,):
            raise ValueError("MODAL MVI tangent vector shape")
        def force(sign):
            state = sign * self.step * direction
            q = state[:self.nm].reshape(self.shape)
            p = None if not self.ne else state[self.nm:].reshape(self.p0.shape)
            _, residual, _ = self.prepared.problem.evaluate_effective_residual(q, p)
            return residual
        return (force(1.) - force(-1.)) / (2. * self.step)

    def projected_bases(self, x, y, axis):
        """Constrain only the actual frozen-partner X/Y test directions."""
        ncm = self.shape[0]
        dm = separated_profile_directions(x[:ncm], y[:ncm], axis)
        dm = self.scales[:self.nm, None] * dm
        bm = affine_trace_space(self.cm, np.zeros(len(self.cm)),
                                np.zeros(self.nm), directions=dm).basis
        if not bm.shape[1]:
            raise ValueError("MODAL MVI: no admissible mechanical directions in " + axis)
        if not self.ne:
            return bm, np.empty((0, 0))
        de = separated_profile_directions(x[ncm:], y[ncm:], axis)
        de = self.scales[self.nm:, None] * de
        be = affine_trace_space(self.ce, np.zeros(len(self.ce)),
                                np.zeros(self.ne), directions=de).basis
        return bm, be

    def halfstep(self, x, y, axis, previous, *, on_stage=None):
        """Rayleigh--Ritz on one frozen-factor test space; no FDM eigenmodes."""
        bm, be = self.projected_bases(x, y, axis)
        nm, ne = bm.shape[1], be.shape[1]
        b = np.zeros((self.nm + self.ne, nm + ne))
        b[:self.nm, :nm] = bm
        if ne:
            b[self.nm:, nm:] = be
        k = np.empty((nm + ne, nm + ne))
        if on_stage is not None:on_stage("K_BEGIN",0,nm+ne)
        for j in range(nm + ne):
            if on_stage is not None:on_stage("K_COLUMN_START",j+1,nm+ne)
            k[:, j] = b.T @ self.stiffness(b[:, j])
            if on_stage is not None:on_stage("K_COLUMN_DONE",j+1,nm+ne)
        if not np.isfinite(k).all():
            raise ValueError("nonfinite MODAL MVI tangent")
        asym = float(np.linalg.norm(k - k.T, ord=np.inf)) / max(
            float(np.linalg.norm(k, ord=np.inf)), 1.e-30)
        if asym > 1.e-5:
            raise ValueError("MODAL MVI tangent is not symmetric")
        k = 0.5 * (k + k.T)
        m = np.empty((nm, nm))
        if on_stage is not None:on_stage("M_BEGIN",0,nm)
        for j in range(nm):
            if on_stage is not None:on_stage("M_COLUMN_START",j+1,nm)
            m[:, j] = bm.T @ self.mass(bm[:, j])
            if on_stage is not None:on_stage("M_COLUMN_DONE",j+1,nm)
        if not np.isfinite(m).all():
            raise ValueError("nonfinite MODAL MVI mass matrix")
        if np.linalg.norm(m - m.T, ord=np.inf) > (
                1.e-9 * max(float(np.linalg.norm(m, ord=np.inf)), 1.e-30)):
            raise ValueError("MODAL MVI mass matrix is not symmetric")
        m = 0.5 * (m + m.T)
        cholesky(m, lower=True, check_finite=True)
        keff = k[:nm, :nm].copy()
        electric_recovery = None
        if ne:
            electric_recovery = -solve(k[nm:, nm:], k[nm:, :nm],
                                      assume_a="sym", check_finite=True)
            keff += k[:nm, nm:] @ electric_recovery
            keff = 0.5 * (keff + keff.T)
        constraints = (np.stack([bm.T @ self.mass(q) for q in previous])
                       if previous else np.zeros((0, nm)))
        z = _mass_orthogonal_null(constraints)
        if not z.shape[1]:
            raise ValueError("MODAL MVI rank insufficient for additional orthogonal modes")
        kz, mz = z.T @ keff @ z, z.T @ m @ z
        cholesky(mz, lower=True, check_finite=True)
        if on_stage is not None:on_stage("EIGEN_BEGIN",0,1)
        eigenvalues, eigvec = eigh(kz, mz, subset_by_index=(0, 0),
                                  driver="gvx", check_finite=True)
        value = float(eigenvalues[0])
        if on_stage is not None:on_stage("EIGEN_DONE",1,1)
        if value < -1.e-8 * max(1., abs(value)):
            raise ValueError("MODAL MVI: negative squared frequency")
        coeff = z @ eigvec[:, 0]
        ritz_m = m @ coeff
        ritz_k = keff @ coeff
        ritz_relative = float(np.linalg.norm(ritz_k - value*ritz_m) /
            max(np.linalg.norm(ritz_k) + abs(value)*np.linalg.norm(ritz_m), 1.e-30))
        q = bm @ coeff
        p = (None if not self.ne else
             np.zeros(self.ne) if electric_recovery is None else
             be @ (electric_recovery @ coeff))
        return q, p, max(0., value), ritz_relative

    def fit(self, x, y, axis, q, p):
        """Exact rank-preserving X/Y factor recovery, no nodal lift."""
        field = q if p is None else np.concatenate((q, p))
        scaled = (field / self.scales).reshape(self.n_components, self.nx, self.ns)
        if axis == "X":
            x = np.einsum("cxs,crs->crx", scaled, y)
        else:
            y = np.einsum("cxs,crx->crs", scaled, x)
        fit = reconstruct_profiles(x, y)
        if np.linalg.norm(fit - scaled) > 1.e-8 * max(
                float(np.linalg.norm(scaled)), 1.e-30):
            raise ValueError("MODAL MVI separated factor reconstruction failed")
        return x, y

    def eigenvalue(self, q, p):
        field = q if p is None else np.concatenate((q, p))
        denominator = float(q @ self.mass(q))
        if denominator <= 0.:
            raise ValueError("MODAL MVI: nonpositive modal mass")
        return float(field @ self.stiffness(field) / denominator)


def solve_modal_mvi(prepared, density, requested, *, progress_callback=None,
                    sweep_callback=None, detail_callback=None, eigen_tolerance=1.e-7,
                    mode_tolerance=1.e-5):
    """Alternating constrained generalized eigenproblems at one fixed rank.

    Not converged => explicit error, never an FDM fallback or fabricated
    MODAL_COMPLETED output.
    """
    if prepared.method != "MVI" or prepared.mvi_settings is None:
        raise ValueError("MODAL MVI needs explicit MVI CASE and rank")
    rank = prepared.mvi_settings.rank
    max_sweeps = prepared.mvi_settings.max_sweeps
    operator = _SeparatedModalProblem(prepared, density, rank)
    qmodes, pmodes, eigs, cycles = [], [], [], []
    for mode in range(requested):
        x, y = _initial_factors(operator.n_components, rank,
                                operator.nx, operator.ns, mode)
        last_lambda, last_q = None, None
        converged = False
        for sweep in range(1, max_sweeps + 1):
            if detail_callback is not None:
                detail_callback(stage="MVI_SWEEP_START",mode=mode+1,
                    n_modes=requested,rank=rank,sweep=sweep,max_sweeps=max_sweeps)
            # X comes first: otherwise an initial non-clamped X factor can
            # make the clamped Y test-space identically zero.
            for axis in ("X", "Y"):
                if detail_callback is not None:
                    detail_callback(stage="MVI_HALFSTEP_START",mode=mode+1,
                        n_modes=requested,rank=rank,sweep=sweep,
                        max_sweeps=max_sweeps,axis=axis)
                x, y = _normalize_partner(x, y, axis)
                def trace(stage, column, total):
                    if detail_callback is not None:
                        detail_callback(stage=stage,mode=mode+1,n_modes=requested,
                            rank=rank,sweep=sweep,max_sweeps=max_sweeps,
                            axis=axis,column=column,total_columns=total)
                q, p, _, ritz_relative = operator.halfstep(
                    x, y, axis, qmodes, on_stage=trace)
                x, y = operator.fit(x, y, axis, q, p)
                if detail_callback is not None:
                    detail_callback(stage="MVI_HALFSTEP_DONE",mode=mode+1,
                        n_modes=requested,rank=rank,sweep=sweep,axis=axis,
                        ritz_relative=ritz_relative)
            value = operator.eigenvalue(q, p)
            if not np.isfinite(value) or value < 0.:
                raise ValueError("MODAL MVI: invalid Rayleigh quotient")
            relative, change = float("inf"), float("inf")
            if last_q is not None:
                overlap = float(last_q @ operator.mass(q))
                if overlap < 0.:
                    q = -q
                    p = None if p is None else -p
                    x = -x
                    overlap = -overlap
                change = np.sqrt(max(0., 2. - 2. * min(1., overlap)))
                relative = abs(value - last_lambda) / max(
                    1., abs(value), abs(last_lambda))
            if sweep_callback is not None:
                sweep_callback(mode + 1, requested, sweep, max_sweeps, value)
            if detail_callback is not None:
                detail_callback(stage="MVI_SWEEP_DONE",mode=mode+1,
                    n_modes=requested,rank=rank,sweep=sweep,max_sweeps=max_sweeps,
                    frequency_hz=float(np.sqrt(value)/(2.0*np.pi)),
                    omega_rad_s=float(np.sqrt(value)),
                    modal_amplitude_max=float(np.max(abs(q),initial=0.)),
                    frequency_squared_change_relative=(None if last_lambda is None else relative),
                    shape_change_relative=(None if last_q is None else float(change)),
                    ritz_relative=ritz_relative)
            if relative <= eigen_tolerance and change <= mode_tolerance:
                converged = True
                break
            last_lambda, last_q = value, q.copy()
        if not converged:
            raise ValueError(
                f"MODAL_MVI_MODE_NOT_CONVERGED: rank={rank}, mode={mode + 1}, "
                f"max_sweeps={max_sweeps}; adjust grid/rank or increase "
                "reference_mvi.max_sweeps. No FDM fallback.")
        modal_mass = np.sqrt(float(q @ operator.mass(q)))
        q = q / modal_mass
        if p is not None:
            p = p / modal_mass
        qmodes.append(q.copy())
        pmodes.append(None if p is None else p.copy())
        eigs.append(value)
        cycles.append(sweep)
        if detail_callback is not None:
            detail_callback(stage="MVI_MODE_DONE",mode=mode+1,
                n_modes=requested,rank=rank,sweep=sweep,
                frequency_hz=float(np.sqrt(value)/(2.0*np.pi)),
                omega_rad_s=float(np.sqrt(value)),
                modal_amplitude_max=float(np.max(abs(q),initial=0.)),
                ritz_relative=ritz_relative)
        if progress_callback is not None:
            progress_callback(mode + 1, requested)
    order = np.argsort(eigs)
    eigenvalues = np.asarray(eigs)[order]
    q_out = np.stack([qmodes[i].reshape(operator.shape) for i in order])
    p_out = (None if not operator.ne else np.stack(
        [pmodes[i].reshape(operator.p0.shape) for i in order]))
    omega = np.sqrt(np.maximum(eigenvalues, 0.))
    return {
        "theory": operator.f.theory,
        "method": "MVI", "rank": rank, "density_kg_m3": density,
        "frequencies_hz": omega / (2. * np.pi),
        "omega_rad_s": omega, "omega_squared_rad2_s2": eigenvalues,
        "mechanical_modes": q_out, "electrical_modes": p_out,
        "n_mechanical_free_dofs": operator.nm,
        "n_electric_free_dofs": operator.ne,
        "mvi_sweeps": [cycles[i] for i in order],
        "mvi_modal_algorithm": "fixed_rank_alternating_XY_mass_orthogonal_Rayleigh_Ritz",
    }


__all__ = ("MAX_PROJECTED_DIRECTIONS", "solve_modal_mvi")
