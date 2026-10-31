#!/usr/bin/env python3
"""Fixed-rank spatial variational iterations for shell eigenmodes.

Each mechanical and (optional) electrical coefficient has independent
factors sum_r X[c,r,x]*Y[c,r,s]. At fixed partner, solve the constrained
stationarity equations of the modal functional in that X or Y variation
space, then alternate Y -> X as in the static Krysko/Kirichenko solver.

The stationary modal equations are generalized eigen-equations, solved only
on the current separated-factor test space, NOT on the full FDM space.
Electric coefficients are massless and eliminated via stationarity.
All physical shell operators and the existing STATIC/DYNAMIC MVI are unchanged.
This is a small-grid verification implementation, not a certified benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, eigh, qr, solve, solve_triangular

from cylinder_boundary_spatial import affine_trace_space, separated_profile_directions
from cylinder_variational_fdm import state_scales
from cylinder_variational_mvi import _normalize_partner, reconstruct_profiles
from cylinder_modal_harmonic_spectrum import (
    _harmonic_phase_basis, _harmonic_variation_directions,
    _phase_quadrature_partner, _pure_harmonic_fraction,
    _solve_modal_mvi_harmonic_spectrum,
)

MAX_PROJECTED_DIRECTIONS = 1600


def _free_dof_count(constraints):
    """Count unconstrained nodal coefficients without forming a nodal basis."""
    c = np.asarray(constraints, float)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("invalid modal trace matrix")
    if not c.shape[0]:
        return c.shape[1]
    row_scale = np.max(np.abs(c), axis=1, initial=0.)
    c = c[row_scale > 0.] / row_scale[row_scale > 0., None]
    if not c.shape[0]:
        return c.shape[1]
    c /= np.linalg.norm(c, axis=1)[:, None]
    _, r, _ = qr(c.T, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(
        diagonal > 1.e-12 * np.max(diagonal, initial=0.)))
    return c.shape[1] - rank


def _mass_orthogonal_null(matrix):
    """QR null basis of preceding modal mass constraints, without SVD."""
    c = np.asarray(matrix, float)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("invalid modal orthogonality matrix")
    if not c.shape[0]:
        return np.eye(c.shape[1])
    # Each row is a homogeneous orthogonality equation. Its units or
    # magnitude must not decide whether an independent mode is excluded.
    # First scale by the largest entry to avoid overflow/underflow, then
    # give every nonzero row unit norm before the relative QR rank test.
    # Negligible physical overlaps are filtered in the mass norm by the
    # caller, never by comparing the sizes of different constraint rows.
    row_scale = np.max(np.abs(c), axis=1, initial=0.)
    c = c[row_scale > 0.] / row_scale[row_scale > 0., None]
    if not c.shape[0]:
        return np.eye(c.shape[1])
    c = c / np.linalg.norm(c, axis=1)[:, None]
    q, r, _ = qr(c.T, mode="full", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    cutoff = 1.e-11 * max(float(np.max(diagonal, initial=0.)), 1.e-30)
    rank = int(np.sum(diagonal > cutoff))
    return q[:, rank:]


def _significant_modal_constraints(constraints, mass_factor):
    """Keep rows which can exceed the mass-overlap tolerance in this space."""
    if not len(constraints):
        return constraints
    max_overlaps = np.linalg.norm(solve_triangular(
        mass_factor, constraints.T, lower=True, check_finite=False), axis=0)
    # Final independently checked orthogonality tolerance is 1e-6. A row
    # bounded below this stricter limit cannot violate it for any unit mode.
    return constraints[max_overlaps > 2.e-7]


def _modal_variational_step_precision(old_projected, new_projected,
                                      old_physical, new_physical):
    """Check thin-shell stationarity against independent physical field energy.

    Projected stiffness entries and independently evaluated field energy can
    differ slightly for a very thin CUF shell. A small apparent increase in
    the projected Rayleigh quotient is numerical only when the physical
    quotient does not increase and the increase is bounded by the observed
    matrix-versus-field disagreement. Reject any gross disagreement.
    This check never modifies eigenvalues, eigenvectors, K, M or MVI rank.
    """
    values = np.asarray((old_projected, new_projected,
                         old_physical, new_physical), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("MODAL_MVI_NONFINITE_VARIATIONAL_ENERGY")
    tolerance = 1.e-10 * max(1., abs(old_projected), abs(new_projected))
    physical_tolerance = 1.e-10 * max(1., abs(old_physical), abs(new_physical))
    observed_mismatch = (abs(old_physical-old_projected) +
                         abs(new_physical-new_projected))
    increased = new_projected-old_projected
    if increased > tolerance:
        scale = max(1., *(abs(float(v)) for v in values))
        # A large discrepancy cannot be classified as roundoff even if
        # two independently measured physical values happen to match.
        if (observed_mismatch > 1.e-6 * scale or
                increased > observed_mismatch + tolerance or
                new_physical > old_physical + physical_tolerance):
            raise ValueError(
                "MODAL_MVI_PROJECTED_VARIATIONAL_INCREASE: "
                f"lambda_old={old_projected:.12g}, "
                f"lambda_new={new_projected:.12g}, "
                f"physical_old={old_physical:.12g}, "
                f"physical_new={new_physical:.12g}")
        return True, observed_mismatch
    return False, observed_mismatch


def _modal_ritz_value(kz, mz, candidate):
    """Evaluate a computed modal vector against the ORIGINAL projected K/M.

    The low LAPACK eigenvalue of a very thin CUF shell can lose significant
    absolute precision in a pencil with a huge spectral range. Its Rayleigh
    quotient is a more faithful energy measure than the eigenvalue reported
    by the generalized eigensolver. This evaluates rather than adjusts the
    physical eigenvector, and it does not alter K, M or the MVI subspace.
    """
    candidate = np.asarray(candidate, dtype=float)
    numerator = float(candidate @ kz @ candidate)
    denominator = float(candidate @ mz @ candidate)
    if (not np.isfinite(numerator) or not np.isfinite(denominator)
            or denominator <= 0.):
        raise ValueError("MODAL_MVI_INVALID_RITZ_QUOTIENT")
    return numerator / denominator


def _mass_project_previous_mode(bm, z, mz, previous_field, mass_action):
    """Mass-orthogonal old trial state in the CURRENT admissible Y/X space.

    The frozen partner space can acquire new, independent constraints when
    a nearly degenerate sixth mode is made orthogonal to earlier modes.
    Euclidean projection of its coefficients is *not* the admissible mass
    projection and must not be used for the variational comparison.

    A tiny but real change of that projection is expected when rotating
    within a repeated eigenspace: it is not a lost frozen-partner direction.
    Fail if the projected state has effectively no mass instead of treating
    a numerical threshold on Euclidean shape difference as a physics rule.
    """
    old = np.asarray(previous_field, float).ravel()
    trial = bm @ z
    if (old.ndim != 1 or trial.ndim != 2
            or old.size != trial.shape[0] or not np.isfinite(old).all()):
        raise ValueError("MODAL MVI invalid previous-shape mass projection")
    physical_mass = np.asarray(mass_action(old), float).ravel()
    if physical_mass.shape != old.shape or not np.isfinite(physical_mass).all():
        raise ValueError("MODAL MVI invalid previous modal mass action")
    original_mass = float(old @ physical_mass)
    if not np.isfinite(original_mass) or original_mass <= 0.:
        raise ValueError("MODAL MVI: nonpositive previous modal mass")
    rhs = trial.T @ physical_mass
    coords = solve(mz, rhs, assume_a="pos", check_finite=True)
    retained_mass = float(coords @ mz @ coords)
    if (not np.isfinite(retained_mass)
            or retained_mass <= 1.e-12 * original_mass):
        raise ValueError("MODAL_MVI_OLD_MODE_LOST_IN_ORTHOGONAL_SPACE")
    # A true mass projection cannot have more energy than the original.
    # A substantial violation means the assembled mass action and its
    # projected matrix are inconsistent (not a candidate for relaxation).
    if retained_mass > original_mass * (1. + 1.e-6):
        raise ValueError("MODAL_MVI_PROJECTED_MASS_INCONSISTENT")
    return coords


def _initial_factors(components, rank, nx, ns, mode):
    """Reproducible factors, not precomputed FDM mode shapes."""
    rng = np.random.default_rng(94031 + 101 * mode + 37 * rank)
    x = np.empty((components, rank, nx), float)
    y = np.empty((components, rank, ns), float)
    for c in range(components):
        x[c] = np.linalg.qr(rng.normal(size=(nx, rank)), mode="reduced")[0].T
        y[c] = np.linalg.qr(rng.normal(size=(ns, rank)), mode="reduced")[0].T
    return x, y


class _LocalMechanicalTangent:
    """Cache local linearized weak coefficients, not a full-grid matrix."""
    def __init__(self, problem, functional, step):
        from cylinder_functional_assembly import MECHANICAL_DERIVATIVES
        self.problem, self.f, self.step = problem, functional, step
        self.spec = MECHANICAL_DERIVATIVES[:functional.n_derivatives]
        nc, nd = functional.n_mechanical, len(self.spec)
        mechanical = [np.zeros((nc, 1, 1)) for _ in range(nd)]
        electrical = [np.zeros((1, 1, 1)) for _ in range(3)]
        self.local = np.empty((nc * nd, nc * nd))
        for j in range(nc * nd):
            c, d = divmod(j, nd)
            mechanical[d][c, 0, 0] = step
            plus = functional._local(functional._make_jets(mechanical, electrical, 0, 0)).mechanical.ravel()
            mechanical[d][c, 0, 0] = -step
            minus = functional._local(functional._make_jets(mechanical, electrical, 0, 0)).mechanical.ravel()
            mechanical[d][c, 0, 0] = 0.
            self.local[:, j] = (plus - minus) / (2. * step)
        if not np.isfinite(self.local).all():
            raise ValueError("MODAL MVI local tangent is nonfinite")

    def action(self, field):
        from cylinder_functional_assembly import _assemble_adjoint
        from cylinder_variational_boundary import compile_boundary_program
        f, step = self.f, self.step
        jets = f._field_derivatives(field, self.spec)
        values = np.stack(jets, axis=1).reshape(len(self.local), -1)
        weak = (self.local @ values).reshape(f.n_mechanical, len(self.spec), *f.ops.shape)
        result = np.zeros_like(field)
        for d, (_, dx, ds) in enumerate(self.spec):
            result += _assemble_adjoint(weak[:, d], f.W, f.dxp[dx], f.dsp[ds])
        def boundary(sign):
            edge = compile_boundary_program(f, sign * step * field, None,
                spec=self.problem.spec, calculation_kind=self.problem.kind,
                check_charge_balance=False)
            return edge.mechanical_residual_correction
        result += (boundary(1.) - boundary(-1.)) / (2. * step)
        return result.ravel()


class _SeparatedModalProblem:
    def __init__(self, prepared, density, rank, *, harmonic_scan=False):
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
        self.large_harmonic=bool(harmonic_scan and self.nx*self.ns>256
                                 and not self.ne and self.nx<=21 and self.ns<=63)
        count=(self.n_components*2*max(self.nx,rank) if self.large_harmonic
               else self.n_components*rank*max(self.nx,self.ns))
        if count > MAX_PROJECTED_DIRECTIONS:
            raise ValueError("MODAL MVI small-grid limit: projected X/Y family has "
                             f"{count} directions, maximum {MAX_PROJECTED_DIRECTIONS}")
        if self.large_harmonic:
            from cylinder_modal_large_harmonic import mechanical_fixed_trace_mask
            self.fixed_mask=mechanical_fixed_trace_mask(prepared.problem,self.q0)
            c=np.empty((0,self.nm))
            target=np.empty(0)
        else:
            first=prepared.problem.assemble(self.q0,self.p0)
            if np.max(np.abs(first.space.origin),initial=0.)>1.e-10:
                raise ValueError("MODAL MVI needs homogeneous essential boundary conditions")
            c,target=first.space.matrix,first.space.target
        if np.max(np.abs(target), initial=0.) > 1.e-12:
            raise ValueError("MODAL MVI cannot prescribe nonzero supports")
        has_m = np.any(c[:, :self.nm] != 0., axis=1)
        has_e = np.any(c[:, self.nm:] != 0., axis=1)
        if np.any(has_m & has_e) or np.any(~(has_m | has_e)):
            raise ValueError("MODAL MVI cannot separate mixed mechanical/electric constraints")
        self.cm = c[has_m, :self.nm]
        self.ce = c[has_e, self.nm:]
        self.n_mechanical_free_dofs = (self.nm-int(self.fixed_mask.sum())
                                      if self.large_harmonic else _free_dof_count(self.cm))
        self.n_electric_free_dofs = (0 if self.large_harmonic else _free_dof_count(self.ce))
        self.step = prepared.settings.tangent_step_scaled
        self.scales = state_scales(self.f, self.q0, self.p0, prepared.settings)
        self.velocity0 = np.zeros_like(self.q0)
        self.density = density
        # A completed mode never changes. Reuse its physical mass action
        # across all X/Y sweeps and orthogonality projections.
        self._previous_mass_cache = {}
        self._local_tangent = (None if self.ne else
                               _LocalMechanicalTangent(prepared.problem, self.f, self.step))

    def previous_mass(self, direction):
        key = id(direction)
        entry = self._previous_mass_cache.get(key)
        if entry is not None and entry[0] is direction:
            return entry[1]
        result = self.mass(direction)
        self._previous_mass_cache[key] = (direction, result)
        return result

    def mass(self, direction):
        q = np.asarray(direction, float).reshape(self.shape)
        return self.f.mass_action(q, density_kg_m3=self.density,
                                  mechanical=self.q0, velocity=self.velocity0).ravel()

    def stiffness(self, direction):
        """Tangent action via the unchanged discrete first variation."""
        direction = np.asarray(direction, float)
        if direction.shape != (self.nm + self.ne,):
            raise ValueError("MODAL MVI tangent vector shape")
        if self._local_tangent is not None:
            return self._local_tangent.action(direction.reshape(self.shape))
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
        def constrained_basis(constraints, scales, directions):
            # Work in dimensionless field coordinates. Frozen partners have
            # unit norm after QR; an already satisfied trace can leave only
            # roundoff in C*T. A relative rank test against that noise alone
            # invents constraints and removes part of the current shape.
            c = constraints * scales[None, :]
            if self.large_harmonic:
                d=directions.copy()
                d[self.fixed_mask,:]=0.
                norms=np.linalg.norm(d,axis=0)
                keep=norms>max(1.e-13,1.e-12*float(np.max(norms,initial=0.)))
                if not np.any(keep):
                    return np.empty((len(scales),0))
                d=d[:,keep]
                norms=np.linalg.norm(d,axis=0)
                gram=d.T@d
                if np.max(np.abs(gram-np.diag(norms*norms)))<1.e-10*max(1.,np.max(norms*norms)):
                    orth=d/norms[None,:]
                else:
                    orth,rr,_=qr(d,mode="economic",pivoting=True,check_finite=False)
                    diag=np.abs(np.diag(rr))
                    valid=int(np.count_nonzero(diag>1.e-12*np.max(diag,initial=0.)))
                    orth=orth[:,:valid]
                if np.max(np.abs(orth[self.fixed_mask,:]),initial=0.)>1.e-11:
                    raise ValueError("MODAL_LARGE_HARMONIC_TRACE_MISMATCH")
                return scales[:,None]*orth
            space = affine_trace_space(
                c, np.zeros(len(c)), np.zeros(len(scales)),
                directions=directions,
                constraint_rank_floor=1.e-12 * np.max(
                    np.linalg.norm(directions, axis=0), initial=0.))
            if np.max(np.abs(space.matrix @ space.basis), initial=0.) > 1.e-10:
                raise ValueError("MODAL MVI: inadmissible frozen-partner basis")
            return scales[:, None] * space.basis

        harmonic = getattr(self, "circumferential_harmonic", None)
        dm = (separated_profile_directions(x[:ncm], y[:ncm], axis)
              if harmonic is None else _harmonic_variation_directions(
                  x[:ncm], y[:ncm], axis, harmonic))
        if self.large_harmonic and self.ns>21 and harmonic is not None and axis=='Y':
            from cylinder_modal_large_harmonic import projected_harmonic_y_basis
            bm=projected_harmonic_y_basis(x[:ncm],self.fixed_mask,self.scales[:self.nm],harmonic)
        else:
            bm = constrained_basis(self.cm, self.scales[:self.nm], dm)
        if not bm.shape[1]:
            raise ValueError("MODAL MVI: no admissible mechanical directions in " + axis)
        if not self.ne:
            return bm, np.empty((0, 0))
        de = (separated_profile_directions(x[ncm:], y[ncm:], axis)
              if harmonic is None else _harmonic_variation_directions(
                  x[ncm:], y[ncm:], axis, harmonic))
        be = constrained_basis(self.ce, self.scales[self.nm:], de)
        return bm, be

    def halfstep(self, x, y, axis, previous, *, current_q=None, on_stage=None):
        """Stationarity of the modal functional on frozen-partner variations."""
        bm, be = self.projected_bases(x, y, axis)
        # Preserve a valid previous separated direction when numerical
        # constraint-rank decisions have discarded it from the test basis.
        if current_q is not None:
            old_q = np.asarray(current_q, dtype=float).ravel()
            if old_q.shape != (self.nm,) or not np.isfinite(old_q).all():
                raise ValueError("MODAL MVI: invalid previous mechanical field")
            old_norm = float(np.linalg.norm(old_q))
            if old_norm <= 0.:
                raise ValueError("MODAL MVI: zero previous mechanical field")
            trace_error = (float(np.max(np.abs(old_q[self.fixed_mask]),initial=0.))
                           if self.large_harmonic else float(np.linalg.norm(self.cm @ old_q, ord=np.inf)))
            trace_scale = (max(1., old_norm) if self.large_harmonic else
                           max(1., float(np.linalg.norm(self.cm, ord=np.inf)) * old_norm))
            if trace_error > 1.e-9 * trace_scale:
                raise ValueError("MODAL MVI: previous field violates essential traces")
            raw = (separated_profile_directions(
                x[:self.shape[0]], y[:self.shape[0]], axis)
                if getattr(self, "circumferential_harmonic", None) is None
                else _harmonic_variation_directions(
                    x[:self.shape[0]], y[:self.shape[0]], axis,
                    self.circumferential_harmonic))
            raw_coeff = np.linalg.lstsq(raw, old_q, rcond=None)[0]
            raw_error = float(np.linalg.norm(raw @ raw_coeff - old_q) / old_norm)
            if not np.isfinite(raw_error) or raw_error > 1.e-9:
                raise ValueError(
                    "MODAL MVI: previous field not in frozen-partner span "
                    f"axis={axis} difference={raw_error:.6g}")
            old_coeff = ((bm.T@old_q)/np.einsum('ij,ij->j',bm,bm)
                         if self.large_harmonic and self.ns>21 else
                         np.linalg.lstsq(bm, old_q, rcond=None)[0])
            missing = old_q - bm @ old_coeff
            missing_ratio = float(np.linalg.norm(missing) / old_norm)
            if missing_ratio > 1.e-9:
                missing /= np.linalg.norm(missing)
                bm = np.column_stack((bm, missing))
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
        mass_factor = cholesky(m, lower=True, check_finite=True)
        keff = k[:nm, :nm].copy()
        electric_recovery = None
        if ne:
            electric_recovery = -solve(k[nm:, nm:], k[nm:, :nm],
                                      assume_a="sym", check_finite=True)
            keff += k[:nm, nm:] @ electric_recovery
            keff = 0.5 * (keff + keff.T)
        constraints = (np.stack([bm.T @ self.previous_mass(q) for q in previous])
                       if previous else np.zeros((0, nm)))
        if len(constraints):
            # A previous mode can already be orthogonal to the entire frozen
            # partner space. A tiny nonzero row then reflects roundoff; forcing
            # its nullspace removes an arbitrary physical direction. Measure
            # the largest possible mass overlap for a unit-mass vector in
            # this space before deciding whether the row can constrain it.
            constraints = _significant_modal_constraints(constraints, mass_factor)
        z = _mass_orthogonal_null(constraints)
        if not z.shape[1]:
            raise ValueError("MODAL MVI rank insufficient for additional orthogonal modes")
        kz, mz = z.T @ keff @ z, z.T @ m @ z
        cholesky(mz, lower=True, check_finite=True)
        # Compare with the *mass-orthogonal* projection of the old shape
        # into the CURRENT constrained frozen-partner test space. This is a
        # physically admissible variational competitor. A small Y/X rotation
        # among nearly repeated modes must not be rejected based on its
        # Euclidean nodal distance to the previous unconstrained shape.
        old_projected_value = None
        reduced = None
        if current_q is not None:
            previous_field = np.asarray(current_q, dtype=float).ravel()
            reduced = _mass_project_previous_mode(
                bm, z, mz, previous_field, self.mass)
            old_mass = float(reduced @ mz @ reduced)
            old_projected_value = float(reduced @ kz @ reduced / old_mass)
        if on_stage is not None:on_stage("EIGEN_BEGIN",0,1)
        # A repeated eigenvalue has no unique eigenvector. Tracking only
        # the first LAPACK vector makes the shape jump between sweeps, even
        # if the physical eigenspace is unchanged. Rotate WITHIN the first
        # eigenvalue cluster to maximize mass overlap with the old iterate.
        # Fetch just the lowest cluster, not the full projected spectrum.
        # Extend the window only when its last eigenvalue is still degenerate.
        dimension = kz.shape[0]
        last = min(dimension - 1, 1)
        while True:
            eigenvalues, eigvec = eigh(
                kz, mz, subset_by_index=(0, last),
                driver="gvx", check_finite=True)
            value = float(eigenvalues[0])
            cluster_tol = 1.e-9 * max(1., abs(value))
            if last == dimension - 1 or (
                    eigenvalues[-1] - value > cluster_tol):
                break
            last = min(dimension - 1, max(last + 1, 2 * last + 1))
        multiplicity = int(np.count_nonzero(
            np.abs(eigenvalues - value) <= cluster_tol))
        if on_stage is not None:on_stage("EIGEN_DONE",multiplicity,multiplicity)
        if value < -1.e-8 * max(1., abs(value)):
            raise ValueError("MODAL MVI: negative squared frequency")
        selected = eigvec[:, 0]
        if multiplicity > 1 and reduced is not None:
            # Use the SAME admissible mass projection for tracking a repeated
            # eigenspace: a Euclidean least-squares vector can rotate or mix
            # two otherwise orthogonal circumferential partners.
            family = eigvec[:, :multiplicity]
            weights = family.T @ mz @ reduced
            magnitude = float(np.linalg.norm(weights))
            if np.isfinite(magnitude) and magnitude > 1.e-12:
                selected = family @ (weights / magnitude)
        # Use the candidate's actual projected Rayleigh quotient. For a
        # thin CUF TE4 shell the LAPACK eigenvalue may differ from the
        # quotient of its own eigenvector (small eigenvalues below a very
        # large stiffness spectrum); only the quotient has the same
        # variational meaning as old_projected_value.
        value = _modal_ritz_value(kz, mz, selected)
        if (old_projected_value is not None and
                value > old_projected_value and
                value - old_projected_value <= 1.e-8 * max(
                    1., abs(value), abs(old_projected_value))):
            selected = reduced / np.sqrt(float(reduced @ mz @ reduced))
            value = old_projected_value
        coeff = z @ selected
        q = bm @ coeff
        p = (None if not self.ne else
             np.zeros(self.ne) if electric_recovery is None else
             be @ (electric_recovery @ coeff))
        return q, p, max(0., value), old_projected_value

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

    def admissible_start(self, mode, previous):
        """Construct a nonzero, homogeneous-BC-admissible fixed-rank seed.

        Random X factors violate clamped-end traces. Starting with a Y
        halfstep on those inadmissible frozen X factors can leave only the
        zero field. Instead, choose the initial X coefficients in the exact
        admissible X variation space with Y frozen, without computing any
        full-grid modal eigenvector or changing the physical stiffness.
        """
        x, y = _initial_factors(self.n_components, self.rank,
                                self.nx, self.ns, mode)
        harmonic = getattr(self, "circumferential_harmonic", None)
        if harmonic is not None:
            phases = _harmonic_phase_basis(self.ns, harmonic)
            for c in range(self.n_components):
                for r in range(self.rank):
                    y[c, r] = phases[r % len(phases)]
        x, y = _normalize_partner(x, y, "X")
        bm, be = self.projected_bases(x, y, "X")
        constraints = (np.stack([bm.T @ self.previous_mass(q) for q in previous])
                       if previous else np.zeros((0, bm.shape[1])))
        z = _mass_orthogonal_null(constraints)
        if not z.shape[1]:
            raise ValueError("MODAL MVI: no admissible independent start for mode")
        rng = np.random.default_rng(54011 + 103*mode + 31*self.rank)
        coeff = z @ rng.standard_normal(z.shape[1])
        q = bm @ coeff
        norm = np.sqrt(float(q @ self.mass(q)))
        if not np.isfinite(norm) or norm <= 1.e-14:
            raise ValueError("MODAL MVI: degenerate admissible start")
        q = q / norm
        if self.ne:
            # Nonzero electrical seed keeps massless electric Y variations
            # active; it carries no fictitious electrical mass.
            p = (be @ rng.standard_normal(be.shape[1])
                 if be.shape[1] else np.zeros(self.ne))
        else:
            p = None
        x, y = self.fit(x, y, "X", q, p)
        return x, y, q, p

    def eigenvalue(self, q, p):
        field = q if p is None else np.concatenate((q, p))
        denominator = float(q @ self.mass(q))
        if denominator <= 0.:
            raise ValueError("MODAL MVI: nonpositive modal mass")
        return float(field @ self.stiffness(field) / denominator)


def solve_modal_mvi(prepared, density, requested, *, progress_callback=None,
                    sweep_callback=None, detail_callback=None, eigen_tolerance=1.e-7,
                    mode_tolerance=1.e-5, harmonic_scan=False,
                    _cached_operator=None, axial_modes_per_harmonic=1):
    """Constrained modal stationarity in alternating Y/X profile spaces.

    Never fall back to FDM, change rank, or accept a stalled mode as
    a completed physical eigenform.
    """
    if prepared.method != "MVI" or prepared.mvi_settings is None:
        raise ValueError("MODAL MVI needs explicit MVI CASE and rank")
    rank = prepared.mvi_settings.rank
    max_sweeps = prepared.mvi_settings.max_sweeps
    if harmonic_scan:
        return _solve_modal_mvi_harmonic_spectrum(
            prepared, density, requested,
            progress_callback=progress_callback,
            detail_callback=detail_callback,
            eigen_tolerance=eigen_tolerance,
            mode_tolerance=mode_tolerance,
            axial_modes_per_harmonic=axial_modes_per_harmonic)
    operator = (_cached_operator if _cached_operator is not None else
                _SeparatedModalProblem(prepared, density, rank))
    qmodes, pmodes, eigs, cycles = [], [], [], []
    for mode in range(requested):
        x, y, initial_q, initial_p = operator.admissible_start(mode, qmodes)
        last_q = initial_q.copy()
        last_lambda = operator.eigenvalue(initial_q, initial_p)
        half_q, half_p = initial_q, initial_p
        converged = False
        for sweep in range(1, max_sweeps + 1):
            if detail_callback is not None:
                detail_callback(stage="MVI_SWEEP_START",mode=mode+1,
                    n_modes=requested,rank=rank,sweep=sweep,max_sweeps=max_sweeps)
            # Once the admissible X seed is prepared, preserve the classical
            # static-shell ordering Y -> X for each completed MVI sweep.
            for axis in ("Y", "X"):
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
                old_value = operator.eigenvalue(half_q, half_p)
                q, p, projected_value, old_projected = operator.halfstep(
                    x, y, axis, qmodes, current_q=half_q, on_stage=trace)
                value_half = operator.eigenvalue(q, p)
                # The projected minimum cannot exceed the old shape's
                # Rayleigh quotient on the SAME frozen-partner K and M.
                # Compare those values, not two separately finite-differenced
                # full-field evaluations of the nonlinear shell residual.
                if old_projected is None:
                    raise ValueError("MODAL MVI: missing projected old state")
                numerical_stationarity, observed_mismatch = (
                    _modal_variational_step_precision(
                        old_projected, projected_value, old_value, value_half))
                if numerical_stationarity and detail_callback is not None:
                    detail_callback(stage="MVI_NUMERICAL_STATIONARITY",
                        mode=mode+1,sweep=sweep,axis=axis,
                        projected_increase=float(projected_value-old_projected),
                        energy_disagreement=float(observed_mismatch),
                        physical_change=float(value_half-old_value))
                # Separate linearization reproducibility from actual failure
                # of the frozen-partner variational update. This diagnostic
                # makes any mismatch visible; it does not fabricate a mode.
                if detail_callback is not None:
                    detail_callback(stage="MVI_VARIATIONAL_CHECK",mode=mode+1,
                        n_modes=requested,rank=rank,sweep=sweep,axis=axis,
                        projected_lambda_old=float(old_projected),
                        projected_lambda_new=float(projected_value),
                        field_lambda_old=float(old_value),
                        field_lambda_new=float(value_half),
                        tangent_difference_old=float(old_value-old_projected),
                        tangent_difference_new=float(value_half-projected_value))
                x, y = operator.fit(x, y, axis, q, p)
                half_q, half_p = q, p
                if detail_callback is not None:
                    detail_callback(stage="MVI_HALFSTEP_DONE",mode=mode+1,
                        n_modes=requested,rank=rank,sweep=sweep,axis=axis,
                        frequency_hz=float(np.sqrt(max(0.,value_half))/(2*np.pi)))
            value = operator.eigenvalue(q, p)
            if not np.isfinite(value) or value < 0.:
                raise ValueError("MODAL MVI: invalid Rayleigh quotient")
            relative, change = float("inf"), float("inf")
            if last_q is not None:
                old_mass = float(last_q @ operator.mass(last_q))
                new_mass = float(q @ operator.mass(q))
                if not (np.isfinite(old_mass) and np.isfinite(new_mass)
                        and old_mass > 0. and new_mass > 0.):
                    raise ValueError("MODAL MVI: invalid mass in shape comparison")
                overlap = float(last_q @ operator.mass(q))
                if overlap < 0.:
                    q = -q
                    p = None if p is None else -p
                    x = -x
                    overlap = -overlap
                # Compare UNIT-MASS fields. The raw eigenvectors can have
                # different norms in successive frozen-partner subspaces.
                cosine = np.clip(overlap / np.sqrt(old_mass * new_mass), 0., 1.)
                change = float(np.sqrt(max(0., 2. - 2. * cosine)))
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
                    shape_change_relative=(None if last_q is None else float(change)))
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
        for previous in qmodes:
            overlap = abs(float(q @ operator.previous_mass(previous)))
            if overlap > 1.e-6:
                raise ValueError(
                    f"MODAL_MVI_MODE_NOT_ORTHOGONAL: mode={mode+1}, "
                    f"overlap={overlap:.5e}")
        qmodes.append(q.copy())
        pmodes.append(None if p is None else p.copy())
        eigs.append(value)
        cycles.append(sweep)
        if detail_callback is not None:
            detail_callback(stage="MVI_MODE_DONE",mode=mode+1,
                n_modes=requested,rank=rank,sweep=sweep,
                frequency_hz=float(np.sqrt(value)/(2.0*np.pi)),
                omega_rad_s=float(np.sqrt(value)),
                modal_amplitude_max=float(np.max(abs(q),initial=0.)))
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
        "n_mechanical_free_dofs": operator.n_mechanical_free_dofs,
        "n_electric_free_dofs": operator.n_electric_free_dofs,
        "mvi_sweeps": [cycles[i] for i in order],
        "mvi_modal_algorithm": "fixed_rank_constrained_variational_YX_stationarity",
    }


__all__ = ("MAX_PROJECTED_DIRECTIONS", "solve_modal_mvi")

