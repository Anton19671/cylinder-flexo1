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
from cylinder_variational_mvi import _normalize_partner, _initial_profiles, reconstruct_profiles
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


def _modal_shape_changes(previous_q, current_q, previous_p, current_p,
                         mass_action):
    """Mass-normalized mechanical and electric shape changes, sign aligned.

    Only mechanical displacements carry physical inertia. Electrical factors
    are quasi-static but must still stop changing before a coupled
    eigenmode is accepted. This checks convergence; it does not change K/M.
    """
    old_q = np.asarray(previous_q, dtype=float).ravel()
    new_q = np.asarray(current_q, dtype=float).ravel()
    if old_q.shape != new_q.shape or not np.isfinite(old_q).all() or not np.isfinite(new_q).all():
        raise ValueError("MODAL_MVI_INVALID_MODE_SHAPES")
    old_mass = float(old_q @ mass_action(old_q))
    new_mass = float(new_q @ mass_action(new_q))
    if not (np.isfinite(old_mass) and np.isfinite(new_mass)
            and old_mass > 0. and new_mass > 0.):
        raise ValueError("MODAL MVI: invalid mass in shape comparison")
    overlap = float(old_q @ mass_action(new_q))
    if not np.isfinite(overlap):
        raise ValueError("MODAL_MVI_INVALID_MODE_OVERLAP")
    orientation = -1 if overlap < 0. else 1
    cosine = np.clip(orientation * overlap / np.sqrt(old_mass * new_mass), 0., 1.)
    mechanical_change = float(np.sqrt(max(0., 2. - 2. * cosine)))
    if (previous_p is None) != (current_p is None):
        raise ValueError("MODAL_MVI_ELECTRIC_SHAPE_MISSING")
    electric_change = 0.
    if current_p is not None:
        old_p = np.asarray(previous_p, dtype=float).ravel()
        new_p = np.asarray(current_p, dtype=float).ravel()
        if (old_p.shape != new_p.shape or not np.isfinite(old_p).all()
                or not np.isfinite(new_p).all()):
            raise ValueError("MODAL_MVI_INVALID_ELECTRIC_SHAPES")
        old_unit = old_p / np.sqrt(old_mass)
        new_unit = orientation * new_p / np.sqrt(new_mass)
        electric_change = float(np.linalg.norm(new_unit - old_unit) /
            max(np.linalg.norm(new_unit), np.linalg.norm(old_unit), 1.e-30))
    return orientation, mechanical_change, electric_change


def _initial_factors(components, rank, nx, ns, mode):
    """Reproducible factors, not precomputed FDM mode shapes."""
    rng = np.random.default_rng(94031 + 101 * mode + 37 * rank)
    x = np.empty((components, rank, nx), float)
    y = np.empty((components, rank, ns), float)
    for c in range(components):
        x[c] = np.linalg.qr(rng.normal(size=(nx, rank)), mode="reduced")[0].T
        y[c] = np.linalg.qr(rng.normal(size=(ns, rank)), mode="reduced")[0].T
    return x, y


def _kl_linear_local_tangent(functional):
    """Exact Hessian of the linear KL weak energy in surface-jet coordinates.

    Form B^T C B directly at each thickness point.  Differencing weak
    forces after constructing F = I + grad(u) loses small diagonal strains
    when I is subtracted again, which contaminates thin-shell bending terms.
    """
    from cylinder_kl_g3d_weak import (
        _kl_small_virtual_F_library, _variation, gauss_thickness)
    from flexo_gradient_g3d import covariant_strain_gradients, enthalpy_derivatives

    radius, thickness = functional.R, functional.h
    local = np.zeros((30, 30))
    identity = np.eye(3)
    zero = np.zeros((3, 3))
    for z, weight in zip(*gauss_thickness(thickness, functional.n_gauss)):
        z = float(z)
        virtual = _kl_small_virtual_F_library(
            radius, thickness, z, "LINEAR").reshape(30, 4, 3, 3)
        strains = np.empty((30, 6))
        gradients = np.empty((30, 6, 3))
        for i, direction in enumerate(virtual):
            strain, derivs = _variation(
                identity, zero, zero, zero, *direction, "LINEAR")
            strains[i] = strain
            gradients[i] = covariant_strain_gradients(
                strain, *derivs, radius_m=radius, z_m=z)
        for j in range(30):
            stress, higher_stress, _ = enthalpy_derivatives(
                strains[j], gradients[j], np.zeros(3), functional.material)
            local[:, j] += weight * (1. + z / radius) * (
                strains @ stress +
                np.einsum("ijk,jk->i", gradients, higher_stress))
    return local


class _LocalMechanicalTangent:
    """Cache local linearized weak coefficients, not a full-grid matrix."""
    def __init__(self, problem, functional, step):
        from cylinder_functional_assembly import MECHANICAL_DERIVATIVES
        self.problem, self.f, self.step = problem, functional, step
        self.spec = MECHANICAL_DERIVATIVES[:functional.n_derivatives]
        nc, nd = functional.n_mechanical, len(self.spec)
        mechanical = [np.zeros((nc, 1, 1)) for _ in range(nd)]
        electrical = [np.zeros((1, 1, 1)) for _ in range(3)]
        if functional.theory == "KL" and functional.geometry_model == "LINEAR":
            self.local = _kl_linear_local_tangent(functional)
        else:
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
        self.large_harmonic=bool(self.nx*self.ns>256 and not self.ne
                                 and self.nx<=21 and self.ns<=63)
        count=self.n_components*rank*max(self.nx,self.ns)
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
                constraint_rank_floor=1.e-10 * np.max(
                    np.linalg.norm(directions, axis=0), initial=0.))
            if np.max(np.abs(space.matrix @ space.basis), initial=0.) > 1.e-10:
                raise ValueError("MODAL MVI: inadmissible frozen-partner basis")
            return scales[:, None] * space.basis

        if self.large_harmonic:
            from cylinder_modal_separated_space import fixed_trace_xy_basis
            bm = fixed_trace_xy_basis(
                x[:ncm], y[:ncm], axis, self.fixed_mask,
                self.scales[:self.nm], max_directions=MAX_PROJECTED_DIRECTIONS)
            if not bm.shape[1]:
                raise ValueError("MODAL MVI: no admissible mechanical directions in " + axis)
            return bm, np.empty((0, 0))
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
            # QR projection checks membership without the SVD-based lstsq path.
            raw_q, raw_r, _ = qr(raw, mode="economic", pivoting=True,
                                  check_finite=False)
            raw_diag = np.abs(np.diag(raw_r))
            raw_rank = int(np.count_nonzero(
                raw_diag > 1.e-12 * np.max(raw_diag, initial=0.)))
            raw_error = float(np.linalg.norm(
                old_q - raw_q[:, :raw_rank] @
                (raw_q[:, :raw_rank].T @ old_q)) / old_norm)
            if not np.isfinite(raw_error) or raw_error > 1.e-9:
                raise ValueError(
                    "MODAL MVI: previous field not in frozen-partner span "
                    f"axis={axis} difference={raw_error:.6g}")
            # The admissible test basis must remain inside the current
            # frozen-partner variation space. Appending a missing nodal
            # vector changes the variational method into an enriched FDM
            # subspace, even if that vector satisfies essential traces.
            basis_q, basis_r, _ = qr(bm, mode="economic", pivoting=True,
                                      check_finite=False)
            basis_diag = np.abs(np.diag(basis_r))
            basis_rank = int(np.count_nonzero(
                basis_diag > 1.e-12 * np.max(basis_diag, initial=0.)))
            missing = old_q - basis_q[:, :basis_rank] @ (
                basis_q[:, :basis_rank].T @ old_q)
            missing_ratio = float(np.linalg.norm(missing) / old_norm)
            if not np.isfinite(missing_ratio) or missing_ratio > 1.e-9:
                raise ValueError(
                    "MODAL_MVI_FROZEN_PARTNER_BASIS_LOST_CURRENT_FIELD: "
                    f"axis={axis} difference={missing_ratio:.6g}")
        nm, ne = bm.shape[1], be.shape[1]
        # Report the ACTUAL BC-constrained frozen-partner dimensions.
        # This detects rank-independent trial spaces without forcing
        # physically identical eigenfrequencies to differ.
        if on_stage is not None:
            on_stage("SUBSPACE_DIMENSIONS", nm, ne)
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
        """Recover exactly the trial field in its fixed-rank separated space.

        For a circular Fourier sector every component is spanned by a single
        constant phase (n=0) or two cos/sin phases (n>0). Generic QR partner
        completion may insert a numerically arbitrary off-sector direction
        for MVI ranks 3..8. Never fit the physical Fourier field against
        that artificial direction: factor the actual sector coefficients.
        """
        field = q if p is None else np.concatenate((q, p))
        scaled = (field / self.scales).reshape(self.n_components, self.nx, self.ns)
        harmonic = getattr(self, "circumferential_harmonic", None)
        if harmonic is not None:
            phase = _harmonic_phase_basis(self.ns, harmonic)
            xnew = np.zeros_like(x)
            ynew = np.zeros_like(y)
            for c in range(self.n_components):
                coefficients = scaled[c] @ phase.T
                # Pivoted QR of the at-most-two phase coordinates, not SVD.
                frame, triangular, _ = qr(
                    coefficients.T, mode="economic", pivoting=True,
                    check_finite=False)
                diagonal = np.abs(np.diag(triangular))
                active = int(np.count_nonzero(
                    diagonal > 1.e-12 * np.max(diagonal, initial=0.)))
                for r in range(min(self.rank, active)):
                    xnew[c, r, :] = coefficients @ frame[:, r]
                    ynew[c, r, :] = frame[:, r] @ phase
            restored = reconstruct_profiles(xnew, ynew)
            if np.linalg.norm(restored-scaled) > 1.e-8 * max(
                    float(np.linalg.norm(scaled)), 1.e-30):
                raise ValueError("MODAL MVI harmonic factor reconstruction failed")
            return xnew, ynew
        if axis == "X":
            x = np.einsum("cxs,crs->crx", scaled, y)
        else:
            y = np.einsum("cxs,crx->crs", scaled, x)
        fit = reconstruct_profiles(x, y)
        if np.linalg.norm(fit - scaled) > 1.e-8 * max(
                float(np.linalg.norm(scaled)), 1.e-30):
            raise ValueError("MODAL MVI separated factor reconstruction failed")
        return x, y

    def admissible_start(self, mode, previous, *, harmonic_seed=None,
                             harmonic_phase="cos", component_phase="uniform"):
        """Construct a nonzero, homogeneous-BC-admissible fixed-rank seed.

        Random X factors violate clamped-end traces. Starting with a Y
        halfstep on those inadmissible frozen X factors can leave only the
        zero field. Instead, choose the initial X coefficients in the exact
        admissible X variation space with Y frozen, without computing any
        full-grid modal eigenvector or changing the physical stiffness.
        """
        x, y = _initial_factors(self.n_components, self.rank,
                                self.nx, self.ns, mode)
        # Trigonometric INITIAL PARTNERS are seeds, never Fourier constraints:
        # all subsequent frozen-partner X/Y variational directions stay full.
        if harmonic_seed is not None:
            n = int(harmonic_seed)
            if n < 0 or n > self.ns // 2 or harmonic_phase not in ("cos", "sin"):
                raise ValueError("MODAL_MVI_INVALID_PARTNER_SEED")
            angle = 2. * np.pi * np.arange(self.ns) / self.ns
            first = (np.cos(n * angle) if harmonic_phase == "cos"
                     else np.sin(n * angle))
            if float(np.linalg.norm(first)) < 1.e-12:
                raise ValueError("MODAL_MVI_DEGENERATE_PARTNER_SEED")
            first = first / np.linalg.norm(first)
            second = (np.sin(n * angle) if harmonic_phase == "cos"
                      else np.cos(n * angle))
            quadrature_components = (
                _circumferential_seed_components(self.f)
                if component_phase == "circumferential_quadrature" else ())
            if component_phase == "circumferential_quadrature" and (
                    n == 0 or not quadrature_components):
                raise ValueError("MODAL_MVI_INVALID_COMPONENT_PHASE")
            for c in range(self.n_components):
                choices = y[c].T.copy()
                # CUF fields are grouped [u_0..u_N, v_0..v_N, w_0..w_N].
                # Change only the initial angular factors of v coefficients;
                # electrical fields and all later variations remain free.
                selected = (second if c in quadrature_components else first)
                choices[:, 0] = selected / np.linalg.norm(selected)
                if self.rank > 1 and n > 0 and np.linalg.norm(second) > 1.e-12:
                    choices[:, 1] = second / np.linalg.norm(second)
                orth, _ = np.linalg.qr(choices, mode="reduced")
                y[c] = orth.T
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
        if (harmonic_seed is not None and self.rank > 2
                and harmonic is None and not self.ne):
            # All configured rank directions are still present in the MVI.
            # For rank >= 3, however, orthonormalizing the two sine/cosine
            # seed partners adds random off-harmonic Y columns. Taking random
            # coefficients in ALL those columns makes a supposedly n-harmonic
            # start non-harmonic; the first Y minimum can then miss a lower
            # angular branch. Project ONLY this initial trial field onto its
            # requested physical Fourier sector. The subsequent X/Y halfsteps
            # still vary every rank factor WITHOUT an angular restriction.
            phase = _harmonic_phase_basis(self.ns, int(harmonic_seed))
            def sector(field):
                values = np.asarray(field, float).reshape(self.shape)
                return ((values @ phase.T) @ phase).ravel()
            trial = sector(q)
            # Previously accepted modes retain their independent physical
            # mass constraints. Orthogonalize within the seed sector rather
            # than fabricating a new or shifted eigenmode.
            for old in previous:
                old_sector = sector(old)
                old_mass = self.mass(old_sector)
                denominator = float(old_sector @ old_mass)
                if denominator > 1.e-20:
                    trial -= old_sector * (float(trial @ old_mass) / denominator)
            sector_mass = float(trial @ self.mass(trial))
            sector_bc = float(np.max(np.abs(self.cm @ trial), initial=0.))
            if (np.isfinite(sector_mass) and sector_mass > 1.e-20
                    and sector_bc <= 1.e-10 * max(1., float(np.linalg.norm(trial)))
                    and all(abs(float(trial @ self.previous_mass(old))) <=
                            1.e-7 * np.sqrt(sector_mass * float(
                                old @ self.previous_mass(old)))
                            for old in previous)):
                q = trial
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

    def admissible_quadrature_seed(self, previous, electric_previous, earlier):
        """Admissible starting field for the missing rotationally repeated mode.

        ONLY the initial field is rotated. Its factorization retains the
        requested rank; normal unrestricted Y->X MVI then recomputes the
        stationary mode. Never append a transformed field as an eigenmode.
        An earlier member of the same pair or non-harmonic fields are rejected.
        """
        if self.ns % 2 != 1:
            raise ValueError("MODAL_MVI_ROTATED_SEED_NEEDS_ODD_PERIODIC_GRID")
        previous_field = np.asarray(previous, float).reshape(self.shape)
        spectrum = np.abs(np.fft.rfft(previous_field, axis=-1))**2
        energy_by_n = np.sum(spectrum, axis=(0, 1))
        n = int(np.argmax(energy_by_n))
        if n == 0 or _pure_harmonic_fraction(previous_field, n) < 1.-1.e-8:
            raise ValueError("MODAL_MVI_ROTATED_SEED_NOT_SINGLE_HARMONIC")
        q = _phase_quadrature_partner(previous_field, n).ravel()
        norm = np.sqrt(float(q @ self.mass(q)))
        if not np.isfinite(norm) or norm <= 0.:
            raise ValueError("MODAL_MVI_ROTATED_SEED_BAD_MASS")
        q /= norm
        for existing in (*earlier, previous):
            if abs(float(q @ self.previous_mass(existing))) > 1.e-7:
                raise ValueError("MODAL_MVI_ROTATED_SEED_NOT_INDEPENDENT")
        if self.large_harmonic:
            defect = float(np.max(np.abs(q[self.fixed_mask]), initial=0.))
        else:
            defect = float(np.max(np.abs(self.cm @ q), initial=0.))
        if defect > 1.e-10 * max(1., float(np.linalg.norm(q))):
            raise ValueError("MODAL_MVI_ROTATED_SEED_VIOLATES_BC")
        p = (None if electric_previous is None else
             _phase_quadrature_partner(
                 np.asarray(electric_previous).reshape(self.p0.shape), n
             ).ravel() / norm)
        if p is not None and np.max(np.abs(self.ce @ p),initial=0.) > (
                1.e-10 * max(1.,float(np.linalg.norm(p)))):
            raise ValueError("MODAL_MVI_ROTATED_SEED_VIOLATES_ELECTRODES")
        combined = q if p is None else np.concatenate((q,p))
        scaled = (combined / self.scales).reshape(
            self.n_components,self.nx,self.ns)
        x,y = _initial_profiles(
            scaled,self.rank,np.linalg.norm(scaled,axis=(1,2)))
        restored = reconstruct_profiles(x,y)
        if np.linalg.norm(restored-scaled) > 1.e-10 * max(
                1.,float(np.linalg.norm(scaled))):
            raise ValueError("MODAL_MVI_ROTATED_SEED_RANK_RECONSTRUCTION")
        return x,y,q,p

    def eigenvalue(self, q, p):
        field = q if p is None else np.concatenate((q, p))
        denominator = float(q @ self.mass(q))
        if denominator <= 0.:
            raise ValueError("MODAL MVI: nonpositive modal mass")
        return float(field @ self.stiffness(field) / denominator)


def _circumferential_seed_components(functional):
    """Indices of v thickness coefficients in the mechanical field ordering."""
    theory = str(functional.theory).upper()
    if theory == "KL":
        return (1,)
    if theory in ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4"):
        n_mechanical = int(functional.n_mechanical)
        n_thickness = n_mechanical // 3
        if n_mechanical != 3 * n_thickness or n_thickness != int(theory[-1]) + 1:
            raise ValueError("MODAL_MVI_INVALID_CUF_COMPONENT_ORDER")
        return tuple(range(n_thickness, 2 * n_thickness))
    return ()


def _modal_seed_plan(operator, mode, requested):
    """Bounded restarts; partners never restrict the variational halfsteps.

    Random starts can miss lower eigenpairs under sequential deflation.
    The trigonometric partners broaden basins but do not select an operator
    harmonic sector; spectral completeness remains unverified.
    """
    count = operator.n_components * operator.nx * operator.ns
    offsets = (0, 31, 101) if count <= 2500 else (0, 31)
    starts = [{"kind": "random", "index": int(mode + offset)}
              for offset in offsets]
    limit = (min(operator.ns // 2, 4) if count <= 2500
             else min(operator.ns // 2, 2) if count <= 8000
             else min(operator.ns // 2, 1))
    for n in range(limit + 1):
        phases = ("cos", "sin") if n > 0 and count <= 2500 else ("cos",)
        for phase in phases:
            starts.append({"kind": "harmonic_partner",
                           "index": int(mode + 503 + 17*n +
                                        (7 if phase == "sin" else 0)),
                           "n": int(n), "phase": phase})
    if operator.rank == 1 and _circumferential_seed_components(operator.f):
        for n in range(1, limit + 1):
            for phase in ("cos", "sin"):
                starts.append({
                    "kind": "harmonic_partner",
                    "index": int(mode + 1103 + 17*n + (7 if phase == "sin" else 0)),
                    "n": int(n), "phase": phase,
                    "component_phase": "circumferential_quadrature"})
    # The C-C 13x13 benchmark has a low n=5 doublet. Rank-two
    # random starts can miss it even when rank one finds it. Keep explicit
    # starts through n=6 on small problems, using the verified probe seeds.
    if count <= 2500:
        for n in range(max(5, limit + 1), min(operator.ns // 2, 6) + 1):
            for phase in ("cos", "sin"):
                starts.append({
                    "kind": "harmonic_partner",
                    "index": int(mode + 4001 + 17*n + (7 if phase == "sin" else 0)),
                    "n": n, "phase": phase})
    return starts


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
        raise ValueError(
            "MODAL_MVI_HARMONIC_SOLVER_RETIRED: use the unified unrestricted "
            "fixed-rank Y/X modal solver; inspect harmonics after the solve")
    # A reused operator with an earlier rank or CASE silently solves the
    # WRONG variational space while the output claims the requested rank.
    if _cached_operator is not None and (
            getattr(_cached_operator, "rank", None) != rank
            or getattr(_cached_operator, "prepared", None) is not prepared):
        raise ValueError("MODAL_MVI_CACHED_OPERATOR_CASE_OR_RANK_MISMATCH")
    operator = (_cached_operator if _cached_operator is not None else
                _SeparatedModalProblem(prepared, density, rank))
    qmodes, pmodes, eigs, cycles = [], [], [], []
    final_xy_spaces = []
    multistart_audit = []

    def solve_candidate(mode, seed):
        last_xy_spaces = {"Y": None, "X": None}
        if seed["kind"] == "quadrature_previous":
            x, y, initial_q, initial_p = (
                operator.admissible_quadrature_seed(
                    qmodes[-1], pmodes[-1], qmodes[:-1]))
        else:
            x, y, initial_q, initial_p = operator.admissible_start(
                seed["index"], qmodes, harmonic_seed=seed.get("n"),
                harmonic_phase=seed.get("phase", "cos"),
                component_phase=seed.get("component_phase", "uniform"))
        def harmonic_state(q):
            field = np.asarray(q, float).reshape(operator.shape)
            spectrum = np.abs(np.fft.rfft(field, axis=-1)) ** 2
            energy = np.sum(spectrum, axis=(0, 1))
            total = float(np.sum(energy))
            n = int(np.argmax(energy)) if total > 0. else 0
            fraction = float(energy[n] / total) if total > 0. else 0.
            return n, fraction

        def report_candidate_state(stage, q, p, *, sweep=0, axis=None):
            if detail_callback is None:
                return
            n, fraction = harmonic_state(q)
            overlaps = [abs(float(q @ operator.previous_mass(old)))
                        for old in qmodes]
            detail_callback(
                stage=stage, mode=mode+1, n_modes=requested, rank=rank,
                attempt=attempt, seed=seed, sweep=sweep, axis=axis,
                frequency_hz=float(np.sqrt(max(0., operator.eigenvalue(q, p))) /
                                   (2. * np.pi)),
                circumferential_harmonic=n,
                circumferential_harmonic_energy_fraction=fraction,
                max_previous_mass_overlap=(max(overlaps) if overlaps else 0.))

        # A pure Fourier seed is admissible but its random axial factor is
        # not yet characteristic of that harmonic.  Starting directly with
        # an unrestricted Y minimization lets it change n before the
        # corresponding X factor is found.  Perform one exact X-frozen
        # variational halfstep first; it stays in the seed's Y harmonic
        # space, uses the same K/M/traces and the same mass deflation, and is
        # only initialization.  The classical modal iteration below remains
        # Y -> X on every completed sweep.
        report_candidate_state("MVI_CANDIDATE_INITIAL_STATE",
                               initial_q, initial_p)
        seed_n, seed_fraction = harmonic_state(initial_q)
        if (seed["kind"] in ("harmonic_partner", "quadrature_previous")
                and seed_n > 0 and seed_fraction >= 1. - 1.e-8):
            x, y = _normalize_partner(x, y, "X")
            old_value = operator.eigenvalue(initial_q, initial_p)

            def initial_x_trace(stage, column, total):
                if stage == "SUBSPACE_DIMENSIONS":
                    last_xy_spaces["X_INITIAL"] = {
                        "mechanical_directions": int(column),
                        "electric_directions": int(total)}
                if detail_callback is not None:
                    detail_callback(
                        stage="MVI_INITIAL_X_" + stage, mode=mode+1,
                        n_modes=requested, rank=rank, attempt=attempt,
                        seed=seed, sweep=0, axis="X",
                        column=column, total_columns=total)

            # For ranks 3..8, QR partner completion adds angular vectors
            # outside the original seed's sine/cosine pair. The physical
            # initial state was correctly projected onto one Fourier sector,
            # but the unrestricted X trial space would immediately mix those
            # other sectors back in. Restrict ONLY this initial conditioning
            # halfstep to the harmonic actually present in the seed. After
            # factor recovery, restore the unrestricted rank-r Y/X method;
            # this is not a spectrum-by-Fourier-sector solver.
            initialize_in_sector = (
                operator.rank > 2 and operator.ne == 0 and
                getattr(operator, "circumferential_harmonic", None) is None)
            if initialize_in_sector:
                operator.circumferential_harmonic = seed_n
            try:
                q, p, projected_value, old_projected = operator.halfstep(
                    x, y, "X", qmodes, current_q=initial_q,
                    on_stage=initial_x_trace)
                if old_projected is None:
                    raise ValueError("MODAL MVI: missing initial X projected state")
                _modal_variational_step_precision(
                    old_projected, projected_value, old_value,
                    operator.eigenvalue(q, p))
                x, y = operator.fit(x, y, "X", q, p)
            finally:
                if initialize_in_sector:
                    del operator.circumferential_harmonic
            initial_q, initial_p = q, p
            report_candidate_state("MVI_CANDIDATE_INITIAL_X_DONE", q, p,
                                   axis="X")
        last_q = initial_q.copy()
        last_p = None if initial_p is None else initial_p.copy()
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
                    if stage == "SUBSPACE_DIMENSIONS":
                        actual = {
                            "mechanical_directions": int(column),
                            "electric_directions": int(total)}
                        last_xy_spaces[axis] = actual
                        if detail_callback is not None:
                            detail_callback(stage="MVI_FROZEN_PARTNER_SPACE",
                                mode=mode+1,n_modes=requested,rank=rank,
                                sweep=sweep,max_sweeps=max_sweeps,axis=axis,
                                **actual)
                        return
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
                report_candidate_state("MVI_HALFSTEP_HARMONIC_AUDIT", q, p,
                                       sweep=sweep, axis=axis)
                if detail_callback is not None:
                    detail_callback(stage="MVI_HALFSTEP_DONE",mode=mode+1,
                        n_modes=requested,rank=rank,sweep=sweep,axis=axis,
                        frequency_hz=float(np.sqrt(max(0.,value_half))/(2*np.pi)))
            value = operator.eigenvalue(q, p)
            if not np.isfinite(value) or value < 0.:
                raise ValueError("MODAL MVI: invalid Rayleigh quotient")
            relative, change, electric_change = (
                float("inf"), float("inf"), float("inf"))
            if last_q is not None:
                orientation, change, electric_change = _modal_shape_changes(
                    last_q, q, last_p, p, operator.mass)
                if orientation < 0:
                    q = -q
                    p = None if p is None else -p
                    x = -x
                    # The next Y halfstep must receive the SAME signed field
                    # as represented by the updated separated profiles.
                    half_q, half_p = q, p
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
                    electric_shape_change_relative=(
                        None if p is None else float(electric_change)))
            if (relative <= eigen_tolerance and change <= mode_tolerance
                    and electric_change <= mode_tolerance):
                converged = True
                break
            last_lambda, last_q = value, q.copy()
            last_p = None if p is None else p.copy()
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
        return (q.copy(), None if p is None else p.copy(),
                float(value), int(sweep), last_xy_spaces)

    for mode in range(requested):
        plan = _modal_seed_plan(operator, mode, requested)
        # Missing members of an angularly degenerate pair are difficult to
        # reach with random factors. A quarter-phase rotation of the last
        # converged mode is an independent TRIAL SEED, not a new solver or
        # an assumed eigenpair. It must pass all ordinary Y/X validations.
        if qmodes:
            plan.insert(0, {"kind": "quadrature_previous", "index": int(mode)})
        accepted = []
        failures = []
        for attempt, seed in enumerate(plan, 1):
            if detail_callback is not None:
                detail_callback(stage="MVI_CANDIDATE_START",
                    mode=mode+1,n_modes=requested,rank=rank,
                    attempt=attempt,attempts=len(plan),seed=seed)
            try:
                candidate = solve_candidate(mode, seed)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures.append({"attempt":attempt,"seed":seed,
                                 "reason":str(exc)[:240]})
                if detail_callback is not None:
                    detail_callback(stage="MVI_CANDIDATE_REJECTED",
                        mode=mode+1,n_modes=requested,rank=rank,
                        attempt=attempt,reason=str(exc)[:240])
                continue
            accepted.append((*candidate, seed, attempt))
            if detail_callback is not None:
                detail_callback(stage="MVI_CANDIDATE_CONVERGED",
                    mode=mode+1,n_modes=requested,rank=rank,
                    attempt=attempt,seed=seed,
                    frequency_hz=float(np.sqrt(candidate[2])/(2.*np.pi)))
        if not accepted:
            raise ValueError(
                f"MODAL_MVI_NO_CONVERGED_CANDIDATE: rank={rank}, "
                f"mode={mode+1}, seeds={len(plan)}, "
                f"last_failure={failures[-1] if failures else 'none'}")
        # Candidates passed the same physical BC, mass normalization and
        # previous-mode orthogonality checks. Choose the lowest actual
        # Rayleigh eigenvalue rather than the first local stationary point.
        q,p,value,sweep,last_xy_spaces,winning_seed,winning_attempt = min(
            accepted,key=lambda candidate:(candidate[2],candidate[6]))
        multistart_audit.append({
            "tried":len(plan),"converged":len(accepted),
            "selected_seed":winning_seed,"selected_attempt":winning_attempt,
            "frequencies_hz":sorted(float(np.sqrt(c[2])/(2.*np.pi))
                                    for c in accepted),
            "failed_attempts":failures})
        if detail_callback is not None:
            detail_callback(stage="MVI_CANDIDATE_SELECTED",
                mode=mode+1,n_modes=requested,rank=rank,
                attempt=winning_attempt,seed=winning_seed,
                converged=len(accepted),attempts=len(plan),
                frequency_hz=float(np.sqrt(value)/(2.*np.pi)))
        qmodes.append(q.copy())
        pmodes.append(None if p is None else p.copy())
        eigs.append(value)
        cycles.append(sweep)
        final_xy_spaces.append({
            axis: (None if last_xy_spaces[axis] is None
                   else dict(last_xy_spaces[axis]))
            for axis in ("Y", "X")})
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
    # A configured rank applies independently to every CUF component.
    # Report the rank of the assembled vector field as well, so MVI-1
    # cannot be misrepresented as globally rank-one in FDM comparisons.
    global_mode_ranks = []
    for field in q_out:
        assembled = field.reshape(field.shape[0] * field.shape[1],
                                  field.shape[2])
        _, triangular, _ = qr(assembled.T, mode="economic",
                              pivoting=True, check_finite=False)
        diagonal = np.abs(np.diag(triangular))
        cutoff = 1.e-10 * np.max(diagonal, initial=0.)
        global_mode_ranks.append(int(np.count_nonzero(diagonal > cutoff)))
    return {
        "theory": operator.f.theory,
        "method": "MVI", "rank": rank, "rank_scope": "per_component",
        "global_mode_ranks": global_mode_ranks,
        "density_kg_m3": density,
        "frequencies_hz": omega / (2. * np.pi),
        "omega_rad_s": omega, "omega_squared_rad2_s2": eigenvalues,
        "mechanical_modes": q_out, "electrical_modes": p_out,
        "n_mechanical_free_dofs": operator.n_mechanical_free_dofs,
        "n_electric_free_dofs": operator.n_electric_free_dofs,
        "mvi_sweeps": [cycles[i] for i in order],
        "mvi_final_xy_trial_dimensions": [final_xy_spaces[i] for i in order],
        "mvi_multistart_audit": [multistart_audit[i] for i in order],
        "mvi_seed_strategy": "bounded_random_plus_unrestricted_partner_starts",
        "mode_discovery_indices": [int(i + 1) for i in order],
        # Sequential deflation certifies neither globally lowest ordering
        # nor completeness of the full discrete spectrum.
        "global_spectrum_completeness_verified": False,
        "modal_spectrum_selection": (
            "fourier_restricted_sequential_stationary_candidates"
            if getattr(operator, "circumferential_harmonic", None) is not None
            else "unrestricted_fixed_rank_sequential_stationary_candidates"),
        "mvi_modal_algorithm": (
            "fixed_rank_fourier_restricted_variational_YX_stationarity"
            if getattr(operator, "circumferential_harmonic", None) is not None
            else "fixed_rank_constrained_variational_YX_stationarity"),
        "classical_unrestricted_mvi": (
            getattr(operator, "circumferential_harmonic", None) is None),
        "harmonic_sector_restriction": (
            getattr(operator, "circumferential_harmonic", None) is not None),
    }


__all__ = ("MAX_PROJECTED_DIRECTIONS", "solve_modal_mvi")

