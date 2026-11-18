"""Fast EXACT projected K/M for separated CUF/director modal Y/X directions.

This contracts the unchanged physical local weak Hessian with the true
one-dimensional X/Y shape factors. It does NOT solve a full-space eigenproblem,
change the MVI rank, restrict Fourier sectors, or introduce Ritz nodal modes.

For uniform circular shells W(x,s)=wx(x)*ws(s). Each current frozen-partner
test column is one component times an axial profile times an angular profile.
Consequently the current MVI projected bilinear forms factor into 1D weighted
derivative inner products. The formula is exactly B.T@K@B and B.T@M@B.
Unsupported fields, load-dependent springs, mixed columns, or nonseparable
quadrature revert to the original weak-action column assembly.
"""
from __future__ import annotations

import numpy as np

from cuf_g3d_weak import mass_matrix_te
from cylinder_functional_assembly import MECHANICAL_DERIVATIVES


def _one_component_factors(basis, ncomp, nx, ns):
    """Extract exact rank-one columns, vectorized without rank approximation.

    Identical algebra to the former per-column loop: one mechanical
    component and one separable X/Y profile per existing frozen basis column.
    Reject mixed or nonseparable vectors rather than modifying the MVI space.
    """
    field = np.asarray(basis, float)
    if field.ndim != 2 or field.shape[0] != ncomp * nx * ns:
        return None
    count = field.shape[1]
    if not count:
        return (np.empty(0, dtype=int),
                np.empty((0, nx)), np.empty((0, ns)))
    values = field.T.reshape(count, ncomp, nx, ns)
    norms = np.linalg.norm(values, axis=(2, 3))
    components = np.argmax(norms, axis=1)
    rows = np.arange(count)
    selected_norms = norms[rows, components]
    if not np.isfinite(norms).all() or np.any(selected_norms <= 0.):
        return None
    other_norms = norms.copy()
    other_norms[rows, components] = 0.
    if np.any(np.max(other_norms, axis=1) > 1.e-12 * selected_norms):
        return None
    profiles = values[rows, components]
    index = np.argmax(np.abs(profiles.reshape(count, nx * ns)), axis=1)
    ix, is_ = np.divmod(index, ns)
    pivots = profiles[rows, ix, is_]
    if np.any(pivots == 0.):
        return None
    axial = profiles[
        rows[:, None], np.arange(nx)[None, :], is_[:, None]]
    angular = (profiles[
        rows[:, None], ix[:, None], np.arange(ns)[None, :]] /
        pivots[:, None])
    difference = np.linalg.norm(
        profiles - axial[:, :, None] * angular[:, None, :], axis=(1, 2))
    if (not np.isfinite(difference).all()
            or np.any(difference > 1.e-10 *
                      np.maximum(selected_norms, 1.e-30))):
        return None
    return components, axial, angular

def _spring_free_modal_edges(problem):
    """Conservative spring Hessians are not part of the bulk local Hessian.

    Leave any case with an end spring to the original exact boundary action.
    Dead prescribed tractions have zero Hessian, so do not alter this rule.
    """
    spec = getattr(problem, "spec", None)
    if not isinstance(spec, dict):
        return False
    mechanical = spec.get("mechanical", {})
    if not isinstance(mechanical, dict) or set(mechanical) - {"left", "right"}:
        return False
    for edge in mechanical.values():
        if not isinstance(edge, dict):
            return False
        spring = edge.get("spring")
        if spring is not None:
            try:
                if np.any(np.asarray(spring, dtype=float) != 0.):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _director_inertial_jet_matrix(functional, density, local_tangent):
    """Exact thickness-integrated inertia of FSDT/Reddy surface jets.

    The director velocity depends on q, q_x and q_s. Preserve all
    derivative inertia and curved thickness moments (1 + z/R).
    The local coefficients are independent of the MVI rank and factors.
    """
    key = (functional.theory, float(functional.R), float(functional.h),
           float(density), int(functional.n_gauss),
           int(functional.n_derivatives))
    cached = getattr(local_tangent, "_separated_director_inertia", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if (functional.theory not in ("FSDT", "REDDY") or
            functional.n_mechanical != 5 or functional.n_derivatives < 3):
        return None
    from cylinder_director_g3d import _director_virtual_velocity
    from cuf_g3d_weak import gauss_thickness

    nd = int(functional.n_derivatives)
    mass15 = np.zeros((15, 15), dtype=float)
    for z, weight in zip(*gauss_thickness(functional.h,
                                         functional.n_gauss)):
        virtual = _director_virtual_velocity(
            functional.theory, float(z),
            float(functional.R), float(functional.h))
        # Original library indexing: (derivative, component, Cartesian).
        # Weak modal Hessian indexing: (component, derivative).
        library = np.transpose(virtual, (1, 0, 2)).reshape(15, 3)
        mass15 += (float(density) * float(weight) *
                   (1. + float(z) / functional.R) *
                   (library @ library.T))
    local_mass = np.zeros((5 * nd, 5 * nd), dtype=float)
    indices = [c * nd + d for c in range(5) for d in range(3)]
    local_mass[np.ix_(indices, indices)] = mass15
    if not np.isfinite(local_mass).all():
        raise ValueError("MODAL_MVI_DIRECTOR_INERTIA_NONFINITE")
    local_tangent._separated_director_inertia = (key, local_mass)
    return local_mass


def _kl_inertial_jet_matrix(functional, density, local_tangent):
    """Exact reference-state KL inertia in the surface-jet ordering."""
    key = ("KL", float(functional.R), float(functional.h), float(density),
           str(functional.geometry_model), int(functional.n_derivatives))
    cached = getattr(local_tangent, "_separated_kl_inertia", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if functional.theory != "KL" or functional.n_mechanical != 3:
        return None
    from cylinder_kl_exact_inertia import kl_inertial_coefficients_from_jets

    nd = int(functional.n_derivatives)
    if nd < 3:
        return None
    zero = np.zeros((3, 1, 1), dtype=float)
    local9 = np.empty((9, 9), dtype=float)
    for column in range(9):
        component, derivative = divmod(column, 3)
        acceleration = [zero.copy(), zero.copy(), zero.copy()]
        acceleration[derivative][component, 0, 0] = 1.
        coefficients = kl_inertial_coefficients_from_jets(
            (zero, zero, zero), (zero, zero, zero), tuple(acceleration),
            radius_m=functional.R, thickness_m=functional.h,
            density_kg_m3=density, geometry_model=functional.geometry_model)
        # The physical library returns (derivative, component); modal weak
        # jet storage is (component, derivative).
        local9[:, column] = np.transpose(coefficients[:, :, 0, 0],
                                         (1, 0)).reshape(9)
    local_mass = np.zeros((3 * nd, 3 * nd), dtype=float)
    indices = [component * nd + derivative
               for component in range(3) for derivative in range(3)]
    local_mass[np.ix_(indices, indices)] = local9
    if not np.isfinite(local_mass).all():
        raise ValueError("MODAL_MVI_KL_INERTIA_NONFINITE")
    local_tangent._separated_kl_inertia = (key, local_mass)
    return local_mass


def exact_linear_inertial_jet_matrix(functional, density, local_tangent):
    """Return the exact reference-state KL/FSDT/Reddy inertia matrix."""
    if functional.theory == "KL":
        return _kl_inertial_jet_matrix(functional, density, local_tangent)
    return _director_inertial_jet_matrix(functional, density, local_tangent)


def _cuf_inertial_jet_matrix(functional, density, local_tangent):
    """Exact CUF inertia in the common (component, jet) storage format."""
    degree = int(functional.theory[-1])
    key = (functional.theory, float(functional.R), float(functional.h),
           float(density), int(functional.n_derivatives))
    cached = getattr(local_tangent, "_separated_cuf_inertia", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    ncomp, nd = functional.n_mechanical, int(functional.n_derivatives)
    width = degree + 1
    if ncomp != 3 * width or nd < 1:
        return None
    thickness_mass = mass_matrix_te(
        radius_m=functional.R, thickness_m=functional.h,
        rho_kg_m3=float(density), te_order=degree)
    component_mass = np.zeros((ncomp, ncomp), dtype=float)
    for component in range(3):
        sl = slice(component * width, (component + 1) * width)
        component_mass[sl, sl] = thickness_mass
    local_mass = np.zeros((ncomp * nd, ncomp * nd), dtype=float)
    values = np.arange(ncomp) * nd
    local_mass[np.ix_(values, values)] = component_mass
    if not np.isfinite(local_mass).all():
        raise ValueError("MODAL_MVI_CUF_INERTIA_NONFINITE")
    local_tangent._separated_cuf_inertia = (key, local_mass)
    return local_mass


def exact_inertial_jet_matrix(functional, density, local_tangent):
    """Exact reference inertia in the common weak surface-jet ordering."""
    if functional.theory.startswith("CUF_TE"):
        return _cuf_inertial_jet_matrix(functional, density, local_tangent)
    return exact_linear_inertial_jet_matrix(functional, density, local_tangent)



def _exact_batched_projection(problem, density, local_tangent, basis):
    """Exact B.T K B and B.T M B even for trace-mixed fixed-rank columns.

    Boundary-intersection QR may form a linear combination of several
    rank-one X/Y directions. Such a column need not itself be rank-one.
    Apply the original physical LOCAL jet Hessians in small batches, not
    one Python weak-action invocation per nodal column. No rank truncation,
    Fourier restriction, or change of the constrained trial space.
    """
    from cylinder_functional_assembly import _assemble_adjoint

    f = problem.functional
    ncomp, nx, ns = f.n_mechanical, *f.ops.shape
    nd = int(f.n_derivatives)
    spec = MECHANICAL_DERIVATIVES[:nd]
    basis = np.asarray(basis, dtype=float)
    if (basis.ndim != 2 or basis.shape[0] != ncomp * nx * ns or
            not np.isfinite(basis).all() or not basis.shape[1]):
        return None
    stiffness_local = np.asarray(local_tangent.local, dtype=float)
    inertia_local = exact_inertial_jet_matrix(
        f, density, local_tangent)
    if (inertia_local is None or
            stiffness_local.shape != (ncomp * nd, ncomp * nd)):
        return None
    ncols = basis.shape[1]
    k = np.empty((ncols, ncols), dtype=float)
    m = np.empty((ncols, ncols), dtype=float)
    # Keep peak temporary jet memory bounded across all ranks/grid sizes.
    batch = max(1, min(32, 1048576 // (ncomp * nd * nx * ns)))
    for start in range(0, ncols, batch):
        stop = min(start + batch, ncols)
        n = stop - start
        raw = basis[:, start:stop].T.reshape(n * ncomp, nx, ns)
        jets = np.stack([
            derivative.reshape(n, ncomp, nx, ns)
            for derivative in f._field_derivatives(raw, spec)
        ], axis=2)
        flattened = jets.reshape(n, ncomp * nd, nx * ns)
        k_local = (stiffness_local @ flattened).reshape(
            n, ncomp, nd, nx, ns)
        m_local = (inertia_local @ flattened).reshape(
            n, ncomp, nd, nx, ns)
        k_action = np.zeros((n, ncomp, nx, ns))
        m_action = np.zeros_like(k_action)
        for d, (_, dx, ds) in enumerate(spec):
            k_action += _assemble_adjoint(
                k_local[:, :, d].reshape(n * ncomp, nx, ns),
                f.W, f.dxp[dx], f.dsp[ds]).reshape(
                    n, ncomp, nx, ns)
            if d < 3:
                m_action += _assemble_adjoint(
                    m_local[:, :, d].reshape(n * ncomp, nx, ns),
                    f.W, f.dxp[dx], f.dsp[ds]).reshape(
                        n, ncomp, nx, ns)
        # In this spring-free branch, the mechanical boundary Hessian
        # vanishes, exactly as in the reference columnwise weak action.
        k[:, start:stop] = basis.T @ k_action.reshape(n, -1).T
        m[:, start:stop] = basis.T @ m_action.reshape(n, -1).T
    if not np.isfinite(k).all() or not np.isfinite(m).all():
        raise ValueError("MODAL_MVI_BATCHED_DIRECTOR_PROJECTION_NONFINITE")
    return k, m

def exact_separated_cuf_projection(problem, density, local_tangent, basis):
    """Return exact (K, M) on current MVI trial basis or None if unsupported.

    Local tangent corresponds to the unchanged first variation of the
    reference shell functional. Only a tensor contraction changes; there
    is no full nodal stiffness/mass matrix and no global eigenvalue solve.
    """
    f = problem.functional
    if (not (f.theory.startswith("CUF_TE") or
             f.theory in ("KL", "FSDT", "REDDY")) or f.electrical_on
            or local_tangent is None or not _spring_free_modal_edges(problem)):
        return None
    ncomp, nx, ns = f.n_mechanical, *f.ops.shape
    basis = np.asarray(basis, dtype=float)
    factors = _one_component_factors(basis, ncomp, nx, ns)
    if not np.isfinite(basis).all():
        return None
    if factors is None:
        if f.theory.startswith("CUF_TE") or f.theory in ("KL", "FSDT", "REDDY"):
            return _exact_batched_projection(
                problem, density, local_tangent, basis)
        return None
    wx, ws = np.asarray(f.ops.wx, float), np.asarray(f.ops.ws, float)
    if (wx.shape != (nx,) or ws.shape != (ns,)
            or not np.allclose(f.W, wx[:, None] * ws[None, :],
                               rtol=1.e-13, atol=0.)):
        return None
    components, axial, angular = factors
    nd = int(f.n_derivatives)
    spec = MECHANICAL_DERIVATIVES[:nd]
    local = np.asarray(local_tangent.local, float)
    if local.shape != (ncomp * nd, ncomp * nd):
        return None
    powers_x = sorted(set(dx for _, dx, _ in spec))
    powers_s = sorted(set(ds for _, _, ds in spec))
    ax_derivs = {p: axial @ f.dxp[p].T for p in powers_x}
    an_derivs = {p: angular @ f.dsp[p].T for p in powers_s}
    # Reused 1D Gram matrices avoid any repeated 2D weak-action assembly.
    ax_grams = {
        (a, b): (ax_derivs[a] * wx[None, :]) @ ax_derivs[b].T
        for a in powers_x for b in powers_x}
    an_grams = {
        (a, b): (an_derivs[a] * ws[None, :]) @ an_derivs[b].T
        for a in powers_s for b in powers_s}
    count = basis.shape[1]
    k = np.zeros((count, count), dtype=float)
    for d, (_, dx1, ds1) in enumerate(spec):
        left = components * nd + d
        for e, (_, dx2, ds2) in enumerate(spec):
            right = components * nd + e
            local_coefficient = local[np.ix_(left, right)]
            k += (local_coefficient * ax_grams[dx1, dx2] *
                  an_grams[ds1, ds2])
    if f.theory.startswith("CUF_TE"):
        degree = int(f.theory[-1])
        width = degree + 1
        thickness_mass = mass_matrix_te(
            radius_m=f.R, thickness_m=f.h,
            rho_kg_m3=float(density), te_order=degree)
        mass_coefficient = np.zeros((ncomp, ncomp), dtype=float)
        for component in range(3):
            sl = slice(component * width, (component + 1) * width)
            mass_coefficient[sl, sl] = thickness_mass
        m = (mass_coefficient[np.ix_(components, components)] *
             ax_grams[0, 0] * an_grams[0, 0])
    else:
        # Reddy's gradient-dependent inertia is retained exactly: no
        # replacement with an approximate five-component mass matrix.
        inertia = exact_linear_inertial_jet_matrix(
            f, density, local_tangent)
        if inertia is None:
            return None
        m = np.zeros((count, count), dtype=float)
        for d in range(3):
            _, dx1, ds1 = spec[d]
            left = components * nd + d
            for e in range(3):
                _, dx2, ds2 = spec[e]
                right = components * nd + e
                m += (inertia[np.ix_(left, right)] *
                      ax_grams[dx1, dx2] * an_grams[ds1, ds2])
    if not np.isfinite(k).all() or not np.isfinite(m).all():
        raise ValueError("MODAL_MVI_SEPARATED_PROJECTION_NONFINITE")
    return k, m


__all__ = ("exact_separated_cuf_projection",)
