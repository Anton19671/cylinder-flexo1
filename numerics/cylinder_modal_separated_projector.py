"""Fast EXACT projected K/M for separated CUF modal Y/X directions.

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


def exact_separated_cuf_projection(problem, density, local_tangent, basis):
    """Return exact (K, M) on current MVI trial basis or None if unsupported.

    Local tangent corresponds to the unchanged first variation of the
    reference shell functional. Only a tensor contraction changes; there
    is no full nodal stiffness/mass matrix and no global eigenvalue solve.
    """
    f = problem.functional
    if (not f.theory.startswith("CUF_TE") or f.electrical_on
            or local_tangent is None or not _spring_free_modal_edges(problem)):
        return None
    ncomp, nx, ns = f.n_mechanical, *f.ops.shape
    basis = np.asarray(basis, dtype=float)
    factors = _one_component_factors(basis, ncomp, nx, ns)
    if factors is None or not np.isfinite(basis).all():
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
    if not np.isfinite(k).all() or not np.isfinite(m).all():
        raise ValueError("MODAL_MVI_SEPARATED_PROJECTION_NONFINITE")
    return k, m


__all__ = ("exact_separated_cuf_projection",)
