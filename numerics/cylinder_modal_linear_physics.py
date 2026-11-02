"""Shared LINEAR shell Hessian and stiffness action for modal solvers.

This is the theory-specific physical weak tangent, not a numerical eigenvalue
method. Both full-space MKR/FDM and low-rank MVI must use the identical shell
physics while keeping their trial spaces and eigensolvers independent.
The implementation is moved verbatim from cylinder_modal_mvi.py.
"""
from __future__ import annotations

import numpy as np

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
        # Validate the complete boundary passport once. Only end springs
        # depend on displacement in this conservative boundary functional;
        # prescribed tractions/reference offsets have zero tangent.
        from cylinder_variational_boundary import (
            compile_boundary_program, _object, _vector)
        compile_boundary_program(
            functional, np.zeros((nc, *functional.ops.shape)), None,
            spec=problem.spec, calculation_kind=problem.kind,
            check_charge_balance=False)
        self.boundary_diagonal = np.zeros((nc, *functional.ops.shape))
        mechanical_bc = _object({} if problem.spec is None else problem.spec,
                                "mechanical")
        for side, row in (("left", 0), ("right", -1)):
            spring = _vector(_object(mechanical_bc, side).get("spring"),
                             nc, side + " spring")
            self.boundary_diagonal[:, row, :] += (
                spring[:, None] * functional.ops.ws[None, :])

    def action(self, field):
        from cylinder_functional_assembly import _assemble_adjoint
        f = self.f
        jets = f._field_derivatives(field, self.spec)
        values = np.stack(jets, axis=1).reshape(len(self.local), -1)
        weak = (self.local @ values).reshape(f.n_mechanical, len(self.spec), *f.ops.shape)
        result = np.zeros_like(field)
        for d, (_, dx, ds) in enumerate(self.spec):
            result += _assemble_adjoint(weak[:, d], f.W, f.dxp[dx], f.dsp[ds])
        result += self.boundary_diagonal * field
        return result.ravel()

__all__ = ("_kl_linear_local_tangent", "_LocalMechanicalTangent")
