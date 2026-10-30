"""Symmetric positive modal preconditioner using sparse physical-order symbols.

It approximates the inverse of the elastic stiffness only. The consistent
variational K, M, essential traces and frequencies remain unchanged.
Unlike an incomplete unsymmetric ILU solve, a full sparse inverse of a
symmetric positive definite auxiliary block is appropriate for LOBPCG.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from cylinder_krylov_block_preconditioner import _stiffness_matrices


class ModalSparseBlockPreconditioner:
    """SPD constrained block inverse with component/thickness-order scaling."""

    def __init__(self, functional, scales, stiffness_scale, projector, *,
                 max_surface_nodes=2500, detail_callback=None):
        nx, ns = functional.ops.shape
        surface_nodes = nx * ns
        if surface_nodes > max_surface_nodes:
            raise ValueError(
                "MODAL_SPARSE_PRECONDITIONER_NODE_LIMIT: use a smaller verification grid")
        component_count = int(functional.n_mechanical)
        if projector.size != surface_nodes * component_count:
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_SHAPE")
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (projector.size,) or not np.isfinite(scales).all() or (
                np.any(scales <= 0.)):
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_SCALES")
        stiffness_scale = float(stiffness_scale)
        if not math.isfinite(stiffness_scale) or stiffness_scale <= 0:
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_STIFFNESS_SCALE")
        e = float(functional.material.young_pa)
        mu = float(functional.material.shear_modulus)
        h = float(functional.h)
        R = float(functional.R)
        length = float(np.sum(functional.ops.wx))
        if not all(math.isfinite(x) and x > 0 for x in (e, mu, h, R, length)):
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_MATERIAL")
        w, grad, g3d, _ = _stiffness_matrices(functional)
        self.projector = projector
        self.count = component_count
        self.node_count = surface_nodes
        self.factors = {}
        self.blocks = []
        te_order = int(functional.theory[-1]) if functional.theory.startswith("CUF_TE") else None
        characteristic = 1. / (length * length) + 1. / (R * R)
        for component in range(component_count):
            if te_order is not None:
                power = component % (te_order + 1)
                radial = (component // (te_order + 1) == 2)
                thickness_weight = 1. / (2 * power + 1)
                through_stiffness = (4. * power * power * e / (
                    (2 * power - 1) * h)) if power else 0.
            else:
                power = 1 if functional.theory in ("FSDT", "REDDY") and component >= 3 else 0
                radial = component == 2
                thickness_weight = (h*h / 12. if power else 1.)
                through_stiffness = (mu / h if power else 0.)
            # These matrices are nonnegative by construction. The strictly
            # positive zeroth term makes every auxiliary block SPD, including
            # the axial alternating SBP mode. No term is added to physical K.
            gradient = mu * h * thickness_weight
            zeroth = through_stiffness + .02 * mu * h * thickness_weight * characteristic
            if radial and power == 0:
                zeroth += .3 * e * h / (R * R)
            bending = e * h**3 / 12. if radial and power == 0 else 0.
            block = (gradient * grad + zeroth * w + bending * g3d)
            offset = component * surface_nodes
            local_scales = scales[offset:offset+surface_nodes]
            if not np.allclose(local_scales, local_scales[0], rtol=0, atol=0):
                raise ValueError("MODAL_SPARSE_PRECONDITIONER_NONUNIFORM_SCALING")
            block_scale = float(local_scales[0] ** 2 / stiffness_scale)
            matrix = (block * block_scale).tocsc()
            if not np.isfinite(matrix.data).all() or (
                    np.min(matrix.diagonal(), initial=np.inf) <= 0.):
                raise ValueError("MODAL_SPARSE_PRECONDITIONER_NONPOSITIVE_BLOCK")
            key = (power, radial, float(local_scales[0]), thickness_weight)
            if key not in self.factors:
                try:
                    lu = sla.splu(matrix, permc_spec="MMD_AT_PLUS_A")
                except (RuntimeError, ValueError) as exc:
                    raise ValueError("MODAL_SPARSE_PRECONDITIONER_FACTORIZATION_FAILED") from exc
                if not np.isfinite(lu.U.diagonal()).all() or (
                        np.min(np.abs(lu.U.diagonal()), initial=0.) <= 0.):
                    raise ValueError("MODAL_SPARSE_PRECONDITIONER_SINGULAR_BLOCK")
                self.factors[key] = lu
            self.blocks.append((offset, offset+surface_nodes, key))
        if detail_callback is not None:
            detail_callback(stage="MATRIX_FREE_MODAL_PRECONDITIONER_READY",
                            method="SPARSE_SPD_BLOCK",
                            surface_nodes=surface_nodes,
                            unique_sparse_factors=len(self.factors))

    def apply(self, vector):
        v = np.asarray(vector, float)
        if v.shape != (self.projector.size,) or not np.isfinite(v).all():
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_BAD_VECTOR")
        pv = self.projector.project_null(v)
        result = np.empty_like(pv)
        for start, end, key in self.blocks:
            result[start:end] = self.factors[key].solve(pv[start:end])
        if not np.isfinite(result).all():
            raise ValueError("MODAL_SPARSE_PRECONDITIONER_NONFINITE_RESULT")
        return self.projector.project_null(result)

    def linear_operator(self):
        n = self.projector.size
        return sla.LinearOperator(
            (n,n),dtype=float,matvec=self.apply,
            matmat=lambda values: np.column_stack([
                self.apply(values[:, j]) for j in range(values.shape[1])]))


__all__ = ("ModalSparseBlockPreconditioner",)
