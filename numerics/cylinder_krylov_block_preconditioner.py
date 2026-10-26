#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse positive-block ILU PRECONDITIONER for coupled G3D/Flexo Newton GMRES.

This is a deliberately approximate auxiliary operator, not a change to
the physical Green--Lagrange/flexoelectric/Newmark residual or its Jv.
It includes physical length scales (surface Laplace/G3D biharmonic terms,
director or CUF thickness stiffness, dielectric gradients and Newmark
mass/rate scales), but ignores coupling in the inverse. GMRES still
multiplies the EXACT full indefinite coupled Jacobian-vector product.

The helper retains CSR sparsity: O(nx*ns) difference stencil storage,
no dense physical Jacobian and no full-dimensional block factorization.
The ILU factors are cached for repeated identical thickness orders.

For essential constraints C z=d, use P=I-C.T(CC.T)^-1 C. A left
preconditioner with null-space and constraint-normal splitting acts as
    M^{-1} r = P [block_ILU^{-1}(P r)] + (I-P) r.
It does NOT impose spurious boundary equations or an electric gauge.
Positive shifts only regularize the AUXILIARY preconditioner, NEVER the
original saddle operator.

Numerical quality, acceleration and scalability must be measured
independently; this is not a converged engineering benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from scipy import sparse
from scipy.sparse import linalg as sla
from cylinder_surface_linear_algebra import is_sparse_matrix


@dataclass(frozen=True)
class KrylovPreconditionerDiagnostics:
    method: str
    node_count: int
    variable_count: int
    distinct_ilu_factors: int
    combined_factor_nnz: int
    constrained_rows: int


def _stiffness_matrices(functional):
    """Scalar surface symbols: W, grad^T W grad, Hessian^T W Hessian."""
    ops=functional.ops
    Dx,Ds=ops.Dx,ops.Ds
    if not is_sparse_matrix(Dx) or not is_sparse_matrix(Ds):
        raise ValueError("PHYSICS_BLOCK_ILU_REQUIRES_CSR_SURFACE_OPERATORS")
    nx,ns=ops.shape
    ix=sparse.eye(nx,format="csr")
    js=sparse.eye(ns,format="csr")
    gx=sparse.kron(Dx,js,format="csr")
    gs=sparse.kron(ix,Ds,format="csr")
    weights=np.asarray(ops.area_weights,float).reshape(-1)
    if np.any(weights<=0) or not np.isfinite(weights).all():
        raise ValueError("INVALID_PRECONDITIONER_AREA_WEIGHTS")
    w=sparse.diags(weights,format="csr")
    grad=(gx.T@w@gx+gs.T@w@gs).tocsr()
    gx2=(gx@gx).tocsr()
    gs2=(gs@gs).tocsr()
    gxs=(gx@gs).tocsr()
    # The nonnegative approximate biharmonic energy is constructed by
    # weighted transpose products, NOT a squared nonsymmetric stencil.
    g3d=(gx2.T@w@gx2+2.*gxs.T@w@gxs+gs2.T@w@gs2).tocsr()
    grad.sum_duplicates();g3d.sum_duplicates()
    return w,grad,g3d,weights


def _finite_positive(v,label):
    if isinstance(v,(bool,np.bool_)):
        raise ValueError(label+": expected finite positive scalar")
    try: x=float(v)
    except (TypeError,ValueError) as exc:
        raise ValueError(label+": expected finite positive scalar") from exc
    if not math.isfinite(x) or x<=0:
        raise ValueError(label+": expected finite positive scalar")
    return x


class PhysicsBlockILU:
    """Block-diagonal positive sparse ILU approximate inverse in scaled DOFs."""

    def __init__(self, functional, scales, energy_scale, projector, *,
                 drop_tolerance, fill_factor, max_surface_nodes,
                 dynamic=None):
        E=_finite_positive(functional.material.young_pa,"young_pa")
        mu=_finite_positive(functional.material.shear_modulus,"shear_modulus")
        h=_finite_positive(functional.h,"shell thickness")
        ell=float(functional.material.ell_m)
        if not math.isfinite(ell) or ell<0:
            raise ValueError("invalid G3D elastic length")
        kappa=float(functional.material.kappa_f_per_m)
        if not math.isfinite(kappa) or kappa<0:
            raise ValueError("invalid dielectric coefficient")
        e_scale=_finite_positive(energy_scale,"reference energy scale")
        drop=_finite_positive(drop_tolerance,"ILU drop tolerance")
        fill=_finite_positive(fill_factor,"ILU fill factor")
        if drop>=1 or fill<1:
            raise ValueError("ILU requires drop_tolerance < 1 and fill_factor >= 1")
        nx,ns=functional.ops.shape
        n=nx*ns
        if (isinstance(max_surface_nodes,bool) or
                not isinstance(max_surface_nodes,int) or max_surface_nodes<1 or
                n>max_surface_nodes):
            raise ValueError("BLOCK_ILU_SURFACE_NODE_SAFETY_LIMIT")
        scales=np.asarray(scales,float)
        if scales.shape!=(projector.size,) or not np.isfinite(scales).all() or (
                np.any(scales<=0)):
            raise ValueError("BLOCK_ILU_SCALED_STATE_SHAPE")
        self.projector=projector
        self.blocks=[]
        self.factors={}
        w,grad,g3d,weights=_stiffness_matrices(functional)
        R=_finite_positive(functional.R,"shell radius")
        L=_finite_positive(np.sum(functional.ops.wx),"axial length")

        # Discrete positive symbols with the correct dimensional orders.
        # Regularizing terms are PRECONDITIONER ONLY.
        length_scale=(1./R**2+1./L**2)
        mechanical_base=mu*h
        mech_rate=1.
        mass_shift=0.
        if dynamic is not None:
            dt=_finite_positive(dynamic["dt_s"],"time step")
            beta=_finite_positive(dynamic["beta"],"newmark beta")
            gamma=float(dynamic["gamma"])
            rho=_finite_positive(dynamic["density_kg_m3"],"mass density")
            eta=float(dynamic["eta_s"])
            alpha=float(dynamic["mass_damping_rate_per_s"])
            if not all(math.isfinite(x) and x>=0 for x in (gamma,eta,alpha)):
                raise ValueError("invalid rate/damping in BLOCK_ILU")
            mech_rate+=eta*gamma/(beta*dt)
            mass_shift=(rho*h/(beta*dt**2)
                        +alpha*rho*h*gamma/(beta*dt))
        qfields=functional.n_mechanical
        nphi=projector.size//n-qfields
        if (projector.size%n or nphi<0):
            raise ValueError("BLOCK_ILU_COMPONENT_COUNT_MISMATCH")
        if nphi and kappa<=0:
            raise ValueError("BLOCK_ILU_ELECTRIC_COEFFICIENTS_REQUIRE_KAPPA")
        te_order=int(functional.theory[-1]) if functional.theory.startswith("CUF_TE") else None

        # Each field has its own unknown scaling and polynomial order.
        # Identical (type,order,scale) reuse the same sparse ILU factor.
        for component in range(qfields+nphi):
            if component<qfields:
                if te_order is not None:
                    k=component%(te_order+1)
                    thickness_mass=1./(2*k+1)
                    thickness_shear=(2.*k/h)**2*thickness_mass if k else 0.
                elif functional.theory=="KL":
                    k=0;thickness_mass=1.;thickness_shear=0.
                else:
                    k=1 if component>=3 else 0
                    thickness_mass=(h*h/12. if k else 1.)
                    thickness_shear=(1./h**2 if k else 0.)
                # Approximate a positive shell/strain-gradient symbol.
                stiffness=mechanical_base*mech_rate*thickness_mass
                zeroth=(.02*mechanical_base*length_scale
                        +mass_shift*thickness_mass
                        +mechanical_base*thickness_shear*.06)
                # The 2nd derivative block is deliberately approximate:
                # it preconditions (not replaces) exact covariant G3D.
                operator=(stiffness*grad
                          +stiffness*(ell**2 if functional.material.gradient_model=="G3D"
                                      else 0.)*g3d
                          +zeroth*w)
                kind=("M",k)
            else:
                k=component-qfields
                thickness_mass=1./(2*k+1)
                electric_stiffness=kappa*h*thickness_mass
                zeroth=(.02*electric_stiffness*length_scale
                        +kappa*h*(2.*k/h)**2*thickness_mass if k else
                        .02*electric_stiffness*length_scale)
                operator=electric_stiffness*grad+zeroth*w
                kind=("E",k)
            offset=component*n
            block_scale=scales[offset:offset+n]
            if not np.all(block_scale==block_scale[0]):
                raise ValueError("BLOCK_ILU_REQUIRES_CONSTANT_COMPONENT_SCALING")
            # Re-scale the physical auxiliary symbol to dimensionless
            # nodal coordinates used by the full residual.
            scale=float(block_scale[0])
            operator=(operator*(scale**2/e_scale)).tocsc()
            cache_key=(kind,scale)
            factor=self.factors.get(cache_key)
            if factor is None:
                if not np.isfinite(operator.data).all() or (
                        operator.shape!=(n,n)):
                    raise ValueError("NONFINITE_BLOCK_ILU_OPERATOR")
                try:
                    factor=sla.spilu(operator,drop_tol=drop,
                                    fill_factor=fill,permc_spec="COLAMD")
                except (RuntimeError,ValueError) as exc:
                    raise ValueError(
                        "KRYLOV_BLOCK_ILU_BUILD_FAILED; adjust preconditioner "
                        "parameters or select NONE explicitly") from exc
                self.factors[cache_key]=factor
            self.blocks.append((offset,offset+n,cache_key))
        self.diagnostics=KrylovPreconditionerDiagnostics(
            "PHYSICS_BLOCK_ILU",n,projector.size,len(self.factors),
            sum(int(x.L.nnz+x.U.nnz) for x in self.factors.values()),
            projector.count)

    def apply(self,vector):
        v=np.asarray(vector,float)
        if v.shape!=(self.projector.size,) or not np.isfinite(v).all():
            raise ValueError("NONFINITE_BLOCK_ILU_INPUT")
        tangent=self.projector.project_null(v)
        out=np.empty_like(tangent)
        for a,b,key in self.blocks:
            out[a:b]=self.factors[key].solve(tangent[a:b])
        if not np.isfinite(out).all():
            raise ValueError("NONFINITE_BLOCK_ILU_SOLUTION")
        return self.projector.project_null(out)+(v-tangent)

    def linear_operator(self):
        n=self.projector.size
        return sla.LinearOperator((n,n),matvec=self.apply,dtype=float)


__all__=("PhysicsBlockILU","KrylovPreconditionerDiagnostics")
