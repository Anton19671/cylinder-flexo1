#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10A: independent full-grid STRONG-FORM CUF finite-difference solver,
LINEAR ELASTIC.

Unknowns are the complete nodal CUF generalized fields

    q[component,tau,s_index,x_index]

with component = u,v,w and arbitrary supplied thickness-basis size m.
For TE1: m=2 and the nodal fields are [u0,u1,v0,v1,w0,w1].

There is NO separated representation:
- no rank,
- no X/Y profiles,
- no Y->X halfsteps,
- no MVI projection.

Spatial derivatives use the already verified global finite-difference
operators. Same-direction second derivatives use the direct Dxx/Dss
operators rather than the wider D1@D1 composition. Thickness
mechanics/resultants use the already verified Stage7F general linear CUF
implementation.  End conditions use Stage7C
direct strong-residual row replacement.

For LINEAR ELASTIC the discrete residual is affine:

    R(q) = A q + R(0)

so A is assembled exactly from unit nodal coordinates and the global
square system is solved once with np.linalg.solve.

No FEM, no zones, no SVD/pinv/lstsq/QR/Cholesky as solution paths,
no regularization.  A non-SVD 1-norm condition estimate of the equilibrated matrix is available
only as an explicit optional diagnostic and is disabled by default.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import final_numeric_scaling as nscale
import final_residual_scaling as rscale
import final_v3_fd as v3fd
import final_k2_coloring as k2

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7f_general_linear as gl


class CUFStrongFDMError(RuntimeError):
    pass


class CUFStrongFDMLinearSystemSingular(CUFStrongFDMError):
    pass


@dataclass
class CUFStrongFDMLinearResult:
    q: np.ndarray
    converged: bool
    status: str
    ndof: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    boundary_inf: float
    interior_inf: float
    matrix_condition_1: float | None
    matrix_condition_2: float | None
    warnings: tuple[str,...]
    residual_inf_by_block: dict | None = None


def _validate_q(q,ops,basis):
    a=np.asarray(q,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if a.shape!=expected:
        raise ValueError(f"q must have shape {expected}")
    if not np.all(np.isfinite(a)):
        raise ValueError("q contains non-finite values")
    return a


def flatten_q(q,ops,basis):
    return _validate_q(q,ops,basis).ravel()


def unflatten_q(vec,ops,basis):
    x=np.asarray(vec,float)
    n=3*basis.m*ops.grid.ns*ops.grid.nx
    if x.shape!=(n,):
        raise ValueError(f"global CUF vector must have shape ({n},)")
    return x.reshape(3,basis.m,ops.grid.ns,ops.grid.nx).copy()


def surface_first_derivatives(q,ops,basis):
    q=_validate_q(q,ops,basis)
    qx=np.empty_like(q)
    qs=np.empty_like(q)
    for a in range(3):
        for tau in range(basis.m):
            qx[a,tau]=ops.Dx(q[a,tau])
            qs[a,tau]=ops.Ds(q[a,tau])
    return qx,qs


def surface_derivatives_through_second(q,ops,basis):
    """Return first and direct second surface derivatives of the CUF fields.

    Same-direction second derivatives deliberately use the verified direct
    operators Dxx/Dss. Reapplying Dx/Ds to a first derivative would create
    the wider D1@D1 stencil and, for the 3-point stencil, odd/even decoupling.
    """
    q=_validate_q(q,ops,basis)
    qx,qs=surface_first_derivatives(q,ops,basis)
    qxx=np.empty_like(q)
    qss=np.empty_like(q)
    qxs=np.empty_like(q)
    for a in range(3):
        for tau in range(basis.m):
            qxx[a,tau]=ops.Dxx(q[a,tau])
            qss[a,tau]=ops.Dss(q[a,tau])
            qxs[a,tau]=ops.Ds(qx[a,tau])
    return qx,qs,qxx,qss,qxs


def linear_elastic_consistent_raw_residual(
    q,ops,basis,*,E0,nu0,quadrature,generalized_load
):
    """Linear CUF strong residual with direct same-direction D2 operators.

    For constant elastic coefficients the linear generalized resultants are a
    linear function of (q,q_x,q_s). Therefore their surface derivatives are
    obtained exactly by evaluating the same resultant map on
    (q_x,q_xx,q_xs) and (q_s,q_xs,q_ss). This keeps the physical flux form
    while avoiding the discrete D1(D1(q)) composition.
    """
    q=_validate_q(q,ops,basis)
    p=np.asarray(generalized_load,float)
    if p.shape!=q.shape:
        raise ValueError(f"generalized_load must have shape {q.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    qx,qs,qxx,qss,qxs=surface_derivatives_through_second(q,ops,basis)
    ec=gl.lk.ElasticConstants(float(E0),float(nu0))
    kw=dict(
        R=ops.grid.radius,
        K0=ec.K0,
        GB=ec.G0,
        quadrature=quadrature,
        basis=basis,
    )
    resultants=gl.general_linear_thickness_resultants(q,qx,qs,**kw)
    dx_resultants=gl.general_linear_thickness_resultants(qx,qxx,qxs,**kw)
    ds_resultants=gl.general_linear_thickness_resultants(qs,qxs,qss,**kw)

    raw=np.empty_like(p)
    for tau in range(basis.m):
        raw[0,tau]=(
            -dx_resultants.Au_x[tau]
            -ds_resultants.Au_s[tau]
            +resultants.Cu[tau]
            -p[0,tau]
        )
        raw[1,tau]=(
            -dx_resultants.Av_x[tau]
            -ds_resultants.Av_s[tau]
            +resultants.Cv[tau]
            -p[1,tau]
        )
        raw[2,tau]=(
            -dx_resultants.Aw_x[tau]
            -ds_resultants.Aw_s[tau]
            +resultants.Cw[tau]
            -p[2,tau]
        )
    return raw,resultants,qx,qs


def linear_elastic_fullgrid_residual(
    q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load
):
    q=_validate_q(q,ops,basis)
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    expected=q.shape
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    raw,resultants,qx,qs=linear_elastic_consistent_raw_residual(
        q,ops,basis,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=p,
    )

    assembled=g9.assemble_strong_residual_with_end_bcs(
        raw,q,resultants,bc,reject_fully_free=True
    )

    b=g9.full_boundary_residuals(q,resultants,bc)
    boundary=np.concatenate([
        np.asarray(b["left"],float).ravel(),
        np.asarray(b["right"],float).ravel(),
    ])

    if ops.grid.nx>2:
        interior=np.asarray(raw[:,:,:,1:-1],float).ravel()
    else:
        interior=np.empty(0,float)

    return (
        assembled.reshape(-1),
        raw,
        resultants,
        qx,
        qs,
        boundary,
        interior,
    )


def linear_rhs_only(
    ops,basis,bc,*,E0,nu0,quadrature,generalized_load
):
    """Return the affine residual r0=R(q=0) for the supplied load.

    The LINEAR operator is load-independent; this cheap helper lets a load
    series reuse one assembled matrix while recomputing only its right-hand
    side.
    """
    q0=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    r0,*_=linear_elastic_fullgrid_residual(
        q0,ops,basis,bc,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    return r0


def _assemble_linear_operator_and_rhs_dense_reference(
    ops,basis,bc,*,E0,nu0,quadrature,generalized_load
):
    """Pre-K2 one-column assembly retained only for regression self-checks."""
    n=3*basis.m*ops.grid.ns*ops.grid.nx
    r0=linear_rhs_only(
        ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    A=np.empty((n,n),float)
    for j in range(n):
        e=np.zeros(n,float)
        e[j]=1.0
        qj=unflatten_q(e,ops,basis)
        rj,*_=linear_elastic_fullgrid_residual(
            qj,ops,basis,bc,
            E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        A[:,j]=rj-r0
    return A,r0


def assemble_linear_operator_and_rhs(
    ops,basis,bc,*,E0,nu0,quadrature,generalized_load
):
    """K2 safe colored assembly of the same dense affine CUF operator."""
    q0=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
    r0=linear_rhs_only(
        ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )

    def evaluate(q):
        r,*_=linear_elastic_fullgrid_residual(
            q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        return r

    A,_,stats=k2.assemble_cuf_colored_columns(
        q0,ops,evaluate,base_residual=r0,step=1.0,dependency_layers=2
    )
    assemble_linear_operator_and_rhs.last_k2_stats=stats.to_dict()
    return A,r0


assemble_linear_operator_and_rhs.last_k2_stats=None


def solve_from_operator(
    A,r0,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,
    linear_residual_tolerance_relative=1.0e-10,
    compute_condition_number=False
):
    A=np.asarray(A,float)
    r0=np.asarray(r0,float)
    n=3*basis.m*ops.grid.ns*ops.grid.nx
    if A.shape!=(n,n) or r0.shape!=(n,):
        raise ValueError("A/r0 dimension mismatch")

    if np.max(np.abs(r0))==0.0:
        q=np.zeros((3,basis.m,ops.grid.ns,ops.grid.nx),float)
        try:
            As,_,_=nscale.equilibrate_matrix(A)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMLinearSystemSingular(
                "SINGULAR_CUF_STRONG_FDM_LINEAR_SYSTEM"
            ) from exc
    else:
        try:
            x,As,_,_=nscale.equilibrated_solve(A,-r0)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMLinearSystemSingular(
                "SINGULAR_CUF_STRONG_FDM_LINEAR_SYSTEM"
            ) from exc
        q=unflatten_q(x,ops,basis)

    R,_raw,_res,_qx,_qs,_boundary,_interior=linear_elastic_fullgrid_residual(
        q,ops,basis,bc,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )

    # K-2 + V-4 reviewer fixes.  The direct solve still uses the original
    # physical system.  Only the residual MEASURE is left-scaled here.
    h_v4=v3fd.thickness_from_quadrature(quadrature)
    Rm=rscale.cuf_scaled_residual_for_measure(
        R,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h_v4,
    )
    r0m=rscale.cuf_scaled_residual_for_measure(
        r0,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h_v4,
    )
    pm=rscale.cuf_scaled_strong_load_for_measure(
        generalized_load,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,
    )
    residual_inf=float(np.max(np.abs(Rm)))
    scale=rscale.residual_scale(pm,r0m)
    limit=rscale.threshold(linear_residual_tolerance_relative,scale)
    if not (residual_inf<=limit):
        raise CUFStrongFDMLinearSystemSingular(
            "LINEAR_SOLVE_RESIDUAL_NOT_SATISFIED "
            f"residual_inf={residual_inf:.6e} limit={limit:.6e} "
            f"scale={scale:.6e}"
        )

    # Reviewer-requested non-SVD diagnostic on the equilibrated matrix that was
    # actually solved.  This does not alter the solution or provide a fallback.
    cond1=None
    warnings=[]
    if bool(compute_condition_number):
        cond1=float(np.linalg.cond(As,1))
        if cond1>1.0e12:
            warnings.append(
                f"ILL_CONDITIONED_LINEAR_OPERATOR cond_1(As)={cond1:.3e}. "
                "Solution may be dominated by round-off."
            )

    return CUFStrongFDMLinearResult(
        q=q,
        converged=True,
        status="CONVERGED_CUF_STRONG_FDM_LINEAR_ELASTIC",
        ndof=n,
        residual_inf=residual_inf,
        residual_relative=rscale.relative_residual(residual_inf,scale),
        residual_scale=scale,
        boundary_inf=float(np.max(np.abs(Rm.reshape(3,basis.m,ops.grid.ns,ops.grid.nx)[...,[0,-1]]))),
        interior_inf=(
            float(np.max(np.abs(Rm.reshape(3,basis.m,ops.grid.ns,ops.grid.nx)[...,1:-1])))
            if ops.grid.nx>2 else 0.0
        ),
        matrix_condition_1=cond1,
        matrix_condition_2=None,
        warnings=tuple(warnings),
        residual_inf_by_block=rscale.cuf_residual_inf_by_block(
            Rm,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
        ),
    )


def solve_linear_elastic_strong_fdm(
    ops,basis,bc,*,E0,nu0,quadrature,generalized_load,
    linear_residual_tolerance_relative=1.0e-10,
    compute_condition_number=False
):
    A,r0=assemble_linear_operator_and_rhs(
        ops,basis,bc,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    return solve_from_operator(
        A,r0,ops,basis,bc,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
        linear_residual_tolerance_relative=linear_residual_tolerance_relative,
        compute_condition_number=compute_condition_number,
    )
