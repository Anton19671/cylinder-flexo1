#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9B: independent STRONG-FORM KL finite-difference/collocation solver,
LINEAR ELASTIC only.

Unknowns are all nodal values:
    U = [u(s_j,x_i), v(s_j,x_i), w(s_j,x_i)]

No separated representation, no rank, no X/Y profiles, no MVI,
no weak projected residual.

Discrete equation count per circumferential node:
u: (nx-2) interior strong equations + 2 end BC = nx
v: (nx-2) interior strong equations + 2 end BC = nx
w: (nx-4) interior strong equations + 4 end BC = nx

For w the interior strong equation is collocated at x indices 2..nx-3.
Boundary equations are taken directly from the verified KL end mechanics:
u/v geometric or Tu/Tv natural,
w geometric or Tw natural,
theta_n geometric or T_theta natural.

LINEAR ELASTIC residual is affine. The global matrix is assembled exactly
from unit nodal coordinates and solved once with np.linalg.solve.
A non-SVD 1-norm condition estimate of the equilibrated matrix is available only
as an explicit optional diagnostic and is disabled by default.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import final_numeric_scaling as nscale
import final_residual_scaling as rscale

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b1_boundary as kb


class KLStrongFDMError(RuntimeError):
    pass


class KLStrongFDMLinearSystemSingular(KLStrongFDMError):
    pass


@dataclass
class KLStrongFDMLinearResult:
    U: np.ndarray
    converged: bool
    status: str
    ndof: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    interior_strong_inf: float
    boundary_inf: float
    matrix_condition_1: float | None
    matrix_condition_2: float | None
    warnings: tuple[str,...]


def _validate_grid(ops):
    if ops.grid.nx < 7:
        raise ValueError("verified 7-point strong FDM requires nx>=7")
    if ops.grid.ns < 7:
        raise ValueError("verified periodic 7-point strong FDM requires ns>=7")


def _validate_U(U,ops):
    a=np.asarray(U,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if a.shape!=expected:
        raise ValueError(f"U must have shape {expected}")
    if not np.all(np.isfinite(a)):
        raise ValueError("U contains non-finite values")
    return a


def flatten_U(U,ops):
    return _validate_U(U,ops).ravel()


def unflatten_U(vec,ops):
    x=np.asarray(vec,float)
    n=3*ops.grid.ns*ops.grid.nx
    if x.shape!=(n,):
        raise ValueError(f"global vector must have shape ({n},)")
    return x.reshape(3,ops.grid.ns,ops.grid.nx).copy()


def _resultants_and_kinematics(
    U,ops,*,E,nu,h,resultants_method,quadrature
):
    kin=kl.field_kinematics(U,ops,geometry="LINEAR")
    const=kl.KLElasticConstants(E,nu,h)
    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        nm=kl.analytic_elastic_resultants(kin.state,const)
    elif method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required for QUADRATURE resultants")
        nm=kl.quadrature_elastic_resultants(kin.state,const,quadrature)
    else:
        raise ValueError("resultants_method must be ANALYTIC or QUADRATURE")
    return nm,kin


def _bc_component_residual(cond,geom,natural,ns):
    target=cond.data.array(ns)
    if cond.kind=="GEOMETRIC":
        return np.asarray(geom,float)-target
    if cond.kind=="NATURAL":
        return np.asarray(natural,float)-target
    if cond.kind=="ELASTIC":
        return (np.asarray(natural,float)
                +cond.stiffness_array(ns)*(np.asarray(geom,float)-cond.reference_array(ns))
                -target)
    raise ValueError(f"unsupported KL boundary kind: {cond.kind}")


def strong_collocation_residual(
    U,ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    """
    Return one square residual vector of length 3*ns*nx.

    Ordering:
      all u equations by s-row,
      all v equations by s-row,
      all w equations by s-row.

    For each s-row:
      u: left BC, strong Ru at x=1..nx-2, right BC
      v: left BC, strong Rv at x=1..nx-2, right BC
      w: left w-BC, left slope-BC,
         strong Rw at x=2..nx-3,
         right slope-BC, right w-BC

    The order of the two right w equations is immaterial physically;
    this fixed ordering only defines the global row layout.
    """
    _validate_grid(ops)
    U=_validate_U(U,ops)
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    nm,kin=_resultants_and_kinematics(
        U,ops,E=E,nu=nu,h=h,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )
    strong=kl.strong_residual_from_resultants(
        kin,nm,ops,geometry="LINEAR",generalized_load=p
    )

    ql=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="LINEAR",side="LEFT"
    )
    qr=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="LINEAR",side="RIGHT"
    )

    ns,nx=ops.grid.ns,ops.grid.nx

    lu=_bc_component_residual(bc.left.u,ql.u,ql.T_u,ns)
    ru=_bc_component_residual(bc.right.u,qr.u,qr.T_u,ns)
    lv=_bc_component_residual(bc.left.v,ql.v,ql.T_v,ns)
    rv=_bc_component_residual(bc.right.v,qr.v,qr.T_v,ns)

    lw=_bc_component_residual(bc.left.w,ql.w,ql.T_w,ns)
    rw=_bc_component_residual(bc.right.w,qr.w,qr.T_w,ns)
    ls=_bc_component_residual(
        bc.left.slope,ql.theta_n,ql.T_theta,ns
    )
    rs=_bc_component_residual(
        bc.right.slope,qr.theta_n,qr.T_theta,ns
    )

    rows=[]

    # u block
    for js in range(ns):
        rows.append(lu[js])
        rows.extend(strong[0,js,1:nx-1])
        rows.append(ru[js])

    # v block
    for js in range(ns):
        rows.append(lv[js])
        rows.extend(strong[1,js,1:nx-1])
        rows.append(rv[js])

    # w block
    for js in range(ns):
        rows.append(lw[js])
        rows.append(ls[js])
        rows.extend(strong[2,js,2:nx-2])
        rows.append(rs[js])
        rows.append(rw[js])

    R=np.asarray(rows,float)
    expected_n=3*ns*nx
    if R.shape!=(expected_n,):
        raise KLStrongFDMError(
            f"internal residual size {R.shape}, expected ({expected_n},)"
        )

    bvals=np.concatenate([lu,ru,lv,rv,lw,rw,ls,rs])
    interior=np.concatenate([
        strong[0,:,1:nx-1].ravel(),
        strong[1,:,1:nx-1].ravel(),
        strong[2,:,2:nx-2].ravel(),
    ])
    return R,strong,bvals,interior,nm,kin


def linear_rhs_only(
    ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    """Return the affine residual r0=R(U=0) for the supplied load.

    The LINEAR KL operator is load-independent, so a load series can reuse
    one assembled matrix and recompute only this right-hand side.
    """
    U0=np.zeros((3,ops.grid.ns,ops.grid.nx),float)
    r0,*_=strong_collocation_residual(
        U0,ops,bc,E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )
    return r0


def assemble_linear_strong_system(
    ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    n=3*ops.grid.ns*ops.grid.nx
    r0=linear_rhs_only(
        ops,bc,E=E,nu=nu,h=h,generalized_load=generalized_load,
        resultants_method=resultants_method,quadrature=quadrature,
    )

    A=np.empty((n,n),float)
    for j in range(n):
        ej=np.zeros(n,float)
        ej[j]=1.0
        Uj=unflatten_U(ej,ops)
        rj,*_=strong_collocation_residual(
            Uj,ops,bc,E=E,nu=nu,h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method,
            quadrature=quadrature,
        )
        A[:,j]=rj-r0

    return A,r0


def solve_from_operator(
    A,r0,ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None,
    linear_residual_tolerance_relative=1.0e-10,
    compute_condition_number=False
):
    """Solve a LINEAR KL load case using a preassembled operator."""
    bc.check_rigid_mode_requirement()
    A=np.asarray(A,float)
    r0=np.asarray(r0,float)
    n=3*ops.grid.ns*ops.grid.nx
    if A.shape!=(n,n) or r0.shape!=(n,):
        raise ValueError("A/r0 dimension mismatch")

    try:
        x,As,_,_=nscale.equilibrated_solve(A,-r0)
    except np.linalg.LinAlgError as exc:
        raise KLStrongFDMLinearSystemSingular(
            "SINGULAR_STRONG_FDM_LINEAR_SYSTEM"
        ) from exc

    U=unflatten_U(x,ops)
    R,_strong,bvals,interior,_,_=strong_collocation_residual(
        U,ops,bc,E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )

    # K-2 reviewer fix: verify that the direct solution actually satisfies the
    # discrete equations using the existing V2 dimensionless residual scale.
    residual_inf=float(np.max(np.abs(R)))
    scale=rscale.residual_scale(generalized_load,r0)
    limit=rscale.threshold(linear_residual_tolerance_relative,scale)
    if not (residual_inf<=limit):
        raise KLStrongFDMLinearSystemSingular(
            "LINEAR_SOLVE_RESIDUAL_NOT_SATISFIED "
            f"residual_inf={residual_inf:.6e} limit={limit:.6e} "
            f"scale={scale:.6e}"
        )

    # Reviewer-requested 1-norm condition diagnostic on the equilibrated matrix.
    # p=1 does not use the SVD and it does not change or regularize the solve.
    cond1=None
    warnings=[]
    if bool(compute_condition_number):
        cond1=float(np.linalg.cond(As,1))
        if cond1>1.0e12:
            warnings.append(
                f"ILL_CONDITIONED_LINEAR_OPERATOR cond_1(As)={cond1:.3e}. "
                "Solution may be dominated by round-off."
            )

    return KLStrongFDMLinearResult(
        U=U,
        converged=True,
        status="CONVERGED_STRONG_FDM_LINEAR",
        ndof=A.shape[0],
        residual_inf=residual_inf,
        residual_relative=rscale.relative_residual(residual_inf,scale),
        residual_scale=scale,
        interior_strong_inf=float(np.max(np.abs(interior))),
        boundary_inf=float(np.max(np.abs(bvals))),
        matrix_condition_1=cond1,
        matrix_condition_2=None,
        warnings=tuple(warnings),
    )


def solve_linear_elastic_strong_fdm(
    ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None,
    linear_residual_tolerance_relative=1.0e-10,
    compute_condition_number=False
):
    A,r0=assemble_linear_strong_system(
        ops,bc,E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )
    return solve_from_operator(
        A,r0,ops,bc,E=E,nu=nu,h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
        linear_residual_tolerance_relative=linear_residual_tolerance_relative,
        compute_condition_number=compute_condition_number,
    )
