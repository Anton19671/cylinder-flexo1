#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8B-4: weak-Kirchhoff variational projected residual
for the displacement-only closed cylindrical KL shell.

This is the verified free-edge principle already used in the plate branch:
- homogeneous essential conditions restrict admissible X profiles;
- natural end data enter the variational residual;
- free Mx/Tw are NOT imposed by artificial w/slope fixing;
- pointwise natural quantities remain independent diagnostics.

No zonal decomposition.
No KKT.
No Galerkin/Ritz basis.
No MVI solve yet.

For current resultants N,M and an admissible variation dU:

LINEAR:
 d eps_x = du_x
 d eps_s = dv_s + dw/R
 d gamma = du_s + dv_x

VON_KARMAN:
 d eps_x = du_x + w_x dw_x
 d eps_s = dv_s + dw/R + w_s dw_s
 d gamma = du_s + dv_x + w_s dw_x + w_x dw_s

GREEN_LAGRANGE:
 exact directional derivatives of both the midsurface Green--Lagrange strains
 and the full classical changes of curvature. The deformed unit normal and
 second fundamental form are differentiated without Donnell simplifications.

LINEAR/VON_KARMAN curvatures:
 d kappa_x  = -dw_xx
 d kappa_s  = -dw_ss
 d kappa_xs = -dw_xs

Internal first variation:
 int [
   Nx d eps_x + Ns d eps_s + Nxs d gamma
 + Mx d kappa_x + Ms d kappa_s + 2 Mxs d kappa_xs
 ] dA

Distributed external first variation:
 int [p_u du + p_v dv + p_w dw] dA

Natural x-end work:
 int [T_u du + T_v dv + T_w dw + T_theta dtheta_n] ds

The projected residual is dPi/dc for each active separated-profile parameter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_full_gl_kinematics as fullgl
import cylinder_kl_stage8b2_profiles as kp
import cylinder_kl_stage8b3_essential_bc as ke


@dataclass(frozen=True)
class KLVariationBundle:
    dU: np.ndarray
    dUx: np.ndarray
    dUs: np.ndarray
    dUxx: np.ndarray
    dUss: np.ndarray
    dUxs: np.ndarray
    label: tuple


def surface_weights(grid):
    wx=np.full(grid.nx,grid.dx,float)
    wx[0]*=.5
    wx[-1]*=.5
    ws=np.full(grid.ns,grid.ds,float)  # periodic circumference, no duplicate endpoint
    return np.outer(ws,wx)


def edge_weights(grid):
    return np.full(grid.ns,grid.ds,float)


def _x_variations(state,ops,maps):
    state.validate_grid(ops.grid)
    D1=ops.D1x
    # The current KL kinematics uses the verified direct second-derivative
    # matrices ops.D2x/ops.D2s.  Its admissible variations must use exactly
    # the same discrete operators; D1@D1 is a different stencil and made the
    # projected residual inconsistent with the energy being differentiated.
    D2=ops.D2x
    D1s=ops.D1s
    D2s=ops.D2s
    ns,nx=ops.grid.ns,ops.grid.nx

    for a,m in enumerate(maps.maps):
        for rr in range(state.rank):
            y=state.Y[a,rr]
            y1=D1s@y
            y2=D2s@y
            for j in range(m.n_free):
                x=m.B[:,j]
                x1=D1@x
                x2=D2@x

                arrays=[np.zeros((3,ns,nx),float) for _ in range(6)]
                dU,dUx,dUs,dUxx,dUss,dUxs=arrays
                dU[a]=np.outer(y,x)
                dUx[a]=np.outer(y,x1)
                dUs[a]=np.outer(y1,x)
                dUxx[a]=np.outer(y,x2)
                dUss[a]=np.outer(y2,x)
                dUxs[a]=np.outer(y1,x1)
                yield KLVariationBundle(*arrays,(a,rr,j))


def _y_variations(state,ops):
    state.validate_grid(ops.grid)
    D1=ops.D1x
    D2=ops.D2x
    D1s=ops.D1s
    D2s=ops.D2s
    ns,nx=ops.grid.ns,ops.grid.nx

    for a in range(3):
        for rr in range(state.rank):
            x=state.X[a,rr]
            x1=D1@x
            x2=D2@x
            for j in range(ns):
                y=np.zeros(ns); y[j]=1.0
                y1=D1s[:,j]
                y2=D2s[:,j]

                arrays=[np.zeros((3,ns,nx),float) for _ in range(6)]
                dU,dUx,dUs,dUxx,dUss,dUxs=arrays
                dU[a]=np.outer(y,x)
                dUx[a]=np.outer(y,x1)
                dUs[a]=np.outer(y1,x)
                dUxx[a]=np.outer(y,x2)
                dUss[a]=np.outer(y2,x)
                dUxs[a]=np.outer(y1,x1)
                yield KLVariationBundle(*arrays,(a,rr,j))


def direction_variations(state,ops,maps,direction):
    d=str(direction).upper()
    if d=="X":
        return tuple(_x_variations(state,ops,maps))
    if d=="Y":
        return tuple(_y_variations(state,ops))
    raise ValueError("direction must be X or Y")


def _natural_edge_external_variation(var,kin,bc,grid,*,geometry,exact_var=None):
    """
    External/generalized support work at x-ends.

    NATURAL: external traction is prescribed Tbar.
    ELASTIC:  T + k(q-q0) = Tbar, hence the equivalent external traction
              entering internal-external virtual work is Tbar-k(q-q0).
    """
    we=edge_weights(grid)
    total=0.0
    for _side,end,ix,n in (
        ("LEFT",bc.left,0,-1.0),
        ("RIGHT",bc.right,-1,+1.0),
    ):
        if str(geometry).upper()=="GREEN_LAGRANGE":
            if kin.full_gl is None or exact_var is None:
                raise ValueError("full Green-Lagrange edge variation is missing")
            dtheta=-n*exact_var.d_normal[0,:,ix]
            theta=-n*kin.full_gl.normal[0,:,ix]
        else:
            dtheta=n*var.dUx[2,:,ix]
            theta=n*kin.wx[:,ix]
        dq=(
            var.dU[0,:,ix],
            var.dU[1,:,ix],
            var.dU[2,:,ix],
            dtheta,
        )
        q=(
            kin.U[0,:,ix],
            kin.U[1,:,ix],
            kin.U[2,:,ix],
            theta,
        )
        for cond,dqi,qi in zip((end.u,end.v,end.w,end.slope),dq,q):
            if cond.kind=="NATURAL":
                effective=cond.data.array(grid.ns)
                total += float(np.sum(we*effective*dqi))
            elif cond.kind=="ELASTIC":
                target=cond.data.array(grid.ns)
                effective=(target
                           -cond.stiffness_array(grid.ns)*(np.asarray(qi,float)-cond.reference_array(grid.ns)))
                total += float(np.sum(we*effective*dqi))
    return total


def _check_geometric_variation_traces(var,bc,grid,*,kin=None,geometry="LINEAR",exact_var=None,tol=2e-10):
    """
    Essential constraints are carried by X. Therefore every admissible
    X- or Y-direction variation must have zero trace for each GEOMETRIC datum.

    The reconstructed separated variation may be very large even though its
    constrained end trace is zero analytically.  Direct elimination then
    leaves a roundoff trace proportional to that variation scale.  The guard
    therefore combines the established absolute tolerance with a machine-
    precision term; this changes only the diagnostic admissibility check, not
    the constrained profile, projected equations or MVI update.
    """
    for side,end,ix,n in (
        ("LEFT",bc.left,0,-1.0),("RIGHT",bc.right,-1,+1.0)
    ):
        if str(geometry).upper()=="GREEN_LAGRANGE":
            if kin is None or kin.full_gl is None or exact_var is None:
                raise ValueError("full Green-Lagrange geometric trace is missing")
            dslope=-n*exact_var.d_normal[0,:,ix]
            slope_field=exact_var.d_normal[0]
        else:
            dslope=n*var.dUx[2,:,ix]
            slope_field=var.dUx[2]
        dq=(
            var.dU[0,:,ix],
            var.dU[1,:,ix],
            var.dU[2,:,ix],
            dslope,
        )
        full=(var.dU[0],var.dU[1],var.dU[2],slope_field)
        names=("u","v","w","slope")
        for name,cond,dqi,qi in zip(
            names,(end.u,end.v,end.w,end.slope),dq,full
        ):
            err=float(np.max(np.abs(dqi)))
            scale=max(float(np.max(np.abs(qi))),1.0)
            limit=float(tol)+64.0*np.finfo(float).eps*scale
            if cond.kind=="GEOMETRIC" and err>limit:
                raise ke.KLEssentialConstraintError(
                    "admissible variation violates homogeneous geometric BC: "
                    f"side={side}, field={name}, trace_inf={err:.17g}, "
                    f"variation_scale={scale:.17g}, relative={err/scale:.17g}, "
                    f"effective_tolerance={limit:.17g}"
                )


def current_elastic_kinematics_resultants(
    state,ops,*,E,nu,h,geometry,resultants_method="ANALYTIC",quadrature=None
):
    U=kp.reconstruct_fields(state)
    kin=kl.field_kinematics(U,ops,geometry=geometry)
    c=kl.KLElasticConstants(E,nu,h)
    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        nm=kl.analytic_elastic_resultants(kin.state,c)
    elif method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        nm=kl.quadrature_elastic_resultants(kin.state,c,quadrature)
    else:
        raise ValueError("resultants_method must be ANALYTIC or QUADRATURE")
    return kin,nm


def projected_residual_from_variations(
    kin,resultants,variations,ops,bc,*,geometry,generalized_load
):
    bc.check_rigid_mode_requirement()
    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    W=surface_weights(ops.grid)
    mode=str(geometry).upper()
    if mode not in ("LINEAR","VON_KARMAN","GREEN_LAGRANGE"):
        raise ValueError("geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")

    values=[]
    labels=[]
    for var in variations:
        exact_var=None

        dux=var.dUx[0]
        dus=var.dUs[0]
        dvx=var.dUx[1]
        dvs=var.dUs[1]
        dw=var.dU[2]
        dwx=var.dUx[2]
        dws=var.dUs[2]

        dex=dux.copy()
        des=dvs+dw/ops.grid.radius
        dgam=dus+dvx
        if mode=="VON_KARMAN":
            dex += kin.wx*dwx
            des += kin.ws*dws
            dgam += kin.ws*dwx + kin.wx*dws
        if mode=="GREEN_LAGRANGE":
            if kin.full_gl is None:
                raise ValueError("full Green-Lagrange KL kinematics is missing")
            exact_var=fullgl.directional_variation(
                kin.full_gl,
                var.dU,var.dUx,var.dUs,var.dUxx,var.dUss,var.dUxs,
                R=ops.grid.radius,
            )
            dex=exact_var.d_eps_x
            des=exact_var.d_eps_s
            dgam=exact_var.d_gamma_xs
            dkx=exact_var.d_kappa_x
            dks=exact_var.d_kappa_s
            dkxs=exact_var.d_kappa_xs
        else:
            dkx=-var.dUxx[2]
            dks=-var.dUss[2]
            dkxs=-var.dUxs[2]

        _check_geometric_variation_traces(
            var,bc,ops.grid,kin=kin,geometry=mode,exact_var=exact_var
        )

        internal=(
            resultants.Nx*dex
            +resultants.Ns*des
            +resultants.Nxs*dgam
            +resultants.Mx*dkx
            +resultants.Ms*dks
            +2.0*resultants.Mxs*dkxs
        )
        external_area=(
            p[0]*var.dU[0]
            +p[1]*var.dU[1]
            +p[2]*var.dU[2]
        )
        val=float(np.sum(W*(internal-external_area)))
        val -= _natural_edge_external_variation(
            var,kin,bc,ops.grid,geometry=mode,exact_var=exact_var
        )
        values.append(val)
        labels.append(var.label)
    return np.asarray(values,float),tuple(labels)


def j2_algorithmic_projected_tangent(
    kin, resultants, consistent_tangent, variations, ops, bc, *, geometry, quadrature
):
    """Project the frozen-history J2 algorithmic tangent onto KL MVI directions.

    This is the KL counterpart of the working plate MVI implementation: the
    constitutive tangent is formed once at the current material state and then
    applied to every reduced profile direction.  It supports LINEAR and
    von-Karman kinematics; full Green--Lagrange remains on its dedicated
    finite-difference path until its complete geometric tangent is supplied.
    """
    mode = str(geometry).upper()
    if mode not in ("LINEAR", "VON_KARMAN"):
        raise NotImplementedError("algorithmic KL J2 tangent is unavailable for GREEN_LAGRANGE")
    for end in (bc.left, bc.right):
        if any(cond.kind == "ELASTIC" for cond in (end.u, end.v, end.w, end.slope)):
            raise NotImplementedError("algorithmic KL J2 tangent with elastic edge supports is unavailable")
    calg = np.asarray(consistent_tangent, float)
    z = np.asarray(quadrature.z, float)
    wz = np.asarray(quadrature.w, float)
    expected = (z.size, ops.grid.ns, ops.grid.nx, 3, 3)
    if calg.shape != expected or not np.all(np.isfinite(calg)):
        raise ValueError(f"J2 algorithmic tangent must have shape {expected}")
    vars_ = tuple(variations)
    n = len(vars_)
    if n == 0:
        return np.empty((0, 0), float)

    B = np.empty((n, 3, ops.grid.ns, ops.grid.nx), float)
    K = np.empty_like(B)
    dwx = np.empty((n, ops.grid.ns, ops.grid.nx), float)
    dws = np.empty_like(dwx)
    for j, var in enumerate(vars_):
        B[j, 0] = var.dUx[0]
        B[j, 1] = var.dUs[1] + var.dU[2] / ops.grid.radius
        B[j, 2] = var.dUs[0] + var.dUx[1]
        dwx[j] = var.dUx[2]
        dws[j] = var.dUs[2]
        if mode == "VON_KARMAN":
            B[j, 0] += kin.wx * dwx[j]
            B[j, 1] += kin.ws * dws[j]
            B[j, 2] += kin.ws * dwx[j] + kin.wx * dws[j]
        K[j, 0] = -var.dUxx[2]
        K[j, 1] = -var.dUss[2]
        K[j, 2] = -var.dUxs[2]

    W = surface_weights(ops.grid)
    matrix = np.empty((n, n), float)
    for col in range(n):
        strain = np.empty((z.size, ops.grid.ns, ops.grid.nx, 3), float)
        strain[..., 0] = B[col, 0][None, :, :] + z[:, None, None] * K[col, 0][None, :, :]
        strain[..., 1] = B[col, 1][None, :, :] + z[:, None, None] * K[col, 1][None, :, :]
        strain[..., 2] = B[col, 2][None, :, :] + 2.0 * z[:, None, None] * K[col, 2][None, :, :]
        dsig = np.einsum("gsxij,gsxj->gsxi", calg, strain, optimize=True)
        dN = np.sum(wz[:, None, None, None] * dsig, axis=0)
        dM = np.sum((z * wz)[:, None, None, None] * dsig, axis=0)
        internal = (
            dN[..., 0][None, :, :] * B[:, 0]
            + dN[..., 1][None, :, :] * B[:, 1]
            + dN[..., 2][None, :, :] * B[:, 2]
            + dM[..., 0][None, :, :] * K[:, 0]
            + dM[..., 1][None, :, :] * K[:, 1]
            + 2.0 * dM[..., 2][None, :, :] * K[:, 2]
        )
        if mode == "VON_KARMAN":
            dbx = dwx[col][None, :, :] * dwx
            dbs = dws[col][None, :, :] * dws
            dbg = dws[col][None, :, :] * dwx + dwx[col][None, :, :] * dws
            internal += (
                resultants.Nx[None, :, :] * dbx
                + resultants.Ns[None, :, :] * dbs
                + resultants.Nxs[None, :, :] * dbg
            )
        matrix[:, col] = np.sum(W[None, :, :] * internal, axis=(1, 2))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite algorithmic KL J2 projected tangent")
    return matrix


def elastic_projected_residual(
    state,ops,maps,bc,*,E,nu,h,geometry,generalized_load,
    direction,resultants_method="ANALYTIC",quadrature=None
):
    kin,nm=current_elastic_kinematics_resultants(
        state,ops,E=E,nu=nu,h=h,geometry=geometry,
        resultants_method=resultants_method,quadrature=quadrature
    )
    variations=direction_variations(state,ops,maps,direction)
    r,labels=projected_residual_from_variations(
        kin,nm,variations,ops,bc,geometry=geometry,
        generalized_load=generalized_load
    )
    return r,labels,kin,nm


def total_potential_elastic(
    state,ops,bc,*,E,nu,h,geometry,generalized_load,
    resultants_method="ANALYTIC",quadrature=None
):
    """
    Independent scalar energy used only as a verification identity:
       r_j = d Pi / d c_j.
    """
    kin,nm=current_elastic_kinematics_resultants(
        state,ops,E=E,nu=nu,h=h,geometry=geometry,
        resultants_method=resultants_method,quadrature=quadrature
    )
    W=surface_weights(ops.grid)
    st=kin.state
    internal_density=0.5*(
        nm.Nx*st.eps_x + nm.Ns*st.eps_s + nm.Nxs*st.gamma_xs
        +nm.Mx*st.kappa_x + nm.Ms*st.kappa_s
        +2.0*nm.Mxs*st.kappa_xs
    )
    p=np.asarray(generalized_load,float)
    external_density=p[0]*kin.U[0]+p[1]*kin.U[1]+p[2]*kin.U[2]
    Pi=float(np.sum(W*(internal_density-external_density)))

    we=edge_weights(ops.grid)
    # external natural-edge potential = - int T q
    mode=str(geometry).upper()
    for end,ix,n in ((bc.left,0,-1.0),(bc.right,-1,+1.0)):
        theta=(
            -n*kin.full_gl.normal[0,:,ix]
            if mode=="GREEN_LAGRANGE" else n*kin.wx[:,ix]
        )
        q=(
            kin.U[0,:,ix],kin.U[1,:,ix],kin.U[2,:,ix],theta,
        )
        for cond,qi in zip((end.u,end.v,end.w,end.slope),q):
            if cond.kind=="NATURAL":
                Pi -= float(np.sum(we*cond.data.array(ops.grid.ns)*qi))
            elif cond.kind=="ELASTIC":
                k=cond.stiffness_array(ops.grid.ns)
                q0=cond.reference_array(ops.grid.ns)
                tbar=cond.data.array(ops.grid.ns)
                dq0=np.asarray(qi,float)-q0
                Pi += float(np.sum(we*(0.5*k*dq0*dq0-tbar*np.asarray(qi,float))))
    return Pi
