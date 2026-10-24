#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10B: independent full-grid STRONG-FORM CUF finite-difference solver,
FULL GREEN-LAGRANGE ELASTIC.

Unknowns:
    q[component,tau,s_index,x_index]

The mechanics are the already verified general CUF Stage7A equations:
    F -> detF -> Green-Lagrange E -> elastic S -> P=F@S
    -> thickness resultants -> strong residual.

The exact nonlinear correction remains in conservative flux-divergence form.
Only the affine linear-elastic part is discretized with the direct Dxx/Dss
operators, avoiding the wider D1@D1 stencil and its odd/even decoupling.

No separated representation, rank, X/Y profiles, MVI, weak projection,
FEM, zones, regularization, line-search, relaxation, retry, or adaptive
load subdivision.

Nonlinear solve:
    J(q_k) dq = -R(q_k)
    q_{k+1}   = q_k + dq

Production J uses the V3b exact analytic material/geometric tangent of the
complete strong collocation residual.  The V3 scaled central FD Jacobian is
retained as an independent regression/reference path.  fd_step remains in the
settings for backward-compatible case files and reference checks.  The initial
q is mandatory and caller-supplied.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import final_progress as fprog

import final_numeric_scaling as nscale
import final_k2_coloring as k2
import final_v3_fd as v3fd
import final_v3_analytic as v3an
import final_residual_scaling as rscale
import final_residual_regions as rregions

import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as fdl


class CUFStrongFDMNonlinearError(RuntimeError):
    pass


class CUFStrongFDMNonlinearSystemSingular(CUFStrongFDMNonlinearError):
    pass


@dataclass(frozen=True)
class CUFStrongFDMNewtonSettings:
    residual_tolerance_relative: float
    increment_tolerance: float
    fd_step: float
    max_iterations: int

    def __post_init__(self):
        if self.residual_tolerance_relative<=0.0:
            raise ValueError("residual_tolerance_relative must be >0")
        if self.increment_tolerance<=0.0:
            raise ValueError("increment_tolerance must be >0")
        if self.fd_step<=0.0:
            raise ValueError("fd_step must be >0")
        if self.max_iterations<1:
            raise ValueError("max_iterations must be >=1")


@dataclass
class CUFStrongFDMNonlinearResult:
    q: np.ndarray
    converged: bool
    status: str
    iterations: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    increment_inf: float
    history: list
    residual_inf_interior: float = 0.0
    residual_inf_boundary: float = 0.0
    residual_inf_load_front: float = 0.0
    residual_convergence_inf: float = 0.0
    residual_convergence_relative: float = 0.0
    residual_inf_by_block: dict | None = None


def full_gl_elastic_fullgrid_residual(
    q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load
):
    q=fdl._validate_q(q,ops,basis)
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    if p.shape!=q.shape:
        raise ValueError(f"generalized_load must have shape {q.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    ec=gl.lk.ElasticConstants(float(E0),float(nu0))

    # Full Green-Lagrange CUF mechanics.  general_local_kernel performs
    # the mandatory detF>0 admissibility check before constitutive use.
    resultants=g7.general_thickness_resultants(
        q,qx,qs,
        R=ops.grid.radius,
        K0=ec.K0,
        GB=ec.G0,
        quadrature=quadrature,
        basis=basis,
    )

    # Discretize the exact nonlinear correction in flux form, but use the
    # direct Dxx/Dss linear operator for the affine part.  This removes the
    # D1@D1 checkerboard mechanism without altering the Green-Lagrange fluxes.
    raw_flux=g7.general_residuals(
        resultants,
        Dx=ops.Dx,
        Ds=ops.Ds,
        generalized_load=p,
    )
    raw_linear_consistent,linear_resultants,_lqx,_lqs=(
        fdl.linear_elastic_consistent_raw_residual(
            q,ops,basis,
            E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=p,
        )
    )
    raw_linear_flux=g7.general_residuals(
        linear_resultants,
        Dx=ops.Dx,
        Ds=ops.Ds,
        generalized_load=p,
    )
    raw=raw_flux-raw_linear_flux+raw_linear_consistent

    assembled=g9.assemble_strong_residual_with_end_bcs(
        raw,q,resultants,bc,reject_fully_free=True
    )

    boundary=g9.full_boundary_residuals(q,resultants,bc)
    bvec=np.concatenate([
        np.asarray(boundary["left"],float).ravel(),
        np.asarray(boundary["right"],float).ravel(),
    ])

    interior=np.asarray(raw[:,:,:,1:-1],float).ravel()

    return (
        assembled.reshape(-1),
        raw,
        resultants,
        qx,
        qs,
        bvec,
        interior,
    )


def _finite_difference_jacobian_dense_reference(
    q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,fd_step,
    base_residual=None
):
    """V3 one-column scaled central Jacobian retained as K2 reference."""
    q=fdl._validate_q(q,ops,basis)
    x=fdl.flatten_q(q,ops,basis)
    if base_residual is None:
        r0,*_=full_gl_elastic_fullgrid_residual(
            q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(x.size,):
            raise ValueError("base_residual size mismatch")
    scales=v3fd.cuf_characteristic_scale(q,ops,quadrature).reshape(-1)
    steps=v3fd.relative_steps(x,scales,fd_step)
    J=np.empty((x.size,x.size),float)
    for j,hj in enumerate(steps):
        xp=x.copy(); xm=x.copy()
        xp[j]+=hj; xm[j]-=hj
        rp,*_=full_gl_elastic_fullgrid_residual(
            fdl.unflatten_q(xp,ops,basis),ops,basis,bc,E0=E0,nu0=nu0,
            quadrature=quadrature,generalized_load=generalized_load,
        )
        rm,*_=full_gl_elastic_fullgrid_residual(
            fdl.unflatten_q(xm,ops,basis),ops,basis,bc,E0=E0,nu0=nu0,
            quadrature=quadrature,generalized_load=generalized_load,
        )
        J[:,j]=(rp-rm)/(2.0*hj)
    return J,r0

def finite_difference_jacobian(
    q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,fd_step,
    base_residual=None
):
    """V3 scaled central Jacobian with K2 safe coloring.

    ``fd_step`` is a dimensionless relative step.
    """
    q=fdl._validate_q(q,ops,basis)
    scales=v3fd.cuf_characteristic_scale(q,ops,quadrature)

    def evaluate(qp):
        r,*_=full_gl_elastic_fullgrid_residual(
            qp,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        return r

    J,r0,stats=k2.assemble_cuf_colored_central_columns(
        q,ops,evaluate,characteristic_scale=scales,relative_step=fd_step,
        base_residual=base_residual,dependency_layers=2
    )
    finite_difference_jacobian.last_k2_stats=stats.to_dict()
    return J,r0




finite_difference_jacobian.last_k2_stats=None


def _assemble_exact_tangent_with_end_bcs(draw,dq,dresultants,bc):
    """Derivative of Stage7C strong row replacement; prescribed data are constant."""
    draw=np.asarray(draw,float)
    dq=np.asarray(dq,float)
    if draw.shape!=dq.shape:
        raise ValueError("draw/dq shape mismatch")
    m,_ns,nx=dq.shape[1],dq.shape[2],dq.shape[3]
    if m!=bc.m:
        raise ValueError("BC m mismatch")
    out=draw.copy()
    natural=(dresultants.Au_x,dresultants.Av_x,dresultants.Aw_x)
    for _side,end,ix,normal in (
        ("LEFT",bc.left,0,-1.0),("RIGHT",bc.right,nx-1,+1.0)
    ):
        for a in range(3):
            for tau in range(m):
                cond=end.conditions[a][tau]
                if cond.kind=="GEOMETRIC":
                    out[a,tau,:,ix]=dq[a,tau,:,ix]
                elif cond.kind=="NATURAL":
                    out[a,tau,:,ix]=normal*natural[a][tau,:,ix]
                elif cond.kind=="ELASTIC":
                    out[a,tau,:,ix]=(normal*natural[a][tau,:,ix]
                                     +cond.stiffness_array(dq.shape[2])*dq[a,tau,:,ix])
                else:
                    raise ValueError(f"unsupported CUF boundary kind: {cond.kind}")
    return out


def _prepare_exact_elastic_tangent_action(q,ops,basis,bc,*,E0,nu0,quadrature):
    """Return exact dR(q)[dq] closure with base F/S cached through thickness."""
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    ec=gl.lk.ElasticConstants(float(E0),float(nu0))
    ctx=g7.prepare_general_elastic_tangent_context(
        q,qx,qs,R=ops.grid.radius,K0=ec.K0,GB=ec.G0,
        quadrature=quadrature,basis=basis,
    )
    zload=np.zeros_like(q)

    def action(dq):
        dq=fdl._validate_q(dq,ops,basis)
        dqx,dqs=fdl.surface_first_derivatives(dq,ops,basis)
        dres=g7.general_thickness_resultants_tangent_from_context(ctx,dq,dqx,dqs)
        draw_flux=g7.general_residuals(
            dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
        )

        # Tangent of the corrected residual: replace only the linear elastic
        # D1@D1 contribution by the same direct-D2 operator used by the residual.
        ec_lin=gl.lk.ElasticConstants(float(E0),float(nu0))
        dlin=gl.general_linear_thickness_resultants(
            dq,dqx,dqs,
            R=ops.grid.radius,K0=ec_lin.K0,GB=ec_lin.G0,
            quadrature=quadrature,basis=basis,
        )
        draw_linear_flux=g7.general_residuals(
            dlin,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
        )
        draw_linear_consistent,_dlin2,_dqx2,_dqs2=(
            fdl.linear_elastic_consistent_raw_residual(
                dq,ops,basis,
                E0=E0,nu0=nu0,quadrature=quadrature,
                generalized_load=zload,
            )
        )
        draw=draw_flux-draw_linear_flux+draw_linear_consistent
        dasm=_assemble_exact_tangent_with_end_bcs(draw,dq,dres,bc)
        return dasm.reshape(-1)
    return action


def analytic_tangent_action(
    q,dq,ops,basis,bc,*,E0,nu0,quadrature,generalized_load=None
):
    """Exact V3b Green-Lagrange elastic CUF tangent action dR(q)[dq]."""
    # generalized_load is accepted for API symmetry; its derivative is zero.
    action=_prepare_exact_elastic_tangent_action(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature
    )
    return action(dq)


def analytic_jacobian(
    q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,base_residual=None
):
    """Exact V3b dense Jacobian assembled from analytic material/geometric tangent."""
    q=fdl._validate_q(q,ops,basis)
    action=_prepare_exact_elastic_tangent_action(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature
    )
    J,stats=v3an.assemble_cuf_colored_exact_tangent(
        q,ops,action,dependency_layers=2
    )
    analytic_jacobian.last_stats=stats.to_dict()
    if base_residual is None:
        r0,*_=full_gl_elastic_fullgrid_residual(
            q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(q.size,):
            raise ValueError("base_residual size mismatch")
    return J,r0


analytic_jacobian.last_stats=None


def solve_full_gl_elastic_strong_fdm(
    initial_q,ops,basis,bc,*,E0,nu0,quadrature,
    generalized_load,settings,load_front_surface_mask=None
):
    q=fdl._validate_q(initial_q,ops,basis).copy()
    history=[]
    last_increment=np.inf

    # V2: compute one fixed problem scale before the Newton updates.
    R_initial,*_=full_gl_elastic_fullgrid_residual(
        q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    h_v4=v3fd.thickness_from_quadrature(quadrature)
    measure_kw=dict(
        basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h_v4,
    )
    R_initial_m=rscale.cuf_scaled_residual_for_measure(R_initial,**measure_kw)
    load_m=rscale.cuf_scaled_strong_load_for_measure(
        generalized_load,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,
    )
    residual_scale=rscale.residual_scale(load_m,R_initial_m)
    residual_limit=rscale.threshold(settings.residual_tolerance_relative,residual_scale)
    region_masks=rregions.resolve_front_exclusion(rregions.cuf_masks(ops,basis,load_front_surface_mask),generalized_load)

    for k in range(1,settings.max_iterations+1):
        # V4: test the current residual before assembling the expensive
        # Jacobian.  Reuse the V2 initial residual on the first iteration.
        if k==1:
            R=R_initial
        else:
            R,*_=full_gl_elastic_fullgrid_residual(
                q,ops,basis,bc,
                E0=E0,nu0=nu0,quadrature=quadrature,
                generalized_load=generalized_load,
            )
        Rm=(R_initial_m if k==1 else rscale.cuf_scaled_residual_for_measure(R,**measure_kw))
        rm=rregions.metrics(Rm,region_masks,residual_scale)
        rinf=rm.residual_inf

        if rregions.converged(rm,residual_limit):
            return CUFStrongFDMNonlinearResult(
                q=q,
                converged=True,
                status="CONVERGED_CUF_STRONG_FDM_FULL_GL_ELASTIC",
                iterations=k-1,
                residual_inf=rinf,
                residual_relative=rm.residual_relative,
                residual_scale=residual_scale,
                increment_inf=0.0,
                history=history,
                residual_inf_interior=rm.residual_inf_interior,
                residual_inf_boundary=rm.residual_inf_boundary,
                residual_inf_load_front=rm.residual_inf_load_front,
                residual_convergence_inf=rm.residual_convergence_inf,
                residual_convergence_relative=rm.residual_convergence_relative,
                residual_inf_by_block=rscale.cuf_residual_inf_by_block(
                    Rm,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
                ),
            )

        J,_=analytic_jacobian(
            q,ops,basis,bc,
            E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,base_residual=R,
        )

        try:
            dq,_,_,_=nscale.equilibrated_solve(J,-R)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMNonlinearSystemSingular(
                f"SINGULAR_CUF_STRONG_FDM_FULL_GL_JACOBIAN / iteration={k}"
            ) from exc

        if not np.all(np.isfinite(dq)):
            raise CUFStrongFDMNonlinearError(
                f"NONFINITE_CUF_STRONG_FDM_FULL_GL_INCREMENT / iteration={k}"
            )

        last_increment=float(np.max(np.abs(dq)))
        q=fdl.unflatten_q(
            fdl.flatten_q(q,ops,basis)+dq,ops,basis
        )

        Rnew,*_=full_gl_elastic_fullgrid_residual(
            q,ops,basis,bc,
            E0=E0,nu0=nu0,quadrature=quadrature,
            generalized_load=generalized_load,
        )
        Rnew_m=rscale.cuf_scaled_residual_for_measure(Rnew,**measure_kw)
        rmnew=rregions.metrics(Rnew_m,region_masks,residual_scale)
        rnew=rmnew.residual_inf

        hrow={"iteration":k,"residual_scale":residual_scale,"increment_inf":last_increment}
        hrow.update(rregions.as_dict(rmnew))
        hrow["residual_inf_by_block"]=rscale.cuf_residual_inf_by_block(
            Rnew_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
        )
        history.append(hrow)
        _w_live=np.asarray(q[2,0],float)
        _wis,_wix=np.unravel_index(int(np.argmax(np.abs(_w_live))),_w_live.shape)
        fprog.emit("newton_iteration",
            iteration=int(k), residual_inf=float(hrow.get("residual_inf", rnew)),
            residual_relative=float(hrow.get("residual_relative", 0.0)),
            residual_convergence_inf=float(hrow.get("residual_convergence_inf", hrow.get("residual_inf", rnew))),
            residual_convergence_relative=float(hrow.get("residual_convergence_relative", hrow.get("residual_relative", 0.0))),
            residual_inf_interior=float(hrow.get("residual_inf_interior", 0.0)),
            residual_inf_boundary=float(hrow.get("residual_inf_boundary", 0.0)),
            residual_inf_load_front=float(hrow.get("residual_inf_load_front", 0.0)),
            increment_inf=float(last_increment), stage="NEWTON",
            w_abs_max=float(np.max(np.abs(_w_live))),
            w_at_abs_max=float(_w_live[_wis,_wix]),
            wmax_is=int(_wis),wmax_ix=int(_wix))

        if (
            rregions.converged(rmnew,residual_limit)
            and last_increment<=settings.increment_tolerance
        ):
            return CUFStrongFDMNonlinearResult(
                q=q,
                converged=True,
                status="CONVERGED_CUF_STRONG_FDM_FULL_GL_ELASTIC",
                iterations=k,
                residual_inf=rnew,
                residual_relative=rmnew.residual_relative,
                residual_scale=residual_scale,
                increment_inf=last_increment,
                history=history,
                residual_inf_interior=rmnew.residual_inf_interior,
                residual_inf_boundary=rmnew.residual_inf_boundary,
                residual_inf_load_front=rmnew.residual_inf_load_front,
                residual_convergence_inf=rmnew.residual_convergence_inf,
                residual_convergence_relative=rmnew.residual_convergence_relative,
                residual_inf_by_block=rscale.cuf_residual_inf_by_block(
                    Rnew_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
                ),
            )

    Rfin,*_=full_gl_elastic_fullgrid_residual(
        q,ops,basis,bc,
        E0=E0,nu0=nu0,quadrature=quadrature,
        generalized_load=generalized_load,
    )
    Rfin_m=rscale.cuf_scaled_residual_for_measure(Rfin,**measure_kw)
    rmfin=rregions.metrics(Rfin_m,region_masks,residual_scale)
    return CUFStrongFDMNonlinearResult(
        q=q,
        converged=False,
        status="MAX_CUF_STRONG_FDM_FULL_GL_NEWTON_ITERATIONS",
        iterations=settings.max_iterations,
        residual_inf=rmfin.residual_inf,
        residual_relative=rmfin.residual_relative,
        residual_scale=residual_scale,
        increment_inf=float(last_increment),
        history=history,
        residual_inf_interior=rmfin.residual_inf_interior,
        residual_inf_boundary=rmfin.residual_inf_boundary,
        residual_inf_load_front=rmfin.residual_inf_load_front,
        residual_convergence_inf=rmfin.residual_convergence_inf,
        residual_convergence_relative=rmfin.residual_convergence_relative,
        residual_inf_by_block=rscale.cuf_residual_inf_by_block(
            Rfin_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
        ),
    )
