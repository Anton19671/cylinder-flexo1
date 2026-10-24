#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10D: independent full-grid STRONG-FORM CUF FDM,
FULL GREEN-LAGRANGE + BIRGER.

Frozen-GB inner problem:
    full nodal q[component,tau,s,x],
    Stage7A full Green-Lagrange CUF mechanics,
    frozen GB(z,s,x) through the complete Newton solve,
    Stage7A strong residual,
    Stage7C end BC,
    V3c exact analytic tangent of the frozen-GB inner problem,
    np.linalg.solve only.

Outer Birger:
    verified Stage5A solve_birger_outer_cycle unchanged.

No MVI/rank/X/Y, weak projection, FEM, zones, line-search, retry,
adaptive load subdivision, regularization, SVD/pinv/lstsq/QR/Cholesky.
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

import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as fdl
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as fg


class CUFStrongFDMFullGLBirgerError(RuntimeError):
    pass


class CUFStrongFDMFullGLBirgerSystemSingular(CUFStrongFDMFullGLBirgerError):
    pass


@dataclass
class CUFFrozenGBFullGLInnerResult:
    state: np.ndarray
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


@dataclass
class CUFStrongFDMFullGLBirgerCounter:
    calls: int = 0


def full_gl_frozen_gb_fullgrid_residual(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load
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
    resultants=g7.general_thickness_resultants(
        q,qx,qs,
        R=ops.grid.radius,
        K0=float(K0),
        GB=GB,
        quadrature=quadrature,
        basis=basis,
    )
    raw=g7.general_residuals(
        resultants,Dx=ops.Dx,Ds=ops.Ds,generalized_load=p
    )
    assembled=g9.assemble_strong_residual_with_end_bcs(
        raw,q,resultants,bc,reject_fully_free=True
    )
    boundary=g9.full_boundary_residuals(q,resultants,bc)
    bvec=np.concatenate([
        np.asarray(boundary["left"],float).ravel(),
        np.asarray(boundary["right"],float).ravel(),
    ])
    interior=np.asarray(raw[:,:,:,1:-1],float).ravel()
    return assembled.reshape(-1),raw,resultants,qx,qs,bvec,interior


def _finite_difference_jacobian_frozen_gb_dense_reference(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,fd_step,
    base_residual=None
):
    """V3 one-column scaled central frozen-GB Jacobian reference."""
    q=fdl._validate_q(q,ops,basis)
    x=fdl.flatten_q(q,ops,basis)
    if base_residual is None:
        r0,*_=full_gl_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
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
        rp,*_=full_gl_frozen_gb_fullgrid_residual(
            fdl.unflatten_q(xp,ops,basis),ops,basis,bc,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load
        )
        rm,*_=full_gl_frozen_gb_fullgrid_residual(
            fdl.unflatten_q(xm,ops,basis),ops,basis,bc,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load
        )
        J[:,j]=(rp-rm)/(2.0*hj)
    return J,r0

def finite_difference_jacobian_frozen_gb(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,fd_step,
    base_residual=None
):
    """V3 scaled central frozen-GB Jacobian with K2 safe coloring."""
    q=fdl._validate_q(q,ops,basis)
    scales=v3fd.cuf_characteristic_scale(q,ops,quadrature)

    def evaluate(qp):
        r,*_=full_gl_frozen_gb_fullgrid_residual(
            qp,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
        return r

    J,r0,stats=k2.assemble_cuf_colored_central_columns(
        q,ops,evaluate,characteristic_scale=scales,relative_step=fd_step,
        base_residual=base_residual,dependency_layers=2
    )
    finite_difference_jacobian_frozen_gb.last_k2_stats=stats.to_dict()
    return J,r0


finite_difference_jacobian_frozen_gb.last_k2_stats=None


def _prepare_exact_frozen_gb_tangent_action(q,ops,basis,bc,*,K0,GB,quadrature):
    """V3c exact dR(q;GB)[dq] with GB strictly frozen.

    The verified Birger algorithm updates GB only in the outer material cycle.
    Therefore the inner Newton derivative is a partial derivative with respect
    to q at fixed GB.  No dGB/depsilon_i term belongs in this Jacobian.
    """
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    ctx=g7.prepare_general_elastic_tangent_context(
        q,qx,qs,R=ops.grid.radius,K0=float(K0),GB=GB,
        quadrature=quadrature,basis=basis,
    )
    zload=np.zeros_like(q)

    def action(dq):
        dq=fdl._validate_q(dq,ops,basis)
        dqx,dqs=fdl.surface_first_derivatives(dq,ops,basis)
        dres=g7.general_thickness_resultants_tangent_from_context(
            ctx,dq,dqx,dqs
        )
        draw=g7.general_residuals(
            dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
        )
        dasm=fg._assemble_exact_tangent_with_end_bcs(
            draw,dq,dres,bc
        )
        return dasm.reshape(-1)
    return action


def analytic_tangent_action_frozen_gb(
    q,dq,ops,basis,bc,*,K0,GB,quadrature,generalized_load=None
):
    """Exact V3c CUF Birger inner tangent action at fixed GB."""
    action=_prepare_exact_frozen_gb_tangent_action(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature
    )
    return action(dq)


def analytic_jacobian_frozen_gb(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,
    base_residual=None
):
    """Exact V3c frozen-GB Jacobian; no FD and no GB update inside Newton."""
    q=fdl._validate_q(q,ops,basis)
    action=_prepare_exact_frozen_gb_tangent_action(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature
    )
    J,stats=v3an.assemble_cuf_colored_exact_tangent(
        q,ops,action,dependency_layers=2
    )
    analytic_jacobian_frozen_gb.last_stats=stats.to_dict()
    if base_residual is None:
        r0,*_=full_gl_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(q.size,):
            raise ValueError("base_residual size mismatch")
    return J,r0


analytic_jacobian_frozen_gb.last_stats=None


def solve_frozen_gb_full_gl_strong_fdm(
    initial_q,ops,basis,bc,*,E0,K0,GB,quadrature,
    generalized_load,newton_settings,load_front_surface_mask=None
):
    q=fdl._validate_q(initial_q,ops,basis).copy()
    history=[]
    last_increment=np.inf

    # Exact zero-residual early acceptance; no Jacobian is needed.
    R0,*_=full_gl_frozen_gb_fullgrid_residual(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    h_v4=v3fd.thickness_from_quadrature(quadrature)
    measure_kw=dict(
        basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,bc=bc,E0=E0,h=h_v4,
    )
    R0_m=rscale.cuf_scaled_residual_for_measure(R0,**measure_kw)
    load_m=rscale.cuf_scaled_strong_load_for_measure(
        generalized_load,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x,
    )
    residual_scale=rscale.residual_scale(load_m,R0_m)
    residual_limit=rscale.threshold(newton_settings.residual_tolerance_relative,residual_scale)
    region_masks=rregions.resolve_front_exclusion(rregions.cuf_masks(ops,basis,load_front_surface_mask),generalized_load)
    rm0=rregions.metrics(R0_m,region_masks,residual_scale)
    r0inf=rm0.residual_inf
    if rregions.converged(rm0,residual_limit):
        return CUFFrozenGBFullGLInnerResult(
            state=q,converged=True,
            status="CONVERGED_CUF_STRONG_FDM_FROZEN_GB_FULL_GL",
            iterations=0,residual_inf=r0inf,
            residual_relative=rm0.residual_relative,
            residual_scale=residual_scale,increment_inf=0.0,history=[],
            residual_inf_interior=rm0.residual_inf_interior,
            residual_inf_boundary=rm0.residual_inf_boundary,
            residual_inf_load_front=rm0.residual_inf_load_front,
            residual_convergence_inf=rm0.residual_convergence_inf,
            residual_convergence_relative=rm0.residual_convergence_relative,
            residual_inf_by_block=rscale.cuf_residual_inf_by_block(
                R0_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
            )
        )

    prev_R=R0
    for k in range(1,newton_settings.max_iterations+1):
        J,R=analytic_jacobian_frozen_gb(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,base_residual=prev_R
        )
        try:
            dq,_,_,_=nscale.equilibrated_solve(J,-R)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMFullGLBirgerSystemSingular(
                f"SINGULAR_CUF_STRONG_FDM_FROZEN_GB_FULL_GL_JACOBIAN / iteration={k}"
            ) from exc
        if not np.all(np.isfinite(dq)):
            raise CUFStrongFDMFullGLBirgerError(
                f"NONFINITE_CUF_STRONG_FDM_FROZEN_GB_FULL_GL_INCREMENT / iteration={k}"
            )
        last_increment=float(np.max(np.abs(dq)))
        q=fdl.unflatten_q(fdl.flatten_q(q,ops,basis)+dq,ops,basis)
        Rnew,*_=full_gl_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
        prev_R=Rnew
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
            and last_increment<=newton_settings.increment_tolerance
        ):
            return CUFFrozenGBFullGLInnerResult(
                state=q,converged=True,
                status="CONVERGED_CUF_STRONG_FDM_FROZEN_GB_FULL_GL",
                iterations=k,residual_inf=rnew,
                residual_relative=rmnew.residual_relative,
                residual_scale=residual_scale,
                increment_inf=last_increment,history=history,
                residual_inf_interior=rmnew.residual_inf_interior,
                residual_inf_boundary=rmnew.residual_inf_boundary,
                residual_inf_load_front=rmnew.residual_inf_load_front,
                residual_convergence_inf=rmnew.residual_convergence_inf,
                residual_convergence_relative=rmnew.residual_convergence_relative,
                residual_inf_by_block=rscale.cuf_residual_inf_by_block(
                    Rnew_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
                )
            )

    Rfin,*_=full_gl_frozen_gb_fullgrid_residual(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    Rfin_m=rscale.cuf_scaled_residual_for_measure(Rfin,**measure_kw)
    rmfin=rregions.metrics(Rfin_m,region_masks,residual_scale)
    return CUFFrozenGBFullGLInnerResult(
        state=q,converged=False,
        status="MAX_CUF_STRONG_FDM_FROZEN_GB_FULL_GL_NEWTON_ITERATIONS",
        iterations=newton_settings.max_iterations,
        residual_inf=rmfin.residual_inf,
        residual_relative=rmfin.residual_relative,
        residual_scale=residual_scale,
        increment_inf=float(last_increment),history=history,
        residual_inf_interior=rmfin.residual_inf_interior,
        residual_inf_boundary=rmfin.residual_inf_boundary,
        residual_inf_load_front=rmfin.residual_inf_load_front,
        residual_convergence_inf=rmfin.residual_convergence_inf,
        residual_convergence_relative=rmfin.residual_convergence_relative,
        residual_inf_by_block=rscale.cuf_residual_inf_by_block(
            Rfin_m,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
        )
    )


def evaluate_full_gl_material_fullgrid(
    q,ops,basis,*,K0,GB,quadrature
):
    """K2b batched material-point evaluation for nonlinear CUF FDM."""
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    surf=(ops.grid.ns,ops.grid.nx)
    nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float)
    sig=np.empty_like(eps)
    det=np.empty_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    R=float(ops.grid.radius); K0=float(K0)

    for g,z in enumerate(quadrature.z):
        if mode=="scalar": GBg=float(GBa)
        elif mode=="surface": GBg=GBa
        else: GBg=GBa[g]
        _H,_Ft,_Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        F=abc.copy(); F[0,0]+=1.0; F[1,1]+=1.0; F[2,2]+=1.0
        J=g7._batched_det3(F)
        g7._raise_if_invalid_batched_F(J,z,surf)
        E=g7._batched_green_lagrange(F)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[g]=g7._batched_strain_intensity(E)
        sig[g]=g7._batched_stress_intensity(S)
        det[g]=J
    return b5.MaterialPointState(eps,sig,det)


def make_inner_solver(
    ops,basis,bc,*,E0,K0,quadrature,generalized_load,
    newton_settings,counter=None,load_front_surface_mask=None
):
    if counter is None:
        counter=CUFStrongFDMFullGLBirgerCounter()
    def inner(state_guess,GB):
        counter.calls+=1
        result=solve_frozen_gb_full_gl_strong_fdm(
            state_guess,ops,basis,bc,E0=E0,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load,
            newton_settings=newton_settings,load_front_surface_mask=load_front_surface_mask
        )
        inner.last_result=result
        return result
    inner.counter=counter
    inner.last_result=None
    return inner


def make_material_evaluator(ops,basis,*,K0,quadrature):
    def evaluator(q,GB):
        return evaluate_full_gl_material_fullgrid(
            q,ops,basis,K0=K0,GB=GB,quadrature=quadrature
        )
    return evaluator


def solve_full_gl_birger_strong_fdm(
    initial_q,initial_GB,*,ops,basis,bc,E0,nu0,quadrature,
    generalized_load,newton_settings,material_diagram,birger_settings,
    previous_accepted_epsilon_i=None,inner_counter=None,load_front_surface_mask=None
):
    initial_q=fdl._validate_q(initial_q,ops,basis)
    ec=gl.lk.ElasticConstants(float(E0),float(nu0))
    expected=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    GB0=np.asarray(initial_GB,float)
    if GB0.shape!=expected:
        raise ValueError(f"initial_GB must have shape {expected}")
    if np.any(~np.isfinite(GB0)) or np.any(GB0<=0.0):
        raise ValueError("initial_GB must be finite and positive")
    inner=make_inner_solver(
        ops,basis,bc,E0=E0,K0=ec.K0,quadrature=quadrature,
        generalized_load=generalized_load,newton_settings=newton_settings,
        counter=inner_counter,load_front_surface_mask=load_front_surface_mask
    )
    evaluator=make_material_evaluator(
        ops,basis,K0=ec.K0,quadrature=quadrature
    )
    result=b5.solve_birger_outer_cycle(
        initial_q,GB0,G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,evaluate_material=evaluator,
        state_vector=lambda st: np.asarray(st,float).reshape(-1),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i
    )
    if inner.last_result is not None:
        setattr(result,"inner_history",list(getattr(inner.last_result,"history",[]) or []))
        for name in (
            "residual_inf","residual_relative","residual_scale","increment_inf",
            "residual_inf_interior","residual_inf_boundary","residual_inf_load_front",
            "residual_convergence_inf","residual_convergence_relative"
        ):
            setattr(result,name,getattr(inner.last_result,name))
        # Reviewer V-4: preserve the per-(component,tau) scaled residual norms
        # for STEP_XXXX.json; diagnostics only, no change to the inner equations.
        setattr(
            result,"residual_inf_by_block",
            getattr(inner.last_result,"residual_inf_by_block",None)
        )
    return result,inner.counter
