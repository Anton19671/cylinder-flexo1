#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strong-form full-grid CUF von-Karman Hooke/Birger branch.

The mechanics and exact tangent are supplied by Stage 7H.  This module keeps
the existing full-grid Newton and Birger outer-cycle policies; it changes only
the selected physical residual.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import final_progress as fprog
import final_numeric_scaling as nscale
import final_v3_fd as v3fd
import final_v3_analytic as v3an
import final_residual_scaling as rscale
import final_residual_regions as rregions

import cylinder_te1_local_kernel as lk
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7c_general_boundary as gbc
import cylinder_cuf_stage7h_general_vk as gvk
import cylinder_cuf_stage10a_strong_fdm_linear_elastic as fdl
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as fgl


class CUFStrongFDMVKError(RuntimeError): pass
class CUFStrongFDMVKSystemSingular(CUFStrongFDMVKError): pass


@dataclass
class CUFStrongFDMVKResult:
    state: np.ndarray
    converged: bool
    status: str
    iterations: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    increment_inf: float
    history: list
    residual_inf_interior: float=0.0
    residual_inf_boundary: float=0.0
    residual_inf_load_front: float=0.0
    residual_convergence_inf: float=0.0
    residual_convergence_relative: float=0.0
    residual_inf_by_block: dict|None=None

    @property
    def q(self): return self.state


@dataclass
class CUFStrongFDMVKCounter:
    calls: int=0


def vk_frozen_gb_fullgrid_residual(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load
):
    q=fdl._validate_q(q,ops,basis)
    p=np.asarray(generalized_load,float)
    if p.shape!=q.shape: raise ValueError(f"generalized_load must have shape {q.shape}")
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    resultants=gvk.general_vk_thickness_resultants(
        q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
        quadrature=quadrature,basis=basis
    )
    raw=g7.general_residuals(resultants,Dx=ops.Dx,Ds=ops.Ds,generalized_load=p)
    assembled=gbc.assemble_strong_residual_with_end_bcs(
        raw,q,resultants,bc,reject_fully_free=True
    )
    boundary=gbc.full_boundary_residuals(q,resultants,bc)
    bvec=np.concatenate((np.asarray(boundary["left"]).ravel(),
                         np.asarray(boundary["right"]).ravel()))
    interior=np.asarray(raw[:,:,:,1:-1],float).ravel()
    return assembled.reshape(-1),raw,resultants,qx,qs,bvec,interior


def _prepare_tangent_action(q,ops,basis,bc,*,K0,GB,quadrature):
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    ctx=gvk.prepare_vk_tangent_context(
        q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
        quadrature=quadrature,basis=basis
    )
    zload=np.zeros_like(q)
    def action(dq):
        dq=fdl._validate_q(dq,ops,basis)
        dqx,dqs=fdl.surface_first_derivatives(dq,ops,basis)
        dres=gvk.vk_resultants_tangent_from_context(ctx,dq,dqx,dqs)
        draw=g7.general_residuals(dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload)
        return fgl._assemble_exact_tangent_with_end_bcs(draw,dq,dres,bc).reshape(-1)
    return action


def analytic_tangent_action(q,dq,ops,basis,bc,*,K0,GB,quadrature):
    return _prepare_tangent_action(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature
    )(dq)


def analytic_jacobian(
    q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,base_residual=None
):
    q=fdl._validate_q(q,ops,basis)
    action=_prepare_tangent_action(q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature)
    J,stats=v3an.assemble_cuf_colored_exact_tangent(
        q,ops,action,dependency_layers=2
    )
    analytic_jacobian.last_stats=stats.to_dict()
    if base_residual is None:
        r0,*_=vk_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
    else: r0=np.asarray(base_residual,float).reshape(-1)
    return J,r0


analytic_jacobian.last_stats=None


def solve_frozen_gb_vk_strong_fdm(
    initial_q,ops,basis,bc,*,E0,K0,GB,quadrature,generalized_load,
    newton_settings,load_front_surface_mask=None
):
    q=fdl._validate_q(initial_q,ops,basis).copy()
    history=[]; last_increment=np.inf
    R0,*_=vk_frozen_gb_fullgrid_residual(
        q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    h=v3fd.thickness_from_quadrature(quadrature)
    measure_kw=dict(basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
                    L=ops.grid.length_x,bc=bc,E0=E0,h=h)
    R0m=rscale.cuf_scaled_residual_for_measure(R0,**measure_kw)
    loadm=rscale.cuf_scaled_strong_load_for_measure(
        generalized_load,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx,
        L=ops.grid.length_x
    )
    scale=rscale.residual_scale(loadm,R0m)
    limit=rscale.threshold(newton_settings.residual_tolerance_relative,scale)
    masks=rregions.resolve_front_exclusion(
        rregions.cuf_masks(ops,basis,load_front_surface_mask),generalized_load
    )

    def result(status,converged,iterations,R,Rm,inc):
        met=rregions.metrics(Rm,masks,scale)
        return CUFStrongFDMVKResult(
            q,converged,status,iterations,met.residual_inf,
            met.residual_relative,scale,float(inc),history,
            met.residual_inf_interior,met.residual_inf_boundary,
            met.residual_inf_load_front,met.residual_convergence_inf,
            met.residual_convergence_relative,
            rscale.cuf_residual_inf_by_block(
                Rm,basis_m=basis.m,ns=ops.grid.ns,nx=ops.grid.nx
            )
        )

    current_R=R0; current_Rm=R0m
    if rregions.converged(rregions.metrics(R0m,masks,scale),limit):
        return result("CONVERGED_CUF_STRONG_FDM_VK",True,0,R0,R0m,0.0)

    for it in range(1,newton_settings.max_iterations+1):
        J,R=analytic_jacobian(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,base_residual=current_R
        )
        try: dq,_,_,_=nscale.equilibrated_solve(J,-R)
        except np.linalg.LinAlgError as exc:
            raise CUFStrongFDMVKSystemSingular(
                f"SINGULAR_CUF_STRONG_FDM_VK_JACOBIAN / iteration={it}"
            ) from exc
        if not np.all(np.isfinite(dq)):
            raise CUFStrongFDMVKError("NONFINITE_CUF_STRONG_FDM_VK_INCREMENT")
        last_increment=float(np.max(np.abs(dq)))
        q=fdl.unflatten_q(fdl.flatten_q(q,ops,basis)+dq,ops,basis)
        current_R,*_=vk_frozen_gb_fullgrid_residual(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
        current_Rm=rscale.cuf_scaled_residual_for_measure(current_R,**measure_kw)
        met=rregions.metrics(current_Rm,masks,scale)
        row={"iteration":it,"increment_inf":last_increment,"residual_scale":scale}
        row.update(rregions.as_dict(met)); history.append(row)
        w=np.asarray(q[2,0],float); js,ix=np.unravel_index(np.argmax(np.abs(w)),w.shape)
        fprog.emit("newton_iteration",stage="NEWTON",iteration=it,
                   residual_inf=met.residual_inf,
                   residual_relative=met.residual_relative,
                   increment_inf=last_increment,w_abs_max=float(np.max(np.abs(w))),
                   w_at_abs_max=float(w[js,ix]),wmax_is=int(js),wmax_ix=int(ix))
        if rregions.converged(met,limit) and last_increment<=newton_settings.increment_tolerance:
            return result("CONVERGED_CUF_STRONG_FDM_VK",True,it,current_R,current_Rm,last_increment)
    return result("MAX_CUF_STRONG_FDM_VK_NEWTON_ITERATIONS",False,
                  newton_settings.max_iterations,current_R,current_Rm,last_increment)


def solve_vk_hooke_strong_fdm(
    initial_q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,
    settings,load_front_surface_mask=None
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    return solve_frozen_gb_vk_strong_fdm(
        initial_q,ops,basis,bc,E0=E0,K0=ec.K0,GB=ec.G0,
        quadrature=quadrature,generalized_load=generalized_load,
        newton_settings=settings,load_front_surface_mask=load_front_surface_mask
    )


def evaluate_vk_material_fullgrid(q,ops,basis,*,K0,GB,quadrature):
    q=fdl._validate_q(q,ops,basis)
    qx,qs=fdl.surface_first_derivatives(q,ops,basis)
    surf=(ops.grid.ns,ops.grid.nx); nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float); sig=np.empty_like(eps); det=np.ones_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    for ig,z in enumerate(quadrature.z):
        GBg=float(GBa) if mode=="scalar" else (GBa if mode=="surface" else GBa[ig])
        H,Ft,_Fp,abc=g7._batched_general_gradient_quantities(
            q,qx,qs,z,ops.grid.radius,basis
        )
        slope=gvk._batched_vk_slope(qx,qs,H=H,Ft=Ft)
        E=gvk._batched_vk_strain(abc,slope)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[ig]=g7._batched_strain_intensity(E)
        sig[ig]=g7._batched_stress_intensity(S)
    return b5.MaterialPointState(eps,sig,det)


def solve_vk_birger_strong_fdm(
    initial_q,initial_GB,*,ops,basis,bc,E0,nu0,quadrature,
    generalized_load,newton_settings,material_diagram,birger_settings,
    previous_accepted_epsilon_i=None,inner_counter=None,
    load_front_surface_mask=None
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    GB0=np.asarray(initial_GB,float)
    counter=inner_counter or CUFStrongFDMVKCounter()
    last={"result":None}
    def inner(state_guess,GB):
        counter.calls+=1
        rr=solve_frozen_gb_vk_strong_fdm(
            state_guess,ops,basis,bc,E0=E0,K0=ec.K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load,
            newton_settings=newton_settings,
            load_front_surface_mask=load_front_surface_mask
        )
        last["result"]=rr; return rr
    def evaluator(state,GB):
        return evaluate_vk_material_fullgrid(
            state,ops,basis,K0=ec.K0,GB=GB,quadrature=quadrature
        )
    out=b5.solve_birger_outer_cycle(
        initial_q,GB0,G0=ec.G0,material_diagram=material_diagram,
        inner_solve=inner,evaluate_material=evaluator,
        state_vector=lambda a:np.asarray(a,float).reshape(-1),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i
    )
    rr=last["result"]
    if rr is not None:
        setattr(out,"inner_history",list(rr.history))
        for name in ("residual_inf","residual_relative","residual_scale",
                     "increment_inf","residual_inf_interior",
                     "residual_inf_boundary","residual_inf_load_front",
                     "residual_convergence_inf","residual_convergence_relative",
                     "residual_inf_by_block"):
            setattr(out,name,getattr(rr,name))
    return out,counter
