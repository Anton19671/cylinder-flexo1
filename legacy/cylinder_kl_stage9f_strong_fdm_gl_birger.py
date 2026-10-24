#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9F: independent STRONG-FORM KL FDM, GREEN-LAGRANGE + BIRGER.

Frozen-GB inner problem:
- full nodal U(s,x), no MVI/rank/X/Y;
- full midsurface GREEN-LAGRANGE KL kinematics;
- plane-stress Birger resultants integrated directly through thickness;
- strong collocation equations and direct end mechanical BC;
- nonlinear Newton solve with V3c exact frozen-GB analytic tangent.

Outer Birger:
the already verified generic Stage5A solve_birger_outer_cycle is reused
without rewriting its algorithm.

No line search, relaxation, retry, adaptive subdivision, regularization,
SVD/pinv/lstsq/QR/Cholesky, weak projection, FEM, or zones.
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

import cylinder_te1_stage5a_birger as b5
import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_full_gl_kinematics as fullgl
import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_kl_stage9b_strong_fdm_linear as fl
import cylinder_kl_stage9e_strong_fdm_gl_elastic as gl


class KLStrongFDMGLBirgerError(RuntimeError):
    pass


class KLStrongFDMGLBirgerSystemSingular(KLStrongFDMGLBirgerError):
    pass


@dataclass
class KLFrozenGBGLInnerResult:
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


@dataclass
class KLStrongFDMGLInnerCounter:
    calls: int = 0


def frozen_birger_strong_residual_gl(
    U,ops,bc,*,K0,GB,quadrature,generalized_load
):
    fl._validate_grid(ops)
    U=fl._validate_U(U,ops)
    bc.check_rigid_mode_requirement()

    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    kin=kl.field_kinematics(U,ops,geometry="GREEN_LAGRANGE")
    bout=bm.frozen_birger_resultants_from_kinematics(
        kin,K0=K0,GB=GB,quadrature=quadrature
    )
    nm=bout.resultants

    if kin.full_gl is None:
        raise KLStrongFDMGLBirgerError("full Green-Lagrange KL kinematics is missing")
    strong,_full_gl_coefficients=fullgl.variational_strong_residual(
        kin.full_gl,nm,ops,p
    )

    ql=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="GREEN_LAGRANGE",side="LEFT"
    )
    qr=kb.end_mechanical_quantities(
        kin,nm,ops,geometry="GREEN_LAGRANGE",side="RIGHT"
    )

    ns,nx=ops.grid.ns,ops.grid.nx

    lu=fl._bc_component_residual(bc.left.u,ql.u,ql.T_u,ns)
    ru=fl._bc_component_residual(bc.right.u,qr.u,qr.T_u,ns)
    lv=fl._bc_component_residual(bc.left.v,ql.v,ql.T_v,ns)
    rv=fl._bc_component_residual(bc.right.v,qr.v,qr.T_v,ns)
    lw=fl._bc_component_residual(bc.left.w,ql.w,ql.T_w,ns)
    rw=fl._bc_component_residual(bc.right.w,qr.w,qr.T_w,ns)
    ls=fl._bc_component_residual(
        bc.left.slope,ql.theta_n,ql.T_theta,ns
    )
    rs=fl._bc_component_residual(
        bc.right.slope,qr.theta_n,qr.T_theta,ns
    )

    rows=[]

    for js in range(ns):
        rows.append(lu[js])
        rows.extend(strong[0,js,1:nx-1])
        rows.append(ru[js])

    for js in range(ns):
        rows.append(lv[js])
        rows.extend(strong[1,js,1:nx-1])
        rows.append(rv[js])

    for js in range(ns):
        rows.append(lw[js])
        rows.append(ls[js])
        rows.extend(strong[2,js,2:nx-2])
        rows.append(rs[js])
        rows.append(rw[js])

    R=np.asarray(rows,float)
    expected_n=3*ns*nx
    if R.shape!=(expected_n,):
        raise KLStrongFDMGLBirgerError(
            f"residual size {R.shape}, expected ({expected_n},)"
        )
    return R,strong,bout,kin


def finite_difference_jacobian_frozen_gb(
    U,ops,bc,*,K0,GB,quadrature,generalized_load,fd_step,
    base_residual=None
):
    """V3 scaled central frozen-GB FD Jacobian; fd_step is relative."""
    U=fl._validate_U(U,ops)
    x=fl.flatten_U(U,ops)
    if base_residual is None:
        r0,*_=frozen_birger_strong_residual_gl(
            U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(x.size,):
            raise ValueError("base_residual size mismatch")

    h=v3fd.thickness_from_quadrature(quadrature)
    scales=v3fd.kl_characteristic_scale(U,h).reshape(-1)
    steps=v3fd.relative_steps(x,scales,fd_step)
    J=np.empty((x.size,x.size),float)
    for j,hj in enumerate(steps):
        xp=x.copy(); xm=x.copy()
        xp[j]+=hj; xm[j]-=hj
        rp,*_=frozen_birger_strong_residual_gl(
            fl.unflatten_U(xp,ops),ops,bc,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load
        )
        rm,*_=frozen_birger_strong_residual_gl(
            fl.unflatten_U(xm,ops),ops,bc,K0=K0,GB=GB,
            quadrature=quadrature,generalized_load=generalized_load
        )
        J[:,j]=(rp-rm)/(2.0*hj)
    return J,r0


def _prepare_exact_frozen_gb_tangent_action(
    U,ops,bc,*,K0,GB,quadrature
):
    """V3c exact dR(U;GB)[dU] for the verified frozen-GB inner solve."""
    U=fl._validate_U(U,ops)
    base_kin=kl.field_kinematics(U,ops,geometry="GREEN_LAGRANGE")
    base_bout=bm.frozen_birger_resultants_from_kinematics(
        base_kin,K0=K0,GB=GB,quadrature=quadrature
    )
    base_nm=base_bout.resultants
    ns,nx=ops.grid.ns,ops.grid.nx

    def action(dU):
        dkin=gl._exact_tangent_kinematics(dU,ops,base_kin)
        dnm=bm.frozen_birger_resultants_tangent_from_kinematics(
            dkin,K0=K0,GB=GB,quadrature=quadrature
        )
        dstrong=gl._exact_tangent_strong(
            base_kin,base_nm,dkin,dnm,ops
        )
        ql=gl._exact_tangent_end(
            base_kin,base_nm,dkin,dnm,ops,side="LEFT"
        )
        qr=gl._exact_tangent_end(
            base_kin,base_nm,dkin,dnm,ops,side="RIGHT"
        )
        lu=gl._bc_tangent_value(bc.left.u,ql['u'],ql['T_u'])
        ru=gl._bc_tangent_value(bc.right.u,qr['u'],qr['T_u'])
        lv=gl._bc_tangent_value(bc.left.v,ql['v'],ql['T_v'])
        rv=gl._bc_tangent_value(bc.right.v,qr['v'],qr['T_v'])
        lw=gl._bc_tangent_value(bc.left.w,ql['w'],ql['T_w'])
        rw=gl._bc_tangent_value(bc.right.w,qr['w'],qr['T_w'])
        ls=gl._bc_tangent_value(bc.left.slope,ql['theta_n'],ql['T_theta'])
        rs=gl._bc_tangent_value(bc.right.slope,qr['theta_n'],qr['T_theta'])
        rows=[]
        for js in range(ns):
            rows.append(lu[js]); rows.extend(dstrong[0,js,1:nx-1]); rows.append(ru[js])
        for js in range(ns):
            rows.append(lv[js]); rows.extend(dstrong[1,js,1:nx-1]); rows.append(rv[js])
        for js in range(ns):
            rows.append(lw[js]); rows.append(ls[js]); rows.extend(dstrong[2,js,2:nx-2]); rows.append(rs[js]); rows.append(rw[js])
        out=np.asarray(rows,float)
        if out.shape!=(3*ns*nx,):
            raise KLStrongFDMGLBirgerError("analytic tangent residual size mismatch")
        return out
    return action


def analytic_tangent_action_frozen_gb(
    U,dU,ops,bc,*,K0,GB,quadrature,generalized_load=None
):
    """Exact V3c KL Green-Lagrange Birger tangent action at fixed GB."""
    action=_prepare_exact_frozen_gb_tangent_action(
        U,ops,bc,K0=K0,GB=GB,quadrature=quadrature
    )
    return action(dU)


def analytic_jacobian_frozen_gb(
    U,ops,bc,*,K0,GB,quadrature,generalized_load,base_residual=None
):
    """Exact V3c frozen-GB KL Jacobian; outer material update is untouched."""
    U=fl._validate_U(U,ops)
    action=_prepare_exact_frozen_gb_tangent_action(
        U,ops,bc,K0=K0,GB=GB,quadrature=quadrature
    )
    J,stats=v3an.assemble_dense_exact_tangent(U,action)
    analytic_jacobian_frozen_gb.last_stats=stats.to_dict()
    if base_residual is None:
        r0,*_=frozen_birger_strong_residual_gl(
            U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(U.size,):
            raise ValueError("base_residual size mismatch")
    return J,r0


analytic_jacobian_frozen_gb.last_stats=None


def solve_frozen_gb_gl_strong_fdm(
    initial_U,ops,bc,*,K0,GB,quadrature,generalized_load,newton_settings,
    load_front_surface_mask=None
):
    U=fl._validate_U(initial_U,ops).copy()
    history=[]
    last_increment=np.inf

    # V2: compute the relative-residual scale once, before Newton updates.
    R_initial,*_=frozen_birger_strong_residual_gl(
        U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    residual_scale=rscale.residual_scale(generalized_load,R_initial)
    residual_limit=rscale.threshold(newton_settings.residual_tolerance_relative,residual_scale)
    region_masks=rregions.resolve_front_exclusion(rregions.kl_masks(ops,load_front_surface_mask),generalized_load)

    for k in range(1,newton_settings.max_iterations+1):
        # V4-BIRGER: this residual was always tested before the Newton update,
        # but previously only *after* an expensive FD Jacobian had already
        # been assembled.  Test it first and reuse the same base residual in J.
        # This changes no Newton criterion, update, material law or Birger cycle.
        if k==1:
            R=R_initial
        else:
            R,*_=frozen_birger_strong_residual_gl(
                U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
                generalized_load=generalized_load
            )
        rm=rregions.metrics(R,region_masks,residual_scale)
        rinf=rm.residual_inf

        if rregions.converged(rm,residual_limit):
            return KLFrozenGBGLInnerResult(
                state=U,
                converged=True,
                status="CONVERGED_STRONG_FDM_FROZEN_GB_GL",
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
            )

        # Differentiate the complete exact-normal/exact-curvature residual.
        # The previous analytic action belonged to the obsolete hybrid model.
        J,_=finite_difference_jacobian_frozen_gb(
            U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,
            fd_step=newton_settings.fd_step,base_residual=R
        )

        try:
            dx,_,_,_=nscale.equilibrated_solve(J,-R)
        except np.linalg.LinAlgError as exc:
            raise KLStrongFDMGLBirgerSystemSingular(
                f"SINGULAR_STRONG_FDM_FROZEN_GB_GL_JACOBIAN / iteration={k}"
            ) from exc

        if not np.all(np.isfinite(dx)):
            raise KLStrongFDMGLBirgerError(
                f"NONFINITE_STRONG_FDM_FROZEN_GB_GL_INCREMENT / iteration={k}"
            )

        last_increment=float(np.max(np.abs(dx)))
        U=fl.unflatten_U(fl.flatten_U(U,ops)+dx,ops)

        Rnew,*_=frozen_birger_strong_residual_gl(
            U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load
        )
        rmnew=rregions.metrics(Rnew,region_masks,residual_scale)
        rnew=rmnew.residual_inf

        hrow={"iteration":k,"residual_scale":residual_scale,"increment_inf":last_increment}
        hrow.update(rregions.as_dict(rmnew))
        history.append(hrow)
        _w_live=np.asarray(U[2],float)
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
            return KLFrozenGBGLInnerResult(
                state=U,
                converged=True,
                status="CONVERGED_STRONG_FDM_FROZEN_GB_GL",
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
            )

    Rfin,*_=frozen_birger_strong_residual_gl(
        U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
        generalized_load=generalized_load
    )
    rmfin=rregions.metrics(Rfin,region_masks,residual_scale)
    return KLFrozenGBGLInnerResult(
        state=U,
        converged=False,
        status="MAX_STRONG_FDM_FROZEN_GB_GL_NEWTON_ITERATIONS",
        iterations=newton_settings.max_iterations,
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
    )


def evaluate_material_state_gl(
    U,ops,*,K0,GB,quadrature
):
    U=fl._validate_U(U,ops)
    kin=kl.field_kinematics(U,ops,geometry="GREEN_LAGRANGE")
    bout=bm.frozen_birger_resultants_from_kinematics(
        kin,K0=K0,GB=GB,quadrature=quadrature
    )
    return bout.material_state


def make_inner_solver(
    ops,bc,*,K0,quadrature,generalized_load,newton_settings,counter=None,
    load_front_surface_mask=None
):
    if counter is None:
        counter=KLStrongFDMGLInnerCounter()

    def inner(state_guess,GB):
        counter.calls+=1
        result=solve_frozen_gb_gl_strong_fdm(
            state_guess,ops,bc,
            K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,
            newton_settings=newton_settings,
            load_front_surface_mask=load_front_surface_mask
        )
        inner.last_result=result
        return result

    inner.counter=counter
    inner.last_result=None
    return inner


def make_material_evaluator(
    ops,*,K0,quadrature
):
    def evaluate(U,GB):
        return evaluate_material_state_gl(
            U,ops,K0=K0,GB=GB,quadrature=quadrature
        )
    return evaluate


def solve_gl_birger_strong_fdm(
    initial_U,initial_GB,
    *,
    ops,bc,E0,nu0,h,quadrature,generalized_load,
    newton_settings,material_diagram,birger_settings,
    previous_accepted_epsilon_i=None,
    inner_counter=None,
    load_front_surface_mask=None,
):
    initial_U=fl._validate_U(initial_U,ops)
    E0=float(E0); nu0=float(nu0); h=float(h)
    if E0<=0 or h<=0 or not (-1.0<nu0<0.5):
        raise ValueError("invalid E0/nu0/h")

    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)

    expected=(len(quadrature.z),ops.grid.ns,ops.grid.nx)
    GB0=np.asarray(initial_GB,float)
    if GB0.shape!=expected:
        raise ValueError(f"initial_GB must have shape {expected}")
    if np.any(~np.isfinite(GB0)) or np.any(GB0<=0):
        raise ValueError("initial_GB must be finite and positive")

    inner=make_inner_solver(
        ops,bc,K0=K0,quadrature=quadrature,
        generalized_load=generalized_load,
        newton_settings=newton_settings,
        counter=inner_counter,
        load_front_surface_mask=load_front_surface_mask
    )
    evaluator=make_material_evaluator(
        ops,K0=K0,quadrature=quadrature
    )

    result=b5.solve_birger_outer_cycle(
        initial_U,GB0,
        G0=G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: np.asarray(st,float).reshape(-1),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
    if inner.last_result is not None:
        setattr(result,"inner_history",list(getattr(inner.last_result,"history",[]) or []))
        for name in (
            "residual_inf","residual_relative","residual_scale","increment_inf",
            "residual_inf_interior","residual_inf_boundary","residual_inf_load_front",
            "residual_convergence_inf","residual_convergence_relative"
        ):
            setattr(result,name,getattr(inner.last_result,name))
    return result,inner.counter
