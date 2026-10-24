#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""General CUF moderate-rotation von Karman mechanics for a cylinder.

This is a separate mechanics branch from full Green--Lagrange/StVK.

The already verified general CUF expansion and cylindrical linear strains are
retained.  Only the Krysko-school moderate-rotation products used by the KL
control branch are added:

    exx += 1/2 * w_x**2
    ess += 1/2 * w_s**2
    exs += 1/2 * w_x*w_s

At a thickness point ``w_x`` and ``w_s`` are reconstructed from the selected
CUF basis.  The linear circumferential strain still contains ``w/R`` and the
metric factor ``H=1+z/R``.  The quadratic slope uses the radial displacement
derivative itself, exactly as the verified KL branch; it does not silently
re-introduce the full Green--Lagrange ``(w_s-v/R)`` product.

Material law at fixed K0/GB is ordinary isotropic Hooke/Birger in the small
strain tensor above.  The virtual-work resultants and their exact directional
tangent are derived from the same energy.  No StVK deformation gradient and
no detF admissibility test belong to this moderate-rotation branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import cylinder_te1_local_kernel as lk
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as gp
import cylinder_cuf_stage7c_general_boundary as gbc
import cylinder_cuf_stage7d_general_internal_mvi as gmvi
import cylinder_cuf_stage7h_general_variational as gvar


@dataclass(frozen=True)
class GeneralVKLocalResult:
    H: float
    abc: np.ndarray
    E: np.ndarray
    strain_intensity: float
    S: np.ndarray
    stress_intensity: float


def general_vk_local_kernel(q,qx,qs,z,R,K0,GB,basis):
    """Scalar material-point form used by postprocessing/self-checks."""
    H,abc=g7.general_gradient_quantities(q,qx,qs,z,R,basis)
    Ft=basis.values(z)
    slope=np.array([float(Ft@np.asarray(qx,float)[2]),
                    float(Ft@np.asarray(qs,float)[2])/H,0.0])
    E=0.5*(abc+abc.T)+0.5*np.outer(slope,slope)
    S=lk.birger_second_piola(E,float(K0),float(GB))
    return GeneralVKLocalResult(
        H,abc,E,lk.strain_intensity_from_E(E),S,
        lk.stress_intensity_from_S(S)
    )


def _batched_vk_slope(qx, qs, *, H, Ft):
    """Return [w_x,w_s/H,0] at one thickness point."""
    qx=np.asarray(qx,float); qs=np.asarray(qs,float)
    wx=np.einsum('t,t...->...',Ft,qx[2],optimize=False)
    ws=np.einsum('t,t...->...',Ft,qs[2],optimize=False)/float(H)
    out=np.zeros((3,)+wx.shape,float)
    out[0]=wx; out[1]=ws
    return out


def _batched_vk_strain(abc, slope):
    E=g7._batched_linear_strain_from_abc(abc)
    E=E+0.5*np.einsum('i...,j...->ij...',slope,slope,optimize=False)
    return E


def _add_vk_radial_work(acc, traction, *, H, Ft, wz, surface_shape):
    """Add the nonlinear slope variation only to radial derivative rows."""
    shp=(Ft.size,)+(1,)*len(surface_shape)
    FtB=np.asarray(Ft,float).reshape(shp)
    acc["Aw_x"] += float(wz)*(float(H)*FtB*traction[0])
    acc["Aw_s"] += float(wz)*(FtB*traction[1])


def general_vk_thickness_resultants(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """CUF von-Karman Hooke/Birger resultants, batched on the surface."""
    q,qx,qs=g7._check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R<=0.0 or K0<=0.0:
        raise ValueError("R and K0 must be positive")
    surface_shape=q.shape[2:]
    GBa,mode=g7._gb_mode(GB,quadrature.z.size,surface_shape)
    acc={k:np.zeros((basis.m,)+surface_shape,float) for k in g7.RESULTANT_KEYS}

    for ig,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
        GBg=float(GBa) if mode=="scalar" else (GBa if mode=="surface" else GBa[ig])
        H,Ft,Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        slope=_batched_vk_slope(qx,qs,H=H,Ft=Ft)
        E=_batched_vk_strain(abc,slope)
        S=g7._batched_birger_second_piola(E,K0,GBg)

        # Linear strain variation: P=S in the general CUF work coefficients.
        g7._accumulate_general_resultants(
            acc,S,H,Ft,Fp,float(wz),R,surface_shape
        )
        # Quadratic slope variation: delta W_nl=(slope.S).delta(slope).
        radial=np.einsum('i...,ij...->j...',slope,S,optimize=False)
        _add_vk_radial_work(
            acc,radial,H=H,Ft=Ft,wz=wz,surface_shape=surface_shape
        )
    return g7.GeneralResultants(**acc)


@dataclass(frozen=True)
class VKPoint:
    wz: float
    H: float
    Ft: np.ndarray
    Fp: np.ndarray
    S: np.ndarray
    slope: np.ndarray
    GB: object


@dataclass(frozen=True)
class VKTangentContext:
    R: float
    K0: float
    basis: object
    surface_shape: tuple
    points: tuple


def prepare_vk_tangent_context(q,qx,qs,*,R,K0,GB,quadrature,basis):
    q,qx,qs=g7._check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    surface_shape=q.shape[2:]
    GBa,mode=g7._gb_mode(GB,quadrature.z.size,surface_shape)
    points=[]
    for ig,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
        GBg=float(GBa) if mode=="scalar" else (GBa if mode=="surface" else GBa[ig])
        H,Ft,Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        slope=_batched_vk_slope(qx,qs,H=H,Ft=Ft)
        E=_batched_vk_strain(abc,slope)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        points.append(VKPoint(float(wz),float(H),np.asarray(Ft,float),
                              np.asarray(Fp,float),np.asarray(S,float),
                              np.asarray(slope,float),GBg))
    return VKTangentContext(R,K0,basis,surface_shape,tuple(points))


def vk_resultants_tangent_from_context(ctx,dq,dqx,dqs):
    dq,dqx,dqs=g7._check_general_surface_arrays(dq,dqx,dqs,ctx.basis)
    if dq.shape[2:]!=ctx.surface_shape:
        raise ValueError("tangent direction surface shape mismatch")
    acc={k:np.zeros((ctx.basis.m,)+ctx.surface_shape,float) for k in g7.RESULTANT_KEYS}
    for pt in ctx.points:
        dabc=g7._batched_general_gradient_direction(
            dq,dqx,dqs,H=pt.H,Ft=pt.Ft,Fp=pt.Fp,R=ctx.R
        )
        dslope=_batched_vk_slope(dqx,dqs,H=pt.H,Ft=pt.Ft)
        dE=g7._batched_linear_strain_from_abc(dabc)
        dE=dE+0.5*(
            np.einsum('i...,j...->ij...',dslope,pt.slope,optimize=False)
            +np.einsum('i...,j...->ij...',pt.slope,dslope,optimize=False)
        )
        dS=g7._batched_birger_second_piola(dE,ctx.K0,pt.GB)
        g7._accumulate_general_resultants(
            acc,dS,pt.H,pt.Ft,pt.Fp,pt.wz,ctx.R,ctx.surface_shape
        )
        dradial=(
            np.einsum('i...,ij...->j...',dslope,pt.S,optimize=False)
            +np.einsum('i...,ij...->j...',pt.slope,dS,optimize=False)
        )
        _add_vk_radial_work(
            acc,dradial,H=pt.H,Ft=pt.Ft,wz=pt.wz,
            surface_shape=ctx.surface_shape
        )
    return g7.GeneralResultants(**acc)


def vk_raw_residual_from_profiles(
    state,ops,*,K0,GB,quadrature,basis,generalized_load
):
    q,qx,qs=gp.profile_fields_and_first_derivatives(state,ops)
    resultants=general_vk_thickness_resultants(
        q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
        quadrature=quadrature,basis=basis
    )
    raw=g7.general_residuals(
        resultants,Dx=ops.Dx,Ds=ops.Ds,generalized_load=generalized_load
    )
    return raw,resultants,(q,qx,qs)


def make_vk_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    def residual(state):
        raw,resultants,(q,_qx,_qs)=vk_raw_residual_from_profiles(
            state,ops,K0=K0,GB=GB,quadrature=quadrature,basis=basis,
            generalized_load=p
        )
        return gbc.assemble_strong_residual_with_end_bcs(
            raw,q,resultants,bc
        ).reshape(-1)

    def prepare_exact_tangent_action(state):
        q,qx,qs=gp.profile_fields_and_first_derivatives(state,ops)
        ctx=prepare_vk_tangent_context(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        zload=np.zeros_like(q)
        expected=q.shape
        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(expected)
            dqx,dqs=gmvi._surface_first_derivatives_full_direction(dq,ops,basis)
            dres=vk_resultants_tangent_from_context(ctx,dq,dqx,dqs)
            draw=g7.general_residuals(
                dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
            )
            return gmvi._assemble_exact_tangent_with_end_bcs(
                draw,dq,dres,bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    residual.mvi_exact_tangent_profile_coloring=True
    residual.mvi_tangent_mode="EXACT_VON_KARMAN_FROZEN_GB"
    return residual


def make_vk_variational_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    """Discrete first variation for von-Karman CUF at fixed elastic moduli."""
    gvar._require_geometric_ends(bc)
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")

    def residual(state):
        q,qx,qs=gp.profile_fields_and_first_derivatives(state,ops)
        resultants=general_vk_thickness_resultants(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        return gvar.weak_residual_from_resultants(
            resultants,ops=ops,generalized_load=p,bc=bc
        ).reshape(-1)

    def prepare_exact_tangent_action(state):
        q,qx,qs=gp.profile_fields_and_first_derivatives(state,ops)
        ctx=prepare_vk_tangent_context(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        shape=q.shape
        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(shape)
            dqx,dqs=gvar._surface_first_derivatives_full_direction(dq,ops)
            dres=vk_resultants_tangent_from_context(ctx,dq,dqx,dqs)
            return gvar.weak_tangent_from_resultants(
                dres,dq,ops=ops,bc=bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    # Variational assembly uses D^T; strong-form coloring support is not a
    # proven disjoint support for one-sided boundary stencils.
    residual.mvi_exact_tangent_profile_coloring=False
    residual.mvi_tangent_mode="EXACT_VARIATIONAL_VON_KARMAN_FROZEN_GB"
    residual.mvi_projected_reference_residual=gvar.generalized_load_vector(
        ops=ops,generalized_load=p,bc=bc
    ).reshape(-1)
    residual.mvi_formulation="VARIATIONAL"
    residual.mvi_residual_already_integrated=True
    return residual


def make_vk_hooke_residual_function(
    ops,*,E0,nu0,quadrature,basis,generalized_load,bc,formulation="STRONG"
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    form=str(formulation).upper()
    if form=="STRONG":
        return make_vk_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    if form=="VARIATIONAL":
        return make_vk_variational_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    raise ValueError("MVI formulation must be STRONG or VARIATIONAL")


def solve_general_vk_hooke_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,validate_initial_seed=True
):
    fn=make_vk_hooke_residual_function(
        ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
        generalized_load=generalized_load,bc=bc,
        formulation=inner_settings.formulation
    )
    if validate_initial_seed:
        import cylinder_cuf_stage7e_general_driver as driver
        driver.require_seed_if_nonzero_residual(initial_state,ops.grid,fn)
    import cylinder_cuf_stage7e_general_driver as driver
    projection_weights=(
        np.ones(initial_state.n_fields*ops.grid.n_surface_nodes,dtype=float)
        if str(inner_settings.formulation).upper()=="VARIATIONAL"
        else None
    )
    return gmvi.solve_internal_mvi(
        initial_state,ops.grid,bc,fn,inner_settings,
        weights=projection_weights,
        measure_multipliers=driver._v4_measure_multipliers(
            ops,basis,bc,E0=E0,quadrature=quadrature
        ),quadrature=quadrature
    )


def evaluate_general_vk_material_points(state,ops,*,K0,GB,quadrature,basis):
    q,qx,qs=gp.profile_fields_and_first_derivatives(state,ops)
    surf=(ops.grid.ns,ops.grid.nx); nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float); sig=np.empty_like(eps)
    det=np.ones_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    for ig,z in enumerate(quadrature.z):
        GBg=float(GBa) if mode=="scalar" else (GBa if mode=="surface" else GBa[ig])
        H,Ft,_Fp,abc=g7._batched_general_gradient_quantities(
            q,qx,qs,z,ops.grid.radius,basis
        )
        slope=_batched_vk_slope(qx,qs,H=H,Ft=Ft)
        E=_batched_vk_strain(abc,slope)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[ig]=g7._batched_strain_intensity(E)
        sig[ig]=g7._batched_stress_intensity(S)
    return b5.MaterialPointState(eps,sig,det)


def solve_general_vk_birger_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,birger_settings,material_diagram,initial_GB=None,
    previous_accepted_epsilon_i=None,validate_initial_seed=True
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    shape=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    GB0=(np.full(shape,ec.G0,float) if initial_GB is None else
         np.broadcast_to(np.asarray(initial_GB,float),shape).astype(float,copy=True))
    p=np.asarray(generalized_load,float)

    def inner(st,GBfield):
        fn=make_vk_residual_function(
            ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
        import cylinder_cuf_stage7e_general_driver as driver
        if validate_initial_seed:
            driver.require_seed_if_nonzero_residual(st,ops.grid,fn)
        return gmvi.solve_internal_mvi(
            st,ops.grid,bc,fn,inner_settings,
            measure_multipliers=driver._v4_measure_multipliers(
                ops,basis,bc,E0=E0,quadrature=quadrature
            ),quadrature=quadrature
        )

    def evaluator(st,GBfield):
        return evaluate_general_vk_material_points(
            st,ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis
        )

    return b5.solve_birger_outer_cycle(
        initial_state,GB0,G0=ec.G0,material_diagram=material_diagram,
        inner_solve=inner,evaluate_material=evaluator,
        state_vector=lambda st: gp.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i
    )
