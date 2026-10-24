#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7F: GENERAL CUF linear-geometry branches.

Adds the two missing general-CUF combinations:
  CUF_LINEAR_ELASTIC
  CUF_LINEAR_BIRGER

Linear kinematics are the exact linear part of the verified general cylindrical
CUF operator:
  Exx=a1, Ess=b2, Err=c3,
  gamma_xs=a2+b1, gamma_xr=a3+c1, gamma_sr=b3+c2.

For linear geometry:
  P^L = S^L.

Elastic:
  S = lambda tr(E^L) I + 2 G0 E^L.

Birger:
  S = K0 tr(E^L) I + 2 GB dev(E^L),
  with GB frozen inside each MVI equilibrium solve and updated only by the
  verified Stage 5A outer material cycle.

General thickness resultants use the same general CUF virtual-work coefficients
as Stage 7A, but with P=P^L=S^L.

No higher-order basis is invented. No solver algorithm is changed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_te1_local_kernel as lk
import cylinder_te1_stage4a_linear_elastic as l4
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7b_general_mvi_profiles as g8
import cylinder_cuf_stage7c_general_boundary as g9
import cylinder_cuf_stage7d_general_internal_mvi as g10
import cylinder_cuf_stage7e_general_driver as g11
import cylinder_cuf_stage7h_general_variational as g14


@dataclass(frozen=True)
class GeneralLinearLocalResult:
    H: float
    abc: np.ndarray
    E: np.ndarray
    strain_intensity: float
    S: np.ndarray
    stress_intensity: float
    P: np.ndarray


def general_linear_local_kernel(q,qx,qs,z,R,K0,GB,basis):
    H,abc=g7.general_gradient_quantities(q,qx,qs,z,R,basis)
    E=l4.linear_strain_from_abc(abc)
    S=lk.birger_second_piola(E,float(K0),float(GB))
    P=S.copy()
    return GeneralLinearLocalResult(
        H=H,
        abc=abc,
        E=E,
        strain_intensity=lk.strain_intensity_from_E(E),
        S=S,
        stress_intensity=lk.stress_intensity_from_S(S),
        P=P,
    )


def general_linear_elastic_local_kernel(q,qx,qs,z,R,E0,nu0,basis):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    return general_linear_local_kernel(
        q,qx,qs,z,R,ec.K0,ec.G0,basis
    )


def _gb_value(GB,g,idx,nz,surface_shape):
    a=np.asarray(GB,float)
    if a.ndim==0:
        return float(a)
    if a.shape==surface_shape:
        return float(a[idx])
    if a.shape==(nz,)+surface_shape:
        return float(a[(g,)+idx])
    raise ValueError(
        f"GB must be scalar, surface-shaped {surface_shape}, "
        f"or {(nz,)+surface_shape}; got {a.shape}"
    )


def general_linear_thickness_resultants_scalar_reference(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """Pre-K2b scalar linear-CUF reference retained only for regression tests."""
    q,qx,qs=g7._check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R<=0.0 or K0<=0.0:
        raise ValueError("R and K0 must be positive")

    surface_shape=q.shape[2:]
    nz=quadrature.z.size
    acc={
        k:np.zeros((basis.m,)+surface_shape,float)
        for k in g7.RESULTANT_KEYS
    }

    for idx in np.ndindex(surface_shape):
        qpt=q[(slice(None),slice(None))+idx]
        qxpt=qx[(slice(None),slice(None))+idx]
        qspt=qs[(slice(None),slice(None))+idx]
        for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
            GBg=_gb_value(GB,g,idx,nz,surface_shape)
            rr=general_linear_local_kernel(
                qpt,qxpt,qspt,z,R,K0,GBg,basis
            )
            H=rr.H
            Ft=basis.values(z)
            Fp=basis.derivatives(z)
            P=rr.P

            Pxx,Pxs,Pxr=P[0,0],P[0,1],P[0,2]
            Psx,Pss,Psr=P[1,0],P[1,1],P[1,2]
            Prx,Prs,Prr=P[2,0],P[2,1],P[2,2]
            sl=(slice(None),)+idx

            acc["Au_x"][sl]+=wz*(H*Ft*Pxx)
            acc["Au_s"][sl]+=wz*(Ft*Pxs)
            acc["Cu"][sl]+=wz*(H*Fp*Pxr)
            acc["Av_x"][sl]+=wz*(H*Ft*Psx)
            acc["Av_s"][sl]+=wz*(Ft*Pss)
            acc["Cv"][sl]+=wz*(H*Fp*Psr-(Ft/R)*Prs)
            acc["Aw_x"][sl]+=wz*(H*Ft*Prx)
            acc["Aw_s"][sl]+=wz*(Ft*Prs)
            acc["Cw"][sl]+=wz*(H*Fp*Prr+(Ft/R)*Pss)

    return g7.GeneralResultants(**acc)


def general_linear_thickness_resultants(
    q,qx,qs,*,R,K0,GB,quadrature,basis
):
    """K2b production path for linear CUF: batch all surface nodes."""
    q,qx,qs=g7._check_general_surface_arrays(q,qx,qs,basis)
    R=float(R); K0=float(K0)
    if R<=0.0 or K0<=0.0:
        raise ValueError("R and K0 must be positive")

    surface_shape=q.shape[2:]
    nz=quadrature.z.size
    GBa,mode=g7._gb_mode(GB,nz,surface_shape)
    acc={
        k:np.zeros((basis.m,)+surface_shape,float)
        for k in g7.RESULTANT_KEYS
    }

    for g,(z,wz) in enumerate(zip(quadrature.z,quadrature.w)):
        if mode=="scalar":
            GBg=float(GBa)
        elif mode=="surface":
            GBg=GBa
        else:
            GBg=GBa[g]
        H,Ft,Fp,abc=g7._batched_general_gradient_quantities(
            q,qx,qs,z,R,basis
        )
        E=g7._batched_linear_strain_from_abc(abc)
        P=g7._batched_birger_second_piola(E,K0,GBg)
        g7._accumulate_general_resultants(
            acc,P,H,Ft,Fp,float(wz),R,surface_shape
        )

    return g7.GeneralResultants(**acc)


def linear_raw_residual_from_profiles(
    state,ops,*,K0,GB,quadrature,basis,generalized_load
):
    if state.m!=basis.m:
        raise ValueError("state/basis m mismatch")
    q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
    resultants=general_linear_thickness_resultants(
        q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
        quadrature=quadrature,basis=basis
    )
    residual=g7.general_residuals(
        resultants,Dx=ops.Dx,Ds=ops.Ds,
        generalized_load=generalized_load
    )
    return residual,resultants,(q,qx,qs)


def make_linear_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")

    def residual(state):
        raw,resultants,(q,_qx,_qs)=linear_raw_residual_from_profiles(
            state,ops,K0=K0,GB=GB,quadrature=quadrature,
            basis=basis,generalized_load=p
        )
        return g9.assemble_strong_residual_with_end_bcs(
            raw,q,resultants,bc
        ).reshape(-1)

    def prepare_exact_tangent_action(_state):
        # At fixed K0/GB the LINEAR residual is affine in q, hence this is its
        # exact derivative and does not require any state-dependent base cache.
        zload=np.zeros(expected,float)
        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(expected)
            dqx,dqs=g10._surface_first_derivatives_full_direction(dq,ops,basis)
            dres=general_linear_thickness_resultants(
                dq,dqx,dqs,R=ops.grid.radius,K0=K0,GB=GB,
                quadrature=quadrature,basis=basis
            )
            draw=g7.general_residuals(
                dres,Dx=ops.Dx,Ds=ops.Ds,generalized_load=zload
            )
            return g10._assemble_exact_tangent_with_end_bcs(
                draw,dq,dres,bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    residual.mvi_exact_tangent_profile_coloring=True
    residual.mvi_tangent_mode="EXACT_LINEAR_FROZEN_GB"
    return residual


def make_linear_variational_residual_function(
    ops,*,K0,GB,quadrature,basis,generalized_load,bc
):
    """Discrete first variation for LINEAR CUF at fixed elastic moduli."""
    g14._require_geometric_ends(bc)
    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if bc.m!=basis.m:
        raise ValueError("basis/BC m mismatch")

    def residual(state):
        q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
        resultants=general_linear_thickness_resultants(
            q,qx,qs,R=ops.grid.radius,K0=K0,GB=GB,
            quadrature=quadrature,basis=basis
        )
        return g14.weak_residual_from_resultants(
            resultants,ops=ops,generalized_load=p,bc=bc
        ).reshape(-1)

    def prepare_exact_tangent_action(_state):
        shape=expected
        def action(dqvec):
            dq=np.asarray(dqvec,float).reshape(shape)
            dqx,dqs=g14._surface_first_derivatives_full_direction(dq,ops)
            dres=general_linear_thickness_resultants(
                dq,dqx,dqs,R=ops.grid.radius,K0=K0,GB=GB,
                quadrature=quadrature,basis=basis
            )
            return g14.weak_tangent_from_resultants(
                dres,dq,ops=ops,bc=bc
            ).reshape(-1)
        return action

    residual.mvi_prepare_exact_tangent_action=prepare_exact_tangent_action
    residual.mvi_exact_tangent_ops=ops
    # Variational assembly uses D^T; strong-form coloring support is not a
    # proven disjoint support for one-sided boundary stencils.
    residual.mvi_exact_tangent_profile_coloring=False
    residual.mvi_tangent_mode="EXACT_VARIATIONAL_LINEAR_FROZEN_GB"
    residual.mvi_projected_reference_residual=g14.generalized_load_vector(
        ops=ops,generalized_load=p,bc=bc
    ).reshape(-1)
    residual.mvi_formulation="VARIATIONAL"
    residual.mvi_residual_already_integrated=True
    return residual


def make_linear_elastic_residual_function(
    ops,*,E0,nu0,quadrature,basis,generalized_load,bc,formulation="STRONG"
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    form=str(formulation).upper()
    if form=="STRONG":
        return make_linear_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    if form=="VARIATIONAL":
        return make_linear_variational_residual_function(
            ops,K0=ec.K0,GB=ec.G0,quadrature=quadrature,basis=basis,
            generalized_load=generalized_load,bc=bc
        )
    raise ValueError("MVI formulation must be STRONG or VARIATIONAL")


def solve_general_linear_elastic_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,validate_initial_seed=True
):
    fn=make_linear_elastic_residual_function(
        ops,E0=E0,nu0=nu0,quadrature=quadrature,basis=basis,
        generalized_load=generalized_load,bc=bc,
        formulation=inner_settings.formulation
    )
    if validate_initial_seed:
        g11.require_seed_if_nonzero_residual(
            initial_state,ops.grid,fn
        )
    projection_weights=(
        np.ones(initial_state.n_fields*ops.grid.n_surface_nodes,dtype=float)
        if str(inner_settings.formulation).upper()=="VARIATIONAL"
        else None
    )
    return g10.solve_internal_mvi(
        initial_state,ops.grid,bc,fn,inner_settings,
        weights=projection_weights,
        measure_multipliers=g11._v4_measure_multipliers(
            ops,basis,bc,E0=E0,quadrature=quadrature
        ),
        quadrature=quadrature
    )


def evaluate_general_linear_material_points(
    state,ops,*,K0,GB,quadrature,basis
):
    """K2b batched material-point evaluation for linear CUF MVI."""
    q,qx,qs=g8.profile_fields_and_first_derivatives(state,ops)
    surf=(ops.grid.ns,ops.grid.nx)
    nz=quadrature.z.size
    eps=np.empty((nz,)+surf,float)
    sig=np.empty_like(eps)
    det=np.ones_like(eps)
    GBa,mode=g7._gb_mode(GB,nz,surf)
    R=float(ops.grid.radius); K0=float(K0)

    for g,z in enumerate(quadrature.z):
        if mode=="scalar": GBg=float(GBa)
        elif mode=="surface": GBg=GBa
        else: GBg=GBa[g]
        _H,_Ft,_Fp,abc=g7._batched_general_gradient_quantities(q,qx,qs,z,R,basis)
        E=g7._batched_linear_strain_from_abc(abc)
        S=g7._batched_birger_second_piola(E,K0,GBg)
        eps[g]=g7._batched_strain_intensity(E)
        sig[g]=g7._batched_stress_intensity(S)
    return b5.MaterialPointState(eps,sig,det)


def solve_general_linear_birger_load_step(
    initial_state,ops,*,E0,nu0,quadrature,basis,generalized_load,
    bc,inner_settings,birger_settings,material_diagram,
    initial_GB=None,previous_accepted_epsilon_i=None,
    validate_initial_seed=True
):
    ec=lk.ElasticConstants(float(E0),float(nu0))
    gbshape=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    if initial_GB is None:
        GB0=np.full(gbshape,ec.G0,float)
    else:
        GB0=np.broadcast_to(
            np.asarray(initial_GB,float),gbshape
        ).astype(float,copy=True)

    p=np.asarray(generalized_load,float)
    expected=(3,basis.m,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def inner(st,GBfield):
        fn=make_linear_residual_function(
            ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis,
            generalized_load=p,bc=bc
        )
        if validate_initial_seed:
            g11.require_seed_if_nonzero_residual(
                st,ops.grid,fn
            )
        return g10.solve_internal_mvi(
            st,ops.grid,bc,fn,inner_settings,
            measure_multipliers=g11._v4_measure_multipliers(
                ops,basis,bc,E0=E0,quadrature=quadrature
            ),
            quadrature=quadrature
        )

    def evaluator(st,GBfield):
        return evaluate_general_linear_material_points(
            st,ops,K0=ec.K0,GB=GBfield,quadrature=quadrature,basis=basis
        )

    return b5.solve_birger_outer_cycle(
        initial_state,GB0,G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: g8.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
