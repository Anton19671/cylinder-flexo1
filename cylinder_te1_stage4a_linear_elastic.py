#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_te1_local_kernel as nk
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage3b_boundary as b3
import cylinder_te1_stage3c_internal_mvi as c4

I3=np.eye(3,dtype=float)

@dataclass(frozen=True)
class LinearLocalResult:
    H: float
    abc: np.ndarray
    E: np.ndarray
    strain_components: np.ndarray
    S: np.ndarray
    P: np.ndarray

def linear_strain_from_abc(abc):
    abc=np.asarray(abc,float)
    if abc.shape!=(3,3): raise ValueError("abc must be 3x3")
    a1,a2,a3=abc[0]; b1,b2,b3=abc[1]; c1,c2,c3_=abc[2]
    Exx=a1; Ess=b2; Err=c3_
    gxs=a2+b1; gxr=a3+c1; gsr=b3+c2
    return np.array([[Exx,.5*gxs,.5*gxr],
                     [.5*gxs,Ess,.5*gsr],
                     [.5*gxr,.5*gsr,Err]],float)

def linear_components_from_E(E):
    E=np.asarray(E,float)
    return np.array([E[0,0],E[1,1],E[2,2],2*E[0,1],2*E[0,2],2*E[1,2]],float)

def linear_isotropic_second_piola(E,E0,nu0):
    ec=nk.ElasticConstants(float(E0),float(nu0))
    return ec.lame_lambda*np.trace(E)*I3+2.0*ec.G0*E

def linear_local_te1(q,qx,qs,z,R,E0,nu0):
    H,abc=nk.te1_gradient_quantities(q,qx,qs,z,R)
    E=linear_strain_from_abc(abc)
    S=linear_isotropic_second_piola(E,E0,nu0)
    P=S.copy()  # Word eq. (556): P^L=S^L.
    return LinearLocalResult(H,abc,E,linear_components_from_E(E),S,P)

def linear_thickness_cache_te1(q,qx,qs,*,R,E0,nu0,quadrature):
    q=np.asarray(q,float); qx=np.asarray(qx,float); qs=np.asarray(qs,float)
    if q.shape!=qx.shape or q.shape!=qs.shape or q.ndim<1 or q.shape[0]!=6:
        raise ValueError("q/qx/qs must have identical shape (6,...)")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qs))):
        raise ValueError("non-finite q/qx/qs")
    surf_shape=q.shape[1:]
    acc={k:np.zeros(surf_shape,float) for k in s2.CACHE_KEYS}
    for idx in np.ndindex(surf_shape):
        q6=q[(slice(None),)+idx]; qx6=qx[(slice(None),)+idx]; qs6=qs[(slice(None),)+idx]
        for z,wz in zip(quadrature.z,quadrature.w):
            rr=linear_local_te1(q6,qx6,qs6,z,R,E0,nu0)
            H=rr.H; P=rr.P
            Pxx,Pxs,Pxr=P[0,0],P[0,1],P[0,2]
            Psx,Pss,Psr=P[1,0],P[1,1],P[1,2]
            Prx,Prs,Prr=P[2,0],P[2,1],P[2,2]
            acc["Ixx0"][idx]+=wz*(H*Pxx); acc["Ixx1"][idx]+=wz*(H*z*Pxx)
            acc["Ixs0"][idx]+=wz*Pxs; acc["Ixs1"][idx]+=wz*(z*Pxs)
            acc["Ixr1"][idx]+=wz*(H*Pxr)
            acc["Isx0"][idx]+=wz*(H*Psx); acc["Isx1"][idx]+=wz*(H*z*Psx)
            acc["Iss0"][idx]+=wz*Pss; acc["Iss1"][idx]+=wz*(z*Pss)
            acc["Isr1"][idx]+=wz*(H*Psr)
            acc["Irs0"][idx]+=wz*Prs; acc["Irs1"][idx]+=wz*(z*Prs)
            acc["Irx0"][idx]+=wz*(H*Prx); acc["Irx1"][idx]+=wz*(H*z*Prx)
            acc["Irr1"][idx]+=wz*(H*Prr)
    return s2.ThicknessCache(**acc)

def linear_raw_strong_residual_from_profiles(state,ops,*,E0,nu0,quadrature,generalized_load=None):
    q,qx,qs=c3.profile_fields_and_first_derivatives(state,ops)
    cache=linear_thickness_cache_te1(q,qx,qs,R=ops.grid.radius,E0=E0,nu0=nu0,quadrature=quadrature)
    if generalized_load is None:
        generalized_load=s2.zero_generalized_load((ops.grid.ns,ops.grid.nx))
    residual=s2.te1_residuals(cache,Dx=ops.Dx,Ds=ops.Ds,R=ops.grid.radius,generalized_load=generalized_load)
    return residual,cache,(q,qx,qs)

def make_linear_elastic_residual_function(ops,*,E0,nu0,quadrature,generalized_load,bc):
    p=np.asarray(generalized_load,float)
    expected=(6,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected: raise ValueError(f"generalized_load must have shape {expected}")
    def residual(state):
        raw,cache,(q,_qx,_qs)=linear_raw_strong_residual_from_profiles(
            state,ops,E0=E0,nu0=nu0,quadrature=quadrature,generalized_load=p)
        return b3.assemble_strong_residual_with_end_bcs(raw,q,cache,bc).reshape(-1)
    return residual

def solve_linear_elastic_mvi(initial_state,ops,*,E0,nu0,quadrature,generalized_load,bc,settings):
    fn=make_linear_elastic_residual_function(
        ops,E0=E0,nu0=nu0,quadrature=quadrature,generalized_load=generalized_load,bc=bc)
    return c4.solve_internal_mvi(
        initial_state,ops.grid,bc,fn,settings,quadrature=quadrature
    )
