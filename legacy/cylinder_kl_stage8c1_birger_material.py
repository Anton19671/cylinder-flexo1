#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8C-1: frozen-field plane-stress Birger material layer
for the displacement-only Kirchhoff-Love cylindrical shell.

This stage does NOT perform the Birger outer iteration.
It only evaluates, for a given frozen GB(z,s,x):

    K0 = const
    EB = 9 K0 GB / (3 K0 + GB)
    nuB = (3 K0 - 2 GB) / [2(3 K0 + GB)]

KL strains through thickness:
    eps_x(z)   = eps_x^0 + z*kappa_x
    eps_s(z)   = eps_s^0 + z*kappa_s
    gamma_xs(z)= gamma_xs^0 + 2 z*kappa_xs

Plane stress:
    eps_zz = -nuB/(1-nuB) * (eps_x + eps_s)

    sigma_x = EB/(1-nuB^2) * (eps_x + nuB eps_s)
    sigma_s = EB/(1-nuB^2) * (eps_s + nuB eps_x)
    tau_xs  = GB * gamma_xs

Equivalent intensity measures:
    epsilon_i = sqrt(2)/3 * sqrt(
        (eps_x-eps_s)^2
      + (eps_s-eps_zz)^2
      + (eps_zz-eps_x)^2
      + 1.5*gamma_xs^2
    )

    sigma_i = sqrt(
        sigma_x^2 - sigma_x*sigma_s + sigma_s^2 + 3*tau_xs^2
    )

For isotropic elastic response at fixed K0,GB:
    sigma_i = 3 GB epsilon_i

Resultants are obtained ONLY by thickness integration:
    Nx,Ns,Nxs = int sigma dz
    Mx,Ms,Mxs = int z sigma dz

No effective D is introduced in the Birger material path.
No outer update of GB here.
No MVI changes here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_stage8b2_profiles as kp
import cylinder_te1_stage5a_birger as b5


class KLBirgerMaterialError(RuntimeError):
    pass


@dataclass(frozen=True)
class KLBirgerFrozenResultants:
    resultants: kl.KLResultants
    material_state: b5.MaterialPointState
    EB: np.ndarray
    nuB: np.ndarray


def bulk_modulus_from_E_nu(E0, nu0):
    E0=float(E0); nu0=float(nu0)
    if not np.isfinite(E0) or E0<=0:
        raise ValueError("E0 must be finite and positive")
    if not np.isfinite(nu0) or not (-1.0 < nu0 < 0.5):
        raise ValueError("nu0 must satisfy -1 < nu0 < 0.5")
    return E0/(3.0*(1.0-2.0*nu0))


def shear_modulus_from_E_nu(E0, nu0):
    E0=float(E0); nu0=float(nu0)
    if not np.isfinite(E0) or E0<=0:
        raise ValueError("E0 must be finite and positive")
    if not np.isfinite(nu0) or not (-1.0 < nu0 < 0.5):
        raise ValueError("nu0 must satisfy -1 < nu0 < 0.5")
    return E0/(2.0*(1.0+nu0))


def E_nu_from_K_G(K0, GB):
    K0=float(K0)
    G=np.asarray(GB,float)
    if not np.isfinite(K0) or K0<=0:
        raise ValueError("K0 must be finite and positive")
    if np.any(~np.isfinite(G)) or np.any(G<=0):
        raise ValueError("GB must be finite and positive")
    den=3.0*K0+G
    E=9.0*K0*G/den
    nu=(3.0*K0-2.0*G)/(2.0*den)
    if np.any(~np.isfinite(E)) or np.any(~np.isfinite(nu)):
        raise KLBirgerMaterialError("non-finite E/nu reconstructed from K0,GB")
    if np.any(E<=0) or np.any(nu<=-1.0) or np.any(nu>=0.5):
        raise KLBirgerMaterialError("invalid E/nu reconstructed from K0,GB")
    return E,nu


def plane_stress_ezz(ex, es, nuB):
    ex=np.asarray(ex,float); es=np.asarray(es,float); nu=np.asarray(nuB,float)
    if ex.shape!=es.shape:
        raise ValueError("ex/es shape mismatch")
    return -nu/(1.0-nu)*(ex+es)


def plane_stress_stress_from_KG(ex, es, gamma, *, K0, GB):
    ex=np.asarray(ex,float); es=np.asarray(es,float); gam=np.asarray(gamma,float)
    if ex.shape!=es.shape or ex.shape!=gam.shape:
        raise ValueError("strain shape mismatch")
    E,nu=E_nu_from_K_G(K0,GB)
    c=E/(1.0-nu*nu)
    sx=c*(ex+nu*es)
    ss=c*(es+nu*ex)
    tau=np.asarray(GB,float)*gam
    return sx,ss,tau,E,nu


def strain_intensity_plane_stress(ex, es, gamma, nuB):
    ex=np.asarray(ex,float); es=np.asarray(es,float); gam=np.asarray(gamma,float)
    nu=np.asarray(nuB,float)
    ezz=plane_stress_ezz(ex,es,nu)
    rad=(
        (ex-es)**2
        +(es-ezz)**2
        +(ezz-ex)**2
        +1.5*gam**2
    )
    # protect only against tiny negative roundoff
    if np.any(rad < -1e-24):
        raise KLBirgerMaterialError("negative strain-intensity radicand")
    rad=np.maximum(rad,0.0)
    return np.sqrt(2.0)/3.0*np.sqrt(rad),ezz


def stress_intensity_plane_stress(sx, ss, tau):
    sx=np.asarray(sx,float); ss=np.asarray(ss,float); tau=np.asarray(tau,float)
    rad=sx*sx-sx*ss+ss*ss+3.0*tau*tau
    if np.any(rad < -1e-18):
        raise KLBirgerMaterialError("negative stress-intensity radicand")
    return np.sqrt(np.maximum(rad,0.0))


def _validate_GB_shape(GB, quadrature, surf_shape):
    G=np.asarray(GB,float)
    expected=(len(quadrature.z),)+tuple(surf_shape)
    if G.shape!=expected:
        raise ValueError(f"GB must have shape {expected}, got {G.shape}")
    if np.any(~np.isfinite(G)) or np.any(G<=0):
        raise ValueError("GB must be finite and positive")
    return G


def frozen_birger_resultants_from_kinematics(
    kin, *, K0, GB, quadrature
):
    """
    kin: KLFieldKinematics from Stage8A.
    GB: (nz,ns,nx), frozen during the whole inner MVI solve.
    """
    surf_shape=kin.state.eps_x.shape
    G=_validate_GB_shape(GB,quadrature,surf_shape)
    nz=len(quadrature.z)

    Nx=np.zeros(surf_shape,float)
    Ns=np.zeros(surf_shape,float)
    Nxs=np.zeros(surf_shape,float)
    Mx=np.zeros(surf_shape,float)
    Ms=np.zeros(surf_shape,float)
    Mxs=np.zeros(surf_shape,float)

    eps_i=np.empty((nz,)+surf_shape,float)
    sig_i=np.empty_like(eps_i)
    EB=np.empty_like(eps_i)
    nuB=np.empty_like(eps_i)
    detF=np.empty_like(eps_i)

    for g,(z,wz) in enumerate(zip(np.asarray(quadrature.z,float),
                                  np.asarray(quadrature.w,float))):
        ex,es,gam=kl.through_thickness_strains(kin.state,float(z))
        sx,ss,tau,Eb,nub=plane_stress_stress_from_KG(
            ex,es,gam,K0=K0,GB=G[g]
        )
        ei,ezz=strain_intensity_plane_stress(ex,es,gam,nub)
        si=stress_intensity_plane_stress(sx,ss,tau)
        det_c2=(1.0+2.0*ex)*(1.0+2.0*es)-gam*gam
        c33=1.0+2.0*ezz
        det_c3=det_c2*c33
        if np.any(~np.isfinite(det_c3)) or np.any(det_c3<=0.0):
            raise KLBirgerMaterialError(
                "non-positive reconstructed C determinant in full KL GL state"
            )
        detF[g]=np.sqrt(det_c3)

        Nx += wz*sx
        Ns += wz*ss
        Nxs += wz*tau
        Mx += wz*z*sx
        Ms += wz*z*ss
        Mxs += wz*z*tau

        eps_i[g]=ei
        sig_i[g]=si
        EB[g]=Eb
        nuB[g]=nub

    # detF is reconstructed consistently from the in-plane Green--Lagrange
    # metric and the plane-stress thickness strain; it is no longer a
    # placeholder equal to one in the full GL branch.
    nm=kl.KLResultants(Nx,Ns,Nxs,Mx,Ms,Mxs)
    mat=b5.MaterialPointState(eps_i,sig_i,detF)
    return KLBirgerFrozenResultants(nm,mat,EB,nuB)


def frozen_birger_resultants_tangent_from_kinematics(
    dkin, *, K0, GB, quadrature
):
    """Exact derivative of KL resultants for a strictly frozen GB field.

    For fixed K0 and GB, plane-stress stresses are linear in the KL strain
    measures.  Hence applying the same constitutive map to the directional
    strain state gives the exact resultant tangent.  Material intensities are
    deliberately not evaluated here because they belong to the outer Birger
    update, not to the frozen-GB inner Newton derivative.
    """
    surf_shape=dkin.state.eps_x.shape
    G=_validate_GB_shape(GB,quadrature,surf_shape)

    Nx=np.zeros(surf_shape,float)
    Ns=np.zeros(surf_shape,float)
    Nxs=np.zeros(surf_shape,float)
    Mx=np.zeros(surf_shape,float)
    Ms=np.zeros(surf_shape,float)
    Mxs=np.zeros(surf_shape,float)

    for g,(z,wz) in enumerate(zip(np.asarray(quadrature.z,float),
                                  np.asarray(quadrature.w,float))):
        dex,des,dgam=kl.through_thickness_strains(dkin.state,float(z))
        dsx,dss,dtau,_,_=plane_stress_stress_from_KG(
            dex,des,dgam,K0=K0,GB=G[g]
        )
        Nx += wz*dsx
        Ns += wz*dss
        Nxs += wz*dtau
        Mx += wz*z*dsx
        Ms += wz*z*dss
        Mxs += wz*z*dtau

    return kl.KLResultants(Nx,Ns,Nxs,Mx,Ms,Mxs)


def frozen_birger_resultants_from_profiles(
    state,ops,*,K0,GB,quadrature,geometry
):
    U=kp.reconstruct_fields(state)
    kin=kl.field_kinematics(U,ops,geometry=geometry)
    out=frozen_birger_resultants_from_kinematics(
        kin,K0=K0,GB=GB,quadrature=quadrature
    )
    return out,kin


def evaluate_material_points_from_profiles(
    state,ops,*,K0,GB,quadrature,geometry
):
    out,_=frozen_birger_resultants_from_profiles(
        state,ops,K0=K0,GB=GB,quadrature=quadrature,geometry=geometry
    )
    return out.material_state
