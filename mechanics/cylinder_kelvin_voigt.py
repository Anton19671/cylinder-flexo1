#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact finite-strain Kelvin-Voigt weak action for seven G3D shell models.

The corrected Krysko-Zhong manuscript, sections 12 and 21, specifies
**bulk** St. Venant-Kirchhoff-type Kelvin-Voigt viscosity, not gradient
viscosity. For Green-Lagrange E=1/2(F.T F-I), dotE=sym(F.T Fdot),

  S_v=eta_s * (lambda tr(dotE) I + 2 G dotE).
  R_v=eta_s/2 * (lambda tr(dotE)**2 + 2G dotE:dotE) >= 0.
  P_diss = S_v:dotE = 2 R_v.

eta_s is a *time* [s], and is independent of Rayleigh alpha_M [1/s].
The mechanical residual is the weak first variation with respect to
virtual velocity, equivalently int S_v:DE(q)[delta q] dV0, evaluated
at the CURRENT nonlinear configuration q.

The same exact model-specific nonlinear deformation-gradient maps used
by G3D elastic/flexoelectric enthalpy are reused below:
  KL          analytic finite-rotation normal; complex step of GEOMETRY;
  FSDT/Reddy  independent curved director and warping;
  CUF TE1..4  dimensionless 3D thickness polynomials.

All Gauss and surface weights are the same as the conservative
functional. No electric viscosity or additional gradient viscosity is
invented, and no viscous potential is added to conservative enthalpy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from flexo_gradient_g3d import G3DMaterial
from cuf_g3d_weak import (
    MECHANICAL_VARIATIONS as _CUF_FIELDS,
    _kinematic_matrix_jets as _cuf_F,
    _unit_variation as _cuf_unit,
    gauss_thickness,
)
from cylinder_director_g3d import (
    FIELDS as _DIRECTOR_FIELDS,
    _F_jets as _director_F,
    _unit as _director_unit,
)
from cylinder_kl_g3d_weak import (
    FIELDS as _KL_FIELDS,
    _F_jets_raw as _kl_F,
)


@dataclass(frozen=True)
class KelvinVoigtWeak:
    """Local mechanical coefficients and physical dissipation per mid-area."""
    mechanical: np.ndarray   # (n_mechanical, n_surface_derivatives), N
    power_w_per_m2: float     # W/m^2, >= 0 up to roundoff
    rayleigh_w_per_m2: float  # W/m^2, exactly power/2


@dataclass(frozen=True)
class KelvinVoigtGlobal:
    """Assembled force (N) and dissipated power (W); no electrical term."""
    mechanical_force: np.ndarray
    power_w: float


def _real_nonnegative(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a real nonnegative number")
    try:
        x=float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a real nonnegative number") from exc
    if not math.isfinite(x) or x<0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return x


def _de(F,dF):
    """Engineering-shear Voigt of sym(F.T dF), work conjugate to sigma6."""
    sym=.5*(F.T@dF+dF.T@F)
    return np.array((sym[0,0],sym[1,1],sym[2,2],
                     2*sym[0,1],2*sym[0,2],2*sym[1,2]),float)


def viscous_stress_and_power(strain_rate6, material, eta_s):
    """Return engineering-conjugate 2PK stress and local dissipation W/m^3.

    The six-vector stores gamma_ij=2E_ij. Thus sigma6[3:]=G*eta*gamma_dot,
    NOT 2G*eta*gamma_dot. The scalar sigma6.dot(e_dot6) is exactly S:dotE.
    """
    eta=_real_nonnegative(eta_s,"micro.eta_s")
    if not isinstance(material,G3DMaterial):
        raise TypeError("G3DMaterial required for Kelvin-Voigt constitutive law")
    e=np.asarray(strain_rate6,float)
    if e.shape!=(6,) or not np.isfinite(e).all():
        raise ValueError("finite six-component Green-Lagrange rate required")
    lam,G=material.lame_lambda,material.shear_modulus
    sigma=np.empty(6)
    sigma[:3]=eta*(lam*float(np.sum(e[:3]))+2*G*e[:3])
    sigma[3:]=eta*G*e[3:]
    power=float(np.dot(sigma,e))
    if not math.isfinite(power):
        raise ValueError("nonfinite Kelvin-Voigt dissipation")
    # Positive definiteness of the isotropic C makes power >=0 analytically.
    if power<-1.e-12*max(1.,float(np.max(np.abs(sigma),initial=0))):
        raise ValueError("NEGATIVE_KELVIN_VOIGT_DISSIPATION")
    return sigma,max(0.,power)


def _kl_direction(base, delta, R, h, z):
    """Directional derivative of holomorphic KL geometry only.

    Identical complex-step geometric differentiation is already used for
    all ten jet coefficients in KL G3D enthalpy first variations.
    """
    eps=1.e-26
    perturbed={name:np.asarray(base[name],complex)+
               1j*eps*np.asarray(delta[name],float) for name in _KL_FIELDS}
    return np.asarray(_kl_F(perturbed,R,h,z)[0].imag/eps,float)


def kelvin_voigt_thickness_weak(qjets, vjets, material, *, theory,
                                radius_m, thickness_m, eta_s,
                                n_gauss=12):
    """Quadrature-exact Kelvin-Voigt local weak coefficients for one model.

    All generalized coefficients are first variations with respect to the
    velocity's virtual surface jets. For nonlinear q, this equals the
    virtual-displacement work of S_v in the momentum balance. Both
    Green-Lagrange rate and virtual strain use the CURRENT F(q).
    """
    eta=_real_nonnegative(eta_s,"micro.eta_s")
    R,h=float(radius_m),float(thickness_m)
    if not math.isfinite(R) or not math.isfinite(h) or h<=0 or R<=h/2:
        raise ValueError("require R > h/2 > 0 for viscous thickness integration")
    if not isinstance(material,G3DMaterial):
        raise TypeError("G3DMaterial required")
    theory=str(theory).upper()
    if theory.startswith("CUF_TE") and theory[-1] in "1234":
        names=_CUF_FIELDS
        ncoef=qjets.te_order+1
        if int(theory[-1])!=qjets.te_order or vjets.te_order!=qjets.te_order:
            raise ValueError("CUF viscous TE order mismatch")
        nfield=3*ncoef
        if eta:
            unit=lambda comp,k: _cuf_unit(qjets,comp//ncoef,
                                          comp%ncoef,names[k])
    elif theory in ("FSDT","REDDY"):
        names=_DIRECTOR_FIELDS
        nfield=5
        if eta:
            unit=lambda comp,k: _director_unit(qjets,names[k],comp)
    elif theory=="KL":
        names=_KL_FIELDS
        nfield=3
    else:
        raise ValueError("unknown shell theory for Kelvin-Voigt weak action")
    coefficient=np.zeros((nfield,len(names)),float)
    if eta==0:
        return KelvinVoigtWeak(coefficient,0.,0.)
    zs,weights=gauss_thickness(h,n_gauss)
    power=0.
    if theory=="KL":
        base={name:getattr(qjets,name) for name in names}
        direction={name:getattr(vjets,name) for name in names}
    for z,w in zip(zs,weights):
        z=float(z)
        H=1.+z/R
        if theory.startswith("CUF_TE"):
            F=_cuf_F(qjets,z,R,h)[0]
            Fdot=_cuf_F(vjets,z,R,h,variation=True)[0]
            def virtual(comp,k):
                return _cuf_F(unit(comp,k),z,R,h,variation=True)[0]
        elif theory in ("FSDT","REDDY"):
            F=_director_F(qjets,theory,z,R,h)[0]
            Fdot=_director_F(vjets,theory,z,R,h,virtual=True)[0]
            def virtual(comp,k):
                return _director_F(unit(comp,k),theory,z,R,h,virtual=True)[0]
        else:
            F=_kl_F(base,R,h,z)[0]
            Fdot=_kl_direction(base,direction,R,h,z)
            def virtual(comp,k):
                delta={name:np.zeros_like(base[name]) for name in names}
                delta[names[k]][comp]=1.
                return _kl_direction(base,delta,R,h,z)
        det=float(np.linalg.det(F))
        if not math.isfinite(det) or det<=0:
            raise ValueError("INVALID_DEFORMATION_MAPPING in Kelvin-Voigt shell")
        sigma,dissipation=viscous_stress_and_power(_de(F,Fdot),material,eta)
        power+=float(w)*H*dissipation
        for k in range(len(names)):
            for comp in range(nfield):
                delta_e=_de(F,virtual(comp,k))
                coefficient[comp,k]+=float(w)*H*float(np.dot(sigma,delta_e))
    if not np.isfinite(coefficient).all() or not math.isfinite(power):
        raise ValueError("INVALID_KELVIN_VOIGT_WEAK_ACTION")
    return KelvinVoigtWeak(coefficient,float(power),float(.5*power))


__all__=("KelvinVoigtWeak","KelvinVoigtGlobal",
         "viscous_stress_and_power","kelvin_voigt_thickness_weak")
