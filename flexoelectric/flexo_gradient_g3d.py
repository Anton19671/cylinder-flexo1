#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verified local material algebra for the Krysko-Zhong G3D model.

Source: Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx,
sections 6-11, equations (19)-(55). SI reference configuration.

This is NOT a coupled shell solver: no mechanical/electrical field discretization,
no higher-order boundary conditions, no dynamic time stepping are supplied here.
The six engineering strain components are [ex, es, ez, gamma_xs, gamma_xz,
gamma_sz]. The 6x3 covariant strain-gradient array has directions [x,s,z].
For material.gradient_model="GZ", only the six physical z-direction
entries contribute to gradient energy and flexoelectric polarization.
The complete geometric gradient is still computed for transparent
postprocessing; the constitutive 6x3 hyperstress is projected onto z.
G3D remains the default for historical material constructors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class G3DMaterial:
    young_pa: float
    poisson: float
    ell_m: float
    mu1_c_per_m: float
    mu2_c_per_m: float
    kappa_f_per_m: float
    gradient_model: str = "G3D"

    def __post_init__(self):
        for name in ("young_pa", "poisson", "ell_m", "mu1_c_per_m",
                     "mu2_c_per_m", "kappa_f_per_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.young_pa <= 0 or not -1.0 < self.poisson < 0.5:
            raise ValueError("E > 0 and -1 < nu < 0.5 are required")
        if self.ell_m < 0 or self.kappa_f_per_m < 0:
            raise ValueError("ell >= 0 and kappa >= 0 are required")
        if self.kappa_f_per_m == 0 and (self.mu1_c_per_m != 0 or self.mu2_c_per_m != 0):
            raise ValueError("kappa=0 requires mu1=mu2=0 (electric terms off)")
        if not isinstance(self.gradient_model,str):
            raise ValueError("gradient_model must be G3D or Gz")
        mode=self.gradient_model.strip().upper()
        if mode not in ("G3D","GZ"):
            raise ValueError("gradient_model must be G3D or Gz")
        object.__setattr__(self,"gradient_model",mode)

    @property
    def active_gradient_components(self):
        """Physical strain-gradient entries participating in the enthalpy."""
        return 18 if self.gradient_model=="G3D" else 6

    @property
    def gradient_model_label(self):
        return "G3D" if self.gradient_model=="G3D" else "Gz"

    @property
    def lame_lambda(self):
        return self.young_pa*self.poisson / ((1+self.poisson)*(1-2*self.poisson))

    @property
    def shear_modulus(self):
        return self.young_pa/(2*(1+self.poisson))


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return value


def covariant_strain_gradients(strain6, dx6, ds6, dz6, *, radius_m, z_m):
    """Exactly eqs. (19)-(28), including cylindrical rotating-basis terms.

    dx6/ds6/dz6 are ordinary partials in reference x,s,z coordinates;
    column order of the result is physical x, s, z.
    """
    e = _array(strain6, (6,), "strain6")
    dx = _array(dx6, (6,), "dx6")
    ds = _array(ds6, (6,), "ds6")
    dz = _array(dz6, (6,), "dz6")
    radius_m, z_m = float(radius_m), float(z_m)
    if not (math.isfinite(radius_m) and math.isfinite(z_m)
            and radius_m > 0 and radius_m + z_m > 0):
        raise ValueError("require R>0 and R+z>0")
    H = 1.0 + z_m/radius_m
    invr = 1.0/(radius_m+z_m)
    grad = np.empty((6, 3), dtype=float)
    grad[:, 0] = dx
    grad[:, 2] = dz
    grad[:, 1] = ds/H
    grad[1, 1] += e[5]*invr
    grad[2, 1] -= e[5]*invr
    grad[3, 1] += e[4]*invr
    grad[4, 1] -= e[3]*invr
    grad[5, 1] += 2.0*(e[2]-e[1])*invr
    return grad


def active_strain_gradients(grad6x3, gradient_model="G3D"):
    """Material gradient selector; never changes the underlying shell kinematics.

    G3D retains all covariant x,s,z components. Gz keeps only the six
    through-thickness strain derivatives g[:,2]. This projection must
    precede BOTH scalar enthalpy and its analytical first variation,
    otherwise a false in-plane hyperstress remains in the weak form.
    """
    g=_array(grad6x3,(6,3),"gradient")
    if not isinstance(gradient_model,str):
        raise ValueError("gradient_model must be G3D or Gz")
    mode=gradient_model.strip().upper()
    if mode=="G3D":
        return g
    if mode!="GZ":
        raise ValueError("gradient_model must be G3D or Gz")
    active=np.zeros((6,3),dtype=float)
    active[:,2]=g[:,2]
    return active


def gradient_invariants(grad6x3):
    """Eqs. (29)-(33): t_k and b_i = covariant div E (E_xs=gamma_xs/2)."""
    g = _array(grad6x3, (6, 3), "gradient")
    t = g[:3, :].sum(axis=0)
    b = np.array((
        g[0,0] + .5*g[3,1] + .5*g[4,2],
        .5*g[3,0] + g[1,1] + .5*g[5,2],
        .5*g[4,0] + .5*g[5,1] + g[2,2],
    ), dtype=float)
    return t, b


def reference_electric_field(phi_x, phi_s, phi_z, *, radius_m, z_m):
    """Eq. (34): reference electric field with the exact H(z) factor."""
    radius_m, z_m = float(radius_m), float(z_m)
    if not (math.isfinite(radius_m) and math.isfinite(z_m)
            and radius_m > 0 and radius_m+z_m > 0):
        raise ValueError("require R>0 and R+z>0")
    deriv = _array([phi_x, phi_s, phi_z], (3,), "phi derivatives")
    return -deriv / np.array((1.0, 1+z_m/radius_m, 1.0))


def enthalpy_parts(strain6, grad6x3, electric3, material: G3DMaterial):
    """Eqs. (37)-(42), W_C+W_G+W_E+W_F per reference volume."""
    e = _array(strain6, (6,), "strain6")
    g = _array(grad6x3, (6,3), "gradient")
    electric = _array(electric3, (3,), "electric3")
    if not isinstance(material, G3DMaterial):
        raise TypeError("material must be G3DMaterial")
    lam, G = material.lame_lambda, material.shear_modulus
    ell2 = material.ell_m**2
    g = active_strain_gradients(g, material.gradient_model)
    t, b = gradient_invariants(g)
    WC = .5*lam*e[:3].sum()**2 + G*(np.dot(e[:3],e[:3])+.5*np.dot(e[3:],e[3:]))
    WG = ell2*(.5*lam*np.dot(t,t) + G*(np.sum(g[:3]**2) + .5*np.sum(g[3:]**2)))
    WE = -.5*material.kappa_f_per_m*np.dot(electric,electric)
    WF = -material.mu1_c_per_m*np.dot(electric,t) - material.mu2_c_per_m*np.dot(electric,b)
    return {"classical":float(WC), "gradient":float(WG), "dielectric":float(WE),
            "flexoelectric":float(WF), "total":float(WC+WG+WE+WF)}


def enthalpy_derivatives(strain6, grad6x3, electric3, material: G3DMaterial):
    """Eqs. (43)-(55): dH/de6, dH/dgrad6x3 and D=-dH/dElectric."""
    e = _array(strain6, (6,), "strain6")
    g = _array(grad6x3, (6,3), "gradient")
    electric = _array(electric3, (3,), "electric3")
    if not isinstance(material, G3DMaterial):
        raise TypeError("material must be G3DMaterial")
    lam, G = material.lame_lambda, material.shear_modulus
    ell2, m1, m2 = material.ell_m**2, material.mu1_c_per_m, material.mu2_c_per_m
    g = active_strain_gradients(g, material.gradient_model)
    t, b = gradient_invariants(g)
    sigma = np.empty(6,float)
    sigma[:3] = lam*np.sum(e[:3]) + 2*G*e[:3]
    sigma[3:] = G*e[3:]
    hyper = np.zeros((6,3),float)
    for k in range(3):
        hyper[:3,k] = ell2*(lam*t[k] + 2*G*g[:3,k]) - m1*electric[k]
        hyper[k,k] -= m2*electric[k]
    hyper[3:] = ell2*G*g[3:]
    # Engineering shear components: m_(xs,k), m_(xz,k), m_(sz,k).
    # k runs [x,s,z]; the 1/2 prefactor follows eqs. (49)-(51).
    for row,(i,j) in enumerate(((0,1),(0,2),(1,2)),start=3):
        hyper[row,j] -= .5*m2*electric[i]
        hyper[row,i] -= .5*m2*electric[j]
    if material.gradient_model=="GZ":
        # Chain rule for H(E, P_z grad(E)): dH/dgrad(E)=P_z dH/d(P_z grad(E)).
        # Without this, E_x/E_s would incorrectly generate hyperstress in
        # ignored x/s gradients despite their absence from the Gz energy.
        hyper[:,:2]=0.
    D = material.kappa_f_per_m*electric + m1*t + m2*b
    return sigma, hyper, D


__all__ = ["G3DMaterial", "covariant_strain_gradients", "active_strain_gradients", "gradient_invariants",
           "reference_electric_field", "enthalpy_parts", "enthalpy_derivatives"]
