#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local CUF-TE1 kinematics/material kernel for a closed circular cylindrical shell.

This module contains NO literature benchmark data and performs NO MVI/MKR solve.
It implements only the already verified local relations:
H=1+z/R; TE1 displacement reconstruction; F; Green-Lagrange E;
Birger frozen-secant constitutive law S=K0 tr(E)I+2*GB*dev(E);
P=F*S; strain/stress intensities; det(F).

CONSTITUTIVE SETTING (important):
S = K0*tr(E)*I + 2*GB(eps_i)*dev(E) is applied in the finite-deformation
pair E = Green-Lagrange strain and S = second Piola-Kirchhoff stress; eps_i
and sigma_i are the invariants of those tensors. At small strain this reduces
to the classical Birger method of variable elasticity parameters. At finite
strain it is a St.Venant-Kirchhoff-type generalization with a variable shear
modulus, and identifying Phi with a small-strain experimental intensity
diagram requires separate justification. K5 checks only the slope at
eps_i -> 0.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

I3 = np.eye(3, dtype=float)

@dataclass(frozen=True)
class ElasticConstants:
    E0: float
    nu0: float

    def __post_init__(self):
        if self.E0 <= 0.0:
            raise ValueError("E0 must be positive")
        if not (-1.0 < self.nu0 < 0.5):
            raise ValueError("nu0 must satisfy -1 < nu0 < 0.5")

    @property
    def G0(self):
        return self.E0 / (2.0 * (1.0 + self.nu0))

    @property
    def K0(self):
        return self.E0 / (3.0 * (1.0 - 2.0 * self.nu0))

    @property
    def lame_lambda(self):
        return self.K0 - 2.0*self.G0/3.0

@dataclass(frozen=True)
class LocalResult:
    H: float
    abc: np.ndarray       # [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    F: np.ndarray
    E: np.ndarray
    strain_components: np.ndarray  # Exx,Ess,Ezz,gamma_xs,gamma_xz,gamma_sz
    strain_intensity: float
    S: np.ndarray
    stress_intensity: float
    P: np.ndarray
    detF: float

def _as6(a, name):
    a = np.asarray(a, dtype=float)
    if a.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), order [u0,u1,v0,v1,w0,w1]")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a

def dev(A):
    A = np.asarray(A, dtype=float)
    return A - np.trace(A)*I3/3.0

def strain_intensity_from_E(E):
    """
    eps_i = sqrt((2/3) e:e), e=dev(E).
    Tensor form is used directly, so engineering shear cannot be inserted
    with a wrong factor of two.
    """
    e = dev(E)
    val = (2.0/3.0) * float(np.tensordot(e, e, axes=2))
    return float(np.sqrt(max(val, 0.0)))

def stress_intensity_from_S(S):
    """
    sigma_i = sqrt((3/2) s:s), s=dev(S).
    """
    s = dev(S)
    val = (3.0/2.0) * float(np.tensordot(s, s, axes=2))
    return float(np.sqrt(max(val, 0.0)))

def te1_gradient_quantities(q, qx, qs, z, R):
    """
    q, qx, qs order: [u0,u1,v0,v1,w0,w1].
    s is arc length on the midsurface.
    """
    q = _as6(q, "q")
    qx = _as6(qx, "qx")
    qs = _as6(qs, "qs")
    z = float(z); R = float(R)
    if R <= 0.0:
        raise ValueError("R must be positive")
    H = 1.0 + z/R
    if H <= 0.0:
        raise ValueError("H=1+z/R must be positive")

    u0,u1,v0,v1,w0,w1 = q
    _u = u0 + z*u1
    u0x,u1x,v0x,v1x,w0x,w1x = qx
    u0s,u1s,v0s,v1s,w0s,w1s = qs

    v = v0 + z*v1
    w = w0 + z*w1

    ux = u0x + z*u1x
    vx = v0x + z*v1x
    wx = w0x + z*w1x

    us = u0s + z*u1s
    vs = v0s + z*v1s
    ws = w0s + z*w1s

    a1 = ux
    b1 = vx
    c1 = wx

    a2 = us / H
    b2 = (vs + w/R) / H
    c2 = (ws - v/R) / H

    a3 = u1
    b3 = v1
    c3 = w1

    abc = np.array([
        [a1,a2,a3],
        [b1,b2,b3],
        [c1,c2,c3],
    ], dtype=float)
    return H, abc

def deformation_gradient_from_abc(abc):
    abc = np.asarray(abc, dtype=float)
    if abc.shape != (3,3):
        raise ValueError("abc must be 3x3")
    F = abc.copy()
    F[0,0] += 1.0
    F[1,1] += 1.0
    F[2,2] += 1.0
    return F

def green_lagrange(F):
    F = np.asarray(F, dtype=float)
    if F.shape != (3,3):
        raise ValueError("F must be 3x3")
    return 0.5*(F.T @ F - I3)

def expanded_green_components(abc):
    """
    Independent component expansion of E used only for verification.
    Returns [Exx,Ess,Ezz,gamma_xs,gamma_xz,gamma_sz].
    """
    a1,a2,a3 = abc[0]
    b1,b2,b3 = abc[1]
    c1,c2,c3 = abc[2]

    Exx = a1 + 0.5*(a1*a1 + b1*b1 + c1*c1)
    Ess = b2 + 0.5*(a2*a2 + b2*b2 + c2*c2)
    Ezz = c3 + 0.5*(a3*a3 + b3*b3 + c3*c3)

    gxs = a2 + b1 + a1*a2 + b1*b2 + c1*c2
    gxz = a3 + c1 + a1*a3 + b1*b3 + c1*c3
    gsz = b3 + c2 + a2*a3 + b2*b3 + c2*c3
    return np.array([Exx,Ess,Ezz,gxs,gxz,gsz], dtype=float)

def components_from_E(E):
    return np.array([
        E[0,0], E[1,1], E[2,2],
        2.0*E[0,1], 2.0*E[0,2], 2.0*E[1,2]
    ], dtype=float)

def birger_second_piola(E, K0, GB):
    K0 = float(K0); GB = float(GB)
    if K0 <= 0.0 or GB <= 0.0:
        raise ValueError("K0 and frozen GB must be positive")
    theta = float(np.trace(E))
    return K0*theta*I3 + 2.0*GB*dev(E)

def local_te1_kernel(q, qx, qs, z, R, K0, GB):
    H, abc = te1_gradient_quantities(q, qx, qs, z, R)
    F = deformation_gradient_from_abc(abc)
    E = green_lagrange(F)
    S = birger_second_piola(E, K0, GB)
    P = F @ S
    return LocalResult(
        H=H,
        abc=abc,
        F=F,
        E=E,
        strain_components=components_from_E(E),
        strain_intensity=strain_intensity_from_E(E),
        S=S,
        stress_intensity=stress_intensity_from_S(S),
        P=P,
        detF=float(np.linalg.det(F)),
    )
