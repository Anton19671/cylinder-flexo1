#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Krysko-Zhong covariant G3D CUF weak-form coefficients, TE1--TE4.

Source: Krysko_Zhong_TE1_TE4_PARALLEL_ENDS_CORRECTED_20261024.docx,
sections 3--11 and 13--24, eqs. (1)--(135).

This module evaluates the exact *local surface weak first variation* of the
mechanical/electric enthalpy after Gauss thickness integration. It is NOT a
spatial discretization, higher-order boundary-condition assembler, or a time
integrator. Both forward and inverse flexoelectric terms are included through
the SAME material enthalpy. Nothing here enables MICRO dispatch.

Mechanical weak coefficient indices:
    mechanical[component, TE-index, variation]
    components [u, v, w], variations [q, q_x, q_s, q_xx, q_xs, q_ss].
Electrical weak coefficient indices:
    electrical[potential-index, variation]
    variations [psi, psi_x, psi_s].

All quantities are SI and referenced to the undeformed cylinder. The
thickness basis is F_i(z)=(2*z/h)**i, NOT the old solver's z**i basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from flexo_gradient_g3d import (
    G3DMaterial, covariant_strain_gradients, reference_electric_field,
    enthalpy_parts, enthalpy_derivatives,
)


MECHANICAL_VARIATIONS = ("q", "qx", "qs", "qxx", "qxs", "qss")
ELECTRICAL_VARIATIONS = ("psi", "psix", "psis")
_MECH_FIELDS = MECHANICAL_VARIATIONS
_ELEC_FIELDS = ELECTRICAL_VARIATIONS
_EYE = np.eye(3)


@dataclass(frozen=True)
class CUFSurfaceJets:
    """Local surface values/derivatives, order [u,v,w] x [0..N].

    Only one mixed derivative qxs is supplied. This enforces the no-spurious-
    factor-of-two convention in the mixed weak coefficient (source eq. 109).
    """
    q: np.ndarray
    qx: np.ndarray
    qs: np.ndarray
    qxx: np.ndarray
    qxs: np.ndarray
    qss: np.ndarray
    psi: np.ndarray
    psix: np.ndarray
    psis: np.ndarray

    def __post_init__(self):
        base = np.asarray(self.q, dtype=float)
        if base.ndim != 2 or base.shape[0] != 3 or not (2 <= base.shape[1] <= 5):
            raise ValueError("q must have shape (3, N+1) for CUF TE1..TE4")
        for name in _MECH_FIELDS:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != base.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} requires finite shape {base.shape}")
            object.__setattr__(self, name, a.copy())
        p = np.asarray(self.psi, dtype=float)
        if p.ndim != 1 or not (1 <= p.size <= 5):
            raise ValueError("psi must have shape (P+1,) with P=0..4")
        for name in _ELEC_FIELDS:
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != p.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} requires finite shape {p.shape}")
            object.__setattr__(self, name, a.copy())

    @property
    def te_order(self):
        return self.q.shape[1] - 1

    @property
    def potential_order(self):
        return self.psi.size - 1

    def replaced(self, **updates):
        """Copy jets with explicitly replaced arrays (useful for variations)."""
        return CUFSurfaceJets(**{
            field: updates.get(field, getattr(self, field))
            for field in (*_MECH_FIELDS, *_ELEC_FIELDS)
        })


@dataclass(frozen=True)
class PointState:
    z_m: float
    H: float
    deformation_gradient: np.ndarray
    strain6: np.ndarray
    gradient6x3: np.ndarray
    electric3: np.ndarray
    det_f: float


@dataclass(frozen=True)
class WeakCoefficients:
    energy_j_per_m2: float
    mechanical: np.ndarray  # (3,N+1,6), ordered by MECHANICAL_VARIATIONS
    electrical: np.ndarray  # (P+1,3), ordered by ELECTRICAL_VARIATIONS
    gauss_points: int


def _geometry(radius_m, thickness_m):
    radius, thickness = float(radius_m), float(thickness_m)
    if not (math.isfinite(radius) and math.isfinite(thickness)
            and radius > 0 and thickness > 0 and radius > thickness/2):
        raise ValueError("require finite R>h/2>0 for H(z)>0 through thickness")
    return radius, thickness


def te_basis(z_m, thickness_m, order):
    """Dimensionless CUF TE basis and its first two z-derivatives (61--63)."""
    thickness = float(thickness_m)
    if not (math.isfinite(thickness) and thickness > 0):
        raise ValueError("thickness must be finite and positive")
    order = int(order)
    if not 0 <= order <= 4:
        raise ValueError("TE or electric basis order must be in 0..4")
    zeta = 2*float(z_m)/thickness
    if not math.isfinite(zeta):
        raise ValueError("z must be finite")
    c = 2.0/thickness
    val = np.array([zeta**i for i in range(order+1)], dtype=float)
    dz = np.array([0.0 if i == 0 else i*c*zeta**(i-1)
                   for i in range(order+1)], dtype=float)
    dzz = np.array([0.0 if i < 2 else i*(i-1)*c*c*zeta**(i-2)
                    for i in range(order+1)], dtype=float)
    return val, dz, dzz


@lru_cache(maxsize=32)
def _dimensionless_gauss_rule(n):
    """Immutable rules reused by all cells and nonlinear Newton evaluations."""
    xi,wi=np.polynomial.legendre.leggauss(n)
    xi.setflags(write=False)
    wi.setflags(write=False)
    return xi,wi


def gauss_thickness(thickness_m, n_gauss):
    h = float(thickness_m)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    if isinstance(n_gauss, bool) or int(n_gauss) != n_gauss or int(n_gauss) < 2:
        raise ValueError("n_gauss must be an integer >= 2")
    xi,wi=_dimensionless_gauss_rule(int(n_gauss))
    return .5*h*xi,.5*h*wi


def _kinematic_matrix_jets(jets, z_m, radius_m, thickness_m, *, variation=False):
    """F, F_,x, F_,s, F_,z in local material-frame components.

    Uses analytic derivatives of the nine quantities a,b,c,d,e,f,g,k,j,
    including H'(z)=1/R, F'_i and F''_i. In a *variation*, omit identity.
    """
    B, Bz, Bzz = te_basis(z_m, thickness_m, jets.te_order)
    u, ux, us, uxx, uxs, uss = (
        getattr(jets, key) @ B for key in _MECH_FIELDS
    )
    uz = jets.q @ Bz
    uxz = jets.qx @ Bz
    usz = jets.qs @ Bz
    uzz = jets.q @ Bzz
    H = 1.0 + z_m / radius_m
    invH = 1.0/H
    invRH = 1.0/(radius_m*H)

    def circle(v, vs):
        return np.array([vs[0], vs[1]+v[2]/radius_m,
                         vs[2]-v[1]/radius_m], dtype=float)

    n = circle(u, us)
    nx = circle(ux, uxs)
    ns = circle(us, uss)
    nz = circle(uz, usz)
    F = np.column_stack((ux, n*invH, uz))
    Fx = np.column_stack((uxx, nx*invH, uxz))
    Fs = np.column_stack((uxs, ns*invH, usz))
    Fz = np.column_stack((uxz, nz*invH - n*invH*invRH, uzz))
    if not variation:
        F = F + _EYE
    return F, Fx, Fs, Fz


def _strain_components(E):
    return np.array((E[0,0], E[1,1], E[2,2],
                     2*E[0,1], 2*E[0,2], 2*E[1,2]), dtype=float)


def _strain_from_deformation_jets(F, Fx, Fs, Fz):
    e = _strain_components(.5*(F.T@F - _EYE))
    deriv = []
    for Fd in (Fx, Fs, Fz):
        deriv.append(_strain_components(.5*(Fd.T@F + F.T@Fd)))
    return e, deriv


def _strain_variation_from_deformation_jets(F, dF, Fx, Fs, Fz, dFx, dFs, dFz):
    de = _strain_components(.5*(dF.T@F + F.T@dF))
    deriv = []
    for Fd, dFd in ((Fx,dFx),(Fs,dFs),(Fz,dFz)):
        deriv.append(_strain_components(.5*(dFd.T@F + Fd.T@dF +
                                              dF.T@Fd + F.T@dFd)))
    return de, deriv


def point_state(jets: CUFSurfaceJets, *, z_m, radius_m, thickness_m):
    R, h = _geometry(radius_m, thickness_m)
    z = float(z_m)
    if not math.isfinite(z) or abs(z) > .5*h + 1.e-12*h:
        raise ValueError("z lies outside shell thickness")
    H = 1.0 + z/R
    F, Fx, Fs, Fz = _kinematic_matrix_jets(jets, z, R, h)
    det = float(np.linalg.det(F))
    if not math.isfinite(det) or det <= 0:
        raise ValueError(f"INVALID_DEFORMATION_MAPPING: detF={det}, z={z}")
    strain, (sx,ss,sz) = _strain_from_deformation_jets(F,Fx,Fs,Fz)
    gradient = covariant_strain_gradients(strain,sx,ss,sz,radius_m=R,z_m=z)
    G, Gz, _ = te_basis(z,h,jets.potential_order)
    phi_x = float(G @ jets.psix)
    phi_s = float(G @ jets.psis)
    phi_z = float(Gz @ jets.psi)
    electric = reference_electric_field(phi_x,phi_s,phi_z,radius_m=R,z_m=z)
    return PointState(z,H,F,strain,gradient,electric,det)


def thickness_enthalpy(jets: CUFSurfaceJets, material: G3DMaterial, *,
                        radius_m, thickness_m, n_gauss=12):
    """H-weighted integrated electric enthalpy per mid-surface area (eq. 42)."""
    R,h = _geometry(radius_m,thickness_m)
    zs, ws = gauss_thickness(h,n_gauss)
    value = 0.0
    for z,w in zip(zs,ws):
        state = point_state(jets,z_m=z,radius_m=R,thickness_m=h)
        value += w*state.H*enthalpy_parts(
            state.strain6,state.gradient6x3,state.electric3,material)["total"]
    return float(value)


def _unit_variation(jets, comp, index, variation):
    """Virtual jet representing one independent surface q-derivative."""
    arrays={field:np.zeros_like(getattr(jets,field)) for field in _MECH_FIELDS}
    arrays[variation][comp,index]=1.0
    return jets.replaced(**arrays)


@lru_cache(maxsize=192)
def _cuf_virtual_kinematic_library(order,z,radius_m,thickness_m):
    """Exact, immutable CUF unit-jet deformation gradients (no state dependence).

    Shape (3, N+1, 6, 4, 3, 3), with four entries F,Fx,Fs,Fz.
    """
    mechanical=np.zeros((3,order+1),float)
    electric=np.zeros(1,float)
    zero=CUFSurfaceJets(*(mechanical.copy() for _ in _MECH_FIELDS),
                        *(electric.copy() for _ in _ELEC_FIELDS))
    out=np.empty((3,order+1,len(MECHANICAL_VARIATIONS),4,3,3),float)
    for comp in range(3):
        for coefficient in range(order+1):
            for k,name in enumerate(MECHANICAL_VARIATIONS):
                variation=_unit_variation(zero,comp,coefficient,name)
                out[comp,coefficient,k]=np.stack(
                    _kinematic_matrix_jets(
                        variation,z,radius_m,thickness_m,variation=True))
    out.setflags(write=False)
    return out


def thickness_weak_coefficients(jets: CUFSurfaceJets, material: G3DMaterial, *,
                                 radius_m, thickness_m, n_gauss=12):
    """Thickness integrated, exact Gateaux weak coefficients (114--115,128--131).

    Important: this is only the LOCAL weak first variation. Correct assembly of
    surface weak residuals, boundary traces, Newton, time marching, MVI and MKR
    is a later stage and is intentionally NOT dispatched here.
    """
    R,h = _geometry(radius_m,thickness_m)
    zs,ws = gauss_thickness(h,n_gauss)
    coeff = np.zeros((3,jets.te_order+1,6),float)
    electric_coeff = np.zeros((jets.potential_order+1,3),float)
    energy=0.0
    # Unit variations are independent of the current state, so their F-jets
    # can be calculated once per quadrature point using the same analytic map.
    for z,w in zip(zs,ws):
        state=point_state(jets,z_m=z,radius_m=R,thickness_m=h)
        H=state.H
        F,Fx,Fs,Fz=_kinematic_matrix_jets(jets,z,R,h)
        S,M,D=enthalpy_derivatives(
            state.strain6,state.gradient6x3,state.electric3,material
        )
        energy += w*H*enthalpy_parts(
            state.strain6,state.gradient6x3,state.electric3,material)["total"]
        virtual_library=_cuf_virtual_kinematic_library(
            jets.te_order,float(z),R,h)
        for a in range(3):
            for i in range(jets.te_order+1):
                for k in range(len(MECHANICAL_VARIATIONS)):
                    dF,dFx,dFs,dFz=virtual_library[a,i,k]
                    de,(dex,des,dez) = _strain_variation_from_deformation_jets(
                        F,dF,Fx,Fs,Fz,dFx,dFs,dFz
                    )
                    dg=covariant_strain_gradients(
                        de,dex,des,dez,radius_m=R,z_m=z
                    )
                    coeff[a,i,k] += w*H*(np.dot(S,de)+np.sum(M*dg))
        G,Gz,_=te_basis(z,h,jets.potential_order)
        # derivative of enthalpy wrt electric field is -D; E=-grad(phi),
        # hence electrical first variation is +D dot grad(delta phi).
        electric_coeff[:,0] += w*H*Gz*D[2]
        electric_coeff[:,1] += w*H*G*D[0]
        electric_coeff[:,2] += w*G*D[1]
    return WeakCoefficients(float(energy),coeff,electric_coeff,len(zs))


def mass_matrix_te(*, radius_m, thickness_m, rho_kg_m3, te_order):
    """Exact dimensionless-basis mass moments (76--82), including odd terms."""
    R,h=_geometry(radius_m,thickness_m)
    rho=float(rho_kg_m3)
    if not math.isfinite(rho) or rho<=0:
        raise ValueError("rho must be finite and positive")
    n=int(te_order)
    if not 1<=n<=4:
        raise ValueError("CUF order must be TE1..TE4")
    M=np.empty((n+1,n+1),float)
    for i in range(n+1):
        for j in range(n+1):
            p=i+j
            M[i,j]=rho*(h/(p+1) if p%2==0 else h*h/(2*R*(p+2)))
    return M


def gauge_moments(*, radius_m, thickness_m, potential_order):
    """Coefficients I_m of the ONE global electrical gauge (135--136)."""
    R,h=_geometry(radius_m,thickness_m)
    P=int(potential_order)
    if not 0<=P<=4:
        raise ValueError("potential order must be 0..4")
    return np.array([h/(m+1) if m%2==0 else h*h/(2*R*(m+2))
                     for m in range(P+1)],float)


__all__=("CUFSurfaceJets","PointState","WeakCoefficients","te_basis",
         "gauss_thickness","point_state","thickness_enthalpy",
         "thickness_weak_coefficients","mass_matrix_te","gauge_moments",
         "MECHANICAL_VARIATIONS","ELECTRICAL_VARIATIONS")
