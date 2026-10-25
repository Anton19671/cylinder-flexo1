#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent FSDT and curvature-corrected Reddy-type cylindrical directors.

Five surface fields: (u0,v0,w0,theta_x,theta_s), with s=R*theta.
The SAME Green-Lagrange / covariant G3D enthalpy as the verified CUF
material is evaluated on these distinct director kinematics, but NOT by
constraining the CUF polynomial coefficients. Reddy's circumferential
thickness function is curvature-corrected; in the R->infinity limit it
reduces to -4*z**3/(3*h**2). Its small-strain transverse shears vanish
EXACTLY at z=+-h/2 at finite radius.

Mechanical independent weak variations: q, qx, qs, qxx, qxs, qss,
qxxx, qxxs, qxss, qsss (ONE coefficient for each mixed derivative).
In the G3D Reddy energy third spatial derivatives are necessary because
the displacement ansatz itself depends on first surface derivatives.

This is a LOCAL surface-jet and thickness-integrated Gateaux operator,
NOT a complete global FDM, MVI, BC or transient solver. In particular,
plane-stress elimination and natural higher-order edge traces are not
implied by this kinematic restriction of the 3D StVK material law.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from cuf_g3d_weak import te_basis, gauss_thickness
from flexo_gradient_g3d import (
    G3DMaterial, covariant_strain_gradients, reference_electric_field,
    enthalpy_parts, enthalpy_derivatives,
)

FIELDS = ("q", "qx", "qs", "qxx", "qxs", "qss", "qxxx", "qxxs", "qxss", "qsss")
ELECTRICAL_FIELDS = ("psi", "psix", "psis")
_MULTI = ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2),
          (3,0),(2,1),(1,2),(0,3))
_INDEX = dict(zip(_MULTI,FIELDS))
_EYE = np.eye(3)
THEORIES = ("FSDT", "REDDY")


@dataclass(frozen=True)
class DirectorSurfaceJets:
    """Surface derivatives of five fields plus independent potential jets.

    q...qsss are finite 5-vectors, potential jets are finite (P+1)-vectors.
    Zero third derivatives must be supplied for a quadratic trial field.
    """
    q: np.ndarray
    qx: np.ndarray
    qs: np.ndarray
    qxx: np.ndarray
    qxs: np.ndarray
    qss: np.ndarray
    qxxx: np.ndarray
    qxxs: np.ndarray
    qxss: np.ndarray
    qsss: np.ndarray
    psi: np.ndarray
    psix: np.ndarray
    psis: np.ndarray

    def __post_init__(self):
        for name in FIELDS:
            v = np.asarray(getattr(self,name),float)
            if v.shape != (5,) or not np.isfinite(v).all():
                raise ValueError(f"{name} must be a finite five-component vector")
            object.__setattr__(self,name,v.copy())
        P = np.asarray(self.psi,float)
        if P.ndim != 1 or not 1 <= P.size <= 5:
            raise ValueError("psi must have shape (P+1,), P=0..4")
        for name in ELECTRICAL_FIELDS:
            v = np.asarray(getattr(self,name),float)
            if v.shape != P.shape or not np.isfinite(v).all():
                raise ValueError(f"{name} must be a finite vector of shape {P.shape}")
            object.__setattr__(self,name,v.copy())

    @property
    def potential_order(self):
        return len(self.psi)-1

    def replaced(self, **updates):
        allowed = (*FIELDS,*ELECTRICAL_FIELDS)
        if set(updates)-set(allowed):
            raise ValueError("Unexpected surface-jet field")
        return DirectorSurfaceJets(**{n:updates.get(n,getattr(self,n)) for n in allowed})

    def derivative(self, dx, ds):
        return getattr(self,_INDEX[(dx,ds)])


@dataclass(frozen=True)
class DirectorPoint:
    z_m: float
    H: float
    deformation_gradient: np.ndarray
    strain6: np.ndarray
    gradient6x3: np.ndarray
    electric3: np.ndarray
    determinant: float


@dataclass(frozen=True)
class DirectorWeak:
    energy_j_per_m2: float
    mechanical: np.ndarray  # (5,10) in FIELDS order
    electrical: np.ndarray  # (P+1,3) in ELECTRICAL_FIELDS order
    gauss_points: int


def _check_geometry(radius_m, thickness_m, z_m=None):
    R,h = float(radius_m),float(thickness_m)
    if not (math.isfinite(R) and math.isfinite(h) and h > 0 and R > h/2):
        raise ValueError("Require R > h/2 > 0 for nonsingular cylindrical metric")
    if z_m is not None:
        z=float(z_m)
        if not math.isfinite(z) or abs(z)>h/2+1.e-13*h:
            raise ValueError("Quadrature point outside shell thickness")
    return R,h


def reddy_thickness_functions(z_m, radius_m, thickness_m):
    """Return axial and EXACT curvature-corrected tangential warping jets.

    ax=-4*z^3/(3*h^2) ; (H*as' - as/R)=-4*z^2/h^2; as(0)=0.
    Consequently gamma_sz=Q_s*(1-(2*z/h)^2)/H for all R>h/2.
    The expression with log1p is evaluated by a Taylor series near z=0 to
    avoid cancellation of terms of orders zero, one and two.
    """
    R,h=_check_geometry(radius_m,thickness_m,z_m)
    z=float(z_m);t=z/R;H=1+t
    axial=np.array((-4*z**3/(3*h*h),-4*z*z/(h*h),-8*z/(h*h)),float)
    if abs(t)<.08:
        summation=0.
        for n in range(3,25):
            summation += (-1. if n%2==0 else 1.)*(n-2)/n*t**n
        integ=R*summation
    else:
        integ=R*(t+t/(1+t)-2*math.log1p(t))
    a=-4*R*(R+z)*integ/(h*h)
    da=(a/R-4*z*z/(h*h))/H
    dda=-8*z/(h*h*H)
    return axial, np.array((a,da,dda),float)


@lru_cache(maxsize=512)
def _warp(theory,z,R,h):
    if theory not in THEORIES:
        raise ValueError("theory must be FSDT or REDDY")
    if theory=="FSDT":
        return np.zeros(3),np.zeros(3)
    return reddy_thickness_functions(z,R,h)


def _displacement(jets,theory,z,R,h,dx=0,ds=0,dz=0):
    """Cartesian physical components in rotating cylindrical reference basis."""
    ax,bs=_warp(theory,z,R,h)
    q=jets.derivative(dx,ds)
    wx=jets.derivative(dx+1,ds)[2] if dx+ds <= 2 else None
    ws=jets.derivative(dx,ds+1)[2] if dx+ds <= 2 else None
    if dz==0:
        return np.array((
            q[0]+z*q[3]+ax[0]*(q[3]+wx),
            q[1]+z*q[4]+bs[0]*(q[4]+ws-q[1]/R),
            q[2],
        ),float)
    if dz==1:
        return np.array((
            q[3]+ax[1]*(q[3]+wx),
            q[4]+bs[1]*(q[4]+ws-q[1]/R),
            0.,
        ),float)
    if dz==2:
        return np.array((
            ax[2]*(q[3]+wx),
            bs[2]*(q[4]+ws-q[1]/R),
            0.,
        ),float)
    raise ValueError("Only z derivatives up to order two are required")


def _F_jets(jets,theory,z,R,h,*,virtual=False):
    H=1+z/R
    U=_displacement(jets,theory,z,R,h)
    Ux=_displacement(jets,theory,z,R,h,dx=1)
    Us=_displacement(jets,theory,z,R,h,ds=1)
    Uxx=_displacement(jets,theory,z,R,h,dx=2)
    Uxs=_displacement(jets,theory,z,R,h,dx=1,ds=1)
    Uss=_displacement(jets,theory,z,R,h,ds=2)
    Uz=_displacement(jets,theory,z,R,h,dz=1)
    Uxz=_displacement(jets,theory,z,R,h,dx=1,dz=1)
    Usz=_displacement(jets,theory,z,R,h,ds=1,dz=1)
    Uzz=_displacement(jets,theory,z,R,h,dz=2)

    def circ(a,b):
        return np.array((b[0],b[1]+a[2]/R,b[2]-a[1]/R),float)
    c=circ(U,Us)
    F=np.column_stack((Ux,c/H,Uz))
    Fx=np.column_stack((Uxx,circ(Ux,Uxs)/H,Uxz))
    Fs=np.column_stack((Uxs,circ(Us,Uss)/H,Usz))
    Fz=np.column_stack((Uxz,circ(Uz,Usz)/H-c/(R*H*H),Uzz))
    if not virtual:F=F+_EYE
    return F,Fx,Fs,Fz


def _engineering(tensor):
    return np.array([tensor[0,0],tensor[1,1],tensor[2,2],
                     2*tensor[0,1],2*tensor[0,2],2*tensor[1,2]],float)


def _strain(F,Fx,Fs,Fz):
    strain=_engineering(.5*(F.T@F-_EYE))
    der=[_engineering(.5*(Fi.T@F+F.T@Fi)) for Fi in (Fx,Fs,Fz)]
    return strain,der


def _strain_virtual(F,Fx,Fs,Fz,dF,dFx,dFs,dFz):
    de=_engineering(.5*(dF.T@F+F.T@dF))
    der=[_engineering(.5*(df.T@F+fi.T@dF+dF.T@fi+F.T@df))
         for fi,df in zip((Fx,Fs,Fz),(dFx,dFs,dFz))]
    return de,der


def _zero_like(jets):
    return {n:np.zeros_like(getattr(jets,n)) for n in (*FIELDS,*ELECTRICAL_FIELDS)}


def _unit(jets,field,component):
    updates=_zero_like(jets)
    updates[field][component]=1.
    return jets.replaced(**updates)


def director_point(jets: DirectorSurfaceJets, *, theory, z_m, radius_m, thickness_m):
    R,h=_check_geometry(radius_m,thickness_m,z_m)
    z=float(z_m)
    F,Fx,Fs,Fz=_F_jets(jets,theory,z,R,h)
    det=float(np.linalg.det(F))
    if not math.isfinite(det) or det<=0:
        raise ValueError(f"INVALID_DEFORMATION_MAPPING: detF={det}")
    e,der=_strain(F,Fx,Fs,Fz)
    g=covariant_strain_gradients(e,*der,radius_m=R,z_m=z)
    G,Gz,_=te_basis(z,h,jets.potential_order)
    electric=reference_electric_field(float(G@jets.psix),float(G@jets.psis),
                                       float(Gz@jets.psi),radius_m=R,z_m=z)
    return DirectorPoint(z,1+z/R,F,e,g,electric,det)


def thickness_enthalpy(jets: DirectorSurfaceJets, material: G3DMaterial, *,
                        theory, radius_m, thickness_m, n_gauss=12):
    R,h=_check_geometry(radius_m,thickness_m)
    total=0.
    zs,ws=gauss_thickness(h,n_gauss)
    for z,w in zip(zs,ws):
        p=director_point(jets,theory=theory,z_m=z,radius_m=R,thickness_m=h)
        total+=w*p.H*enthalpy_parts(p.strain6,p.gradient6x3,p.electric3,material)["total"]
    return float(total)


@lru_cache(maxsize=192)
def _director_virtual_f_jets(theory,z,R,h):
    zeros=np.zeros(5)
    electric=np.zeros(1)
    state=DirectorSurfaceJets(*(zeros.copy() for _ in FIELDS),
                              *(electric.copy() for _ in ELECTRICAL_FIELDS))
    result=np.empty((len(FIELDS),5,4,3,3))
    for k,name in enumerate(FIELDS):
        for i in range(5):
            result[k,i]=np.stack(_F_jets(
                _unit(state,name,i),theory,z,R,h,virtual=True))
    result.setflags(write=False)
    return result


@lru_cache(maxsize=192)
def _director_virtual_velocity(theory,z,R,h):
    zeros=np.zeros(5)
    electric=np.zeros(1)
    state=DirectorSurfaceJets(*(zeros.copy() for _ in FIELDS),
                              *(electric.copy() for _ in ELECTRICAL_FIELDS))
    result=np.empty((3,5,3))
    for k,name in enumerate(FIELDS[:3]):
        for i in range(5):
            result[k,i]=_displacement(_unit(state,name,i),theory,z,R,h)
    result.setflags(write=False)
    return result


def thickness_weak_coefficients(jets: DirectorSurfaceJets, material: G3DMaterial, *,
                                theory, radius_m, thickness_m, n_gauss=12):
    """Exact local Gateaux coefficients of the thickness-integrated enthalpy.

    This routine does NOT integrate over x,s or enforce edge boundary traces.
    """
    R,h=_check_geometry(radius_m,thickness_m)
    mech=np.zeros((5,len(FIELDS)))
    elec=np.zeros((jets.potential_order+1,len(ELECTRICAL_FIELDS)))
    total=0.
    zs,ws=gauss_thickness(h,n_gauss)
    for z,w in zip(zs,ws):
        library=_director_virtual_f_jets(theory,float(z),R,h)
        p=director_point(jets,theory=theory,z_m=z,radius_m=R,thickness_m=h)
        F,Fx,Fs,Fz=_F_jets(jets,theory,z,R,h)
        sigma,hyper,D=enthalpy_derivatives(p.strain6,p.gradient6x3,p.electric3,material)
        total+=w*p.H*enthalpy_parts(p.strain6,p.gradient6x3,p.electric3,material)["total"]
        for k in range(len(FIELDS)):
            for i in range(5):
                dF,dFx,dFs,dFz=library[k,i]
                de,der=_strain_virtual(F,Fx,Fs,Fz,dF,dFx,dFs,dFz)
                dg=covariant_strain_gradients(de,*der,radius_m=R,z_m=z)
                mech[i,k]+=w*p.H*(sigma@de + np.sum(hyper*dg))
        G,Gz,_=te_basis(z,h,jets.potential_order)
        elec[:,0]+=w*p.H*Gz*D[2]
        elec[:,1]+=w*p.H*G*D[0]
        elec[:,2]+=w*G*D[1]
    return DirectorWeak(float(total),mech,elec,len(zs))


def transverse_shear_linear(jets: DirectorSurfaceJets, *, theory, z_m,
                              radius_m, thickness_m):
    """Exact linearized engineering gamma_xz,gamma_sz at z (not 2PK stress)."""
    R,h=_check_geometry(radius_m,thickness_m,z_m)
    F,*_=_F_jets(jets,theory,float(z_m),R,h)
    return np.array((F[0,2]+F[2,0],F[1,2]+F[2,1]),float)


def kinetic_energy(velocity_jets: DirectorSurfaceJets, *, theory, radius_m,
                   thickness_m, density_kg_m3, n_gauss=12):
    R,h=_check_geometry(radius_m,thickness_m)
    rho=float(density_kg_m3)
    if not math.isfinite(rho) or rho<=0:raise ValueError("rho must be positive and finite")
    zs,ws=gauss_thickness(h,n_gauss)
    out=0.
    for z,w in zip(zs,ws):
        vel=_displacement(velocity_jets,theory,z,R,h)
        out+=.5*rho*w*(1+z/R)*float(vel@vel)
    return float(out)


def kinetic_weak_coefficients(velocity_jets: DirectorSurfaceJets, *, theory,
                              radius_m, thickness_m, density_kg_m3, n_gauss=12):
    """Local inertia coefficients wrt velocities qdot,qdot_x,qdot_s.

    Reddy type kinetic energy explicitly depends on gradients of qdot, so a
    simple 5x5 mass matrix would be incomplete. The operator is consistent
    with a gradient-dependent director ansatz and needs surface assembly.
    """
    R,h=_check_geometry(radius_m,thickness_m)
    rho=float(density_kg_m3)
    if not math.isfinite(rho) or rho<=0:raise ValueError("rho must be positive and finite")
    zs,ws=gauss_thickness(h,n_gauss)
    coeff=np.zeros((5,3))
    for z,w in zip(zs,ws):
        library=_director_virtual_velocity(theory,float(z),R,h)
        vel=_displacement(velocity_jets,theory,z,R,h)
        for k in range(3):
            for i in range(5):
                dvel=library[k,i]
                coeff[i,k]+=rho*w*(1+z/R)*float(vel@dvel)
    return coeff


__all__=("THEORIES","FIELDS","ELECTRICAL_FIELDS","DirectorSurfaceJets",
         "DirectorPoint","DirectorWeak","reddy_thickness_functions",
         "director_point","thickness_enthalpy","thickness_weak_coefficients",
         "transverse_shear_linear","kinetic_energy","kinetic_weak_coefficients")
