#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8A: displacement-only classical Kirchhoff-Love mechanics
for a closed circular cylindrical shell.

Unknowns:
    U = [u(x,s), v(x,s), w(x,s)]

Coordinates:
    x - axial
    s - circumferential arc length
    z - thickness coordinate

Three distinct geometry branches are provided:
- LINEAR: linear membrane strains and linear Donnell bending terms;
- VON_KARMAN: Karman membrane strains and linear Donnell bending terms;
- GREEN_LAGRANGE: exact midsurface metric, exact unit normal and exact
  change of the second fundamental form from cylinder_kl_full_gl_kinematics.

The GREEN_LAGRANGE branch is therefore not the former hybrid
"exact membrane + linear curvature" approximation.

Engineering shear through thickness:
    gamma_xs(z) = gamma_xs^0 + 2 z kappa_xs

Plane-stress isotropic law:
    sigma_x = E/(1-nu^2) (eps_x + nu eps_s)
    sigma_s = E/(1-nu^2) (eps_s + nu eps_x)
    tau_xs  = G gamma_xs

This module deliberately uses the physical twisting resultant
    M_xs = int z tau_xs dz
which agrees with the already verified Kirchhoff plate convention:
    M_xs = -D(1-nu) w_xs.

No Airy stress function F.
No Galerkin.
No FEM.
No MVI yet.
No Birger yet.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import cylinder_kl_full_gl_kinematics as fullgl


class KLMechanicsError(RuntimeError):
    pass


@dataclass(frozen=True)
class KLElasticConstants:
    E: float
    nu: float
    h: float

    def __post_init__(self):
        E=float(self.E); nu=float(self.nu); h=float(self.h)
        if not np.isfinite(E) or E <= 0:
            raise ValueError("E must be finite and positive")
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
            raise ValueError("nu must satisfy -1 < nu < 0.5")

    @property
    def G(self):
        return self.E/(2.0*(1.0+self.nu))

    @property
    def A(self):
        return self.E*self.h/(1.0-self.nu**2)

    @property
    def D(self):
        return self.E*self.h**3/(12.0*(1.0-self.nu**2))


@dataclass(frozen=True)
class KLMidSurfaceState:
    eps_x: np.ndarray
    eps_s: np.ndarray
    gamma_xs: np.ndarray
    kappa_x: np.ndarray
    kappa_s: np.ndarray
    kappa_xs: np.ndarray


@dataclass(frozen=True)
class KLResultants:
    Nx: np.ndarray
    Ns: np.ndarray
    Nxs: np.ndarray
    Mx: np.ndarray
    Ms: np.ndarray
    Mxs: np.ndarray


@dataclass(frozen=True)
class KLFieldKinematics:
    U: np.ndarray
    ux: np.ndarray
    us: np.ndarray
    vx: np.ndarray
    vs: np.ndarray
    wx: np.ndarray
    ws: np.ndarray
    wxx: np.ndarray
    wss: np.ndarray
    wxs: np.ndarray
    state: KLMidSurfaceState
    full_gl: object = None


def _same_shape(*arrs):
    aa=[np.asarray(a,float) for a in arrs]
    shape=aa[0].shape
    if any(a.shape!=shape for a in aa):
        raise ValueError("array shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in aa):
        raise ValueError("non-finite array")
    return aa


def mid_surface_strains(
    ux,us,vx,vs,wx,ws,w,*,R,geometry="VON_KARMAN",v=None
):
    ux,us,vx,vs,wx,ws,w=_same_shape(ux,us,vx,vs,wx,ws,w)
    R=float(R)
    if not np.isfinite(R) or R<=0:
        raise ValueError("R must be finite and positive")
    mode=str(geometry).upper()
    if mode not in ("LINEAR","VON_KARMAN","GREEN_LAGRANGE"):
        raise ValueError("geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")

    eps_x=ux.copy()
    eps_s=vs+w/R
    gamma=us+vx
    if mode=="VON_KARMAN":
        eps_x=eps_x+0.5*wx*wx
        eps_s=eps_s+0.5*ws*ws
        gamma=gamma+wx*ws
    elif mode=="GREEN_LAGRANGE":
        if v is None:
            raise ValueError("v is required for GREEN_LAGRANGE cylinder kinematics")
        v=np.asarray(v,float)
        if v.shape!=w.shape or not np.all(np.isfinite(v)):
            raise ValueError("v must be finite and have the displacement field shape")
        # Exact midsurface Green-Lagrange membrane strains for a circular
        # cylinder in the orthonormal (x,s,r) basis. KL curvature changes
        # retain the displacement-only thin-shell definition below.
        circumferential=1.0+vs+w/R
        radial_slope=ws-v/R
        eps_x=ux+0.5*(ux*ux+vx*vx+wx*wx)
        eps_s=vs+w/R+0.5*(us*us+(circumferential-1.0)**2+radial_slope**2)
        gamma=(1.0+ux)*us+vx*circumferential+wx*radial_slope
    return eps_x,eps_s,gamma


def curvature_changes(wxx,wss,wxs):
    wxx,wss,wxs=_same_shape(wxx,wss,wxs)
    return -wxx,-wss,-wxs


def make_mid_surface_state(
    ux,us,vx,vs,wx,ws,w,wxx,wss,wxs,*,R,geometry,v=None
):
    ex,es,gam=mid_surface_strains(
        ux,us,vx,vs,wx,ws,w,R=R,geometry=geometry,v=v
    )
    kx,ks,kxs=curvature_changes(wxx,wss,wxs)
    return KLMidSurfaceState(ex,es,gam,kx,ks,kxs)


def through_thickness_strains(state,z):
    z=float(z)
    ex=state.eps_x+z*state.kappa_x
    es=state.eps_s+z*state.kappa_s
    gam=state.gamma_xs+2.0*z*state.kappa_xs
    return ex,es,gam


def plane_stress(ex,es,gamma,*,E,nu):
    ex,es,gamma=_same_shape(ex,es,gamma)
    E=float(E); nu=float(nu)
    if E<=0 or not (-1.0<nu<0.5):
        raise ValueError("invalid E/nu")
    c=E/(1.0-nu**2)
    sx=c*(ex+nu*es)
    ss=c*(es+nu*ex)
    tau=E/(2.0*(1.0+nu))*gamma
    return sx,ss,tau


def analytic_elastic_resultants(state,constants):
    if not isinstance(constants,KLElasticConstants):
        raise ValueError("constants must be KLElasticConstants")
    A=constants.A
    D=constants.D
    nu=constants.nu

    Nx=A*(state.eps_x+nu*state.eps_s)
    Ns=A*(state.eps_s+nu*state.eps_x)
    Nxs=constants.G*constants.h*state.gamma_xs

    Mx=D*(state.kappa_x+nu*state.kappa_s)
    Ms=D*(state.kappa_s+nu*state.kappa_x)
    Mxs=D*(1.0-nu)*state.kappa_xs
    return KLResultants(Nx,Ns,Nxs,Mx,Ms,Mxs)


def quadrature_elastic_resultants(state,constants,quadrature):
    if not isinstance(constants,KLElasticConstants):
        raise ValueError("constants must be KLElasticConstants")
    shape=np.asarray(state.eps_x).shape
    acc=[np.zeros(shape,float) for _ in range(6)]
    Nx,Ns,Nxs,Mx,Ms,Mxs=acc
    for z,wz in zip(np.asarray(quadrature.z,float),np.asarray(quadrature.w,float)):
        ex,es,gam=through_thickness_strains(state,z)
        sx,ss,tau=plane_stress(ex,es,gam,E=constants.E,nu=constants.nu)
        Nx += wz*sx
        Ns += wz*ss
        Nxs += wz*tau
        Mx += wz*z*sx
        Ms += wz*z*ss
        Mxs += wz*z*tau
    return KLResultants(Nx,Ns,Nxs,Mx,Ms,Mxs)


def field_kinematics(U,ops,*,geometry):
    U=np.asarray(U,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if U.shape!=expected:
        raise ValueError(f"U must have shape {expected}")
    if not np.all(np.isfinite(U)):
        raise ValueError("U contains non-finite values")
    u,v,w=U

    ux=ops.Dx(u); us=ops.Ds(u)
    vx=ops.Dx(v); vs=ops.Ds(v)
    wx=ops.Dx(w); ws=ops.Ds(w)

    wxx=ops.Dxx(w)
    wss=ops.Dss(w)
    wxs=ops.Ds(wx)

    mode=str(geometry).upper()
    exact=None
    if mode=="GREEN_LAGRANGE":
        Ux=np.stack((ux,vx,wx),axis=0)
        Us=np.stack((us,vs,ws),axis=0)
        Uxx=np.stack([ops.Dxx(q) for q in U],axis=0)
        Uss=np.stack([ops.Dss(q) for q in U],axis=0)
        Uxs=np.stack([ops.Ds(ops.Dx(q)) for q in U],axis=0)
        exact=fullgl.state_from_derivatives(
            U,Ux,Us,Uxx,Uss,Uxs,R=ops.grid.radius
        )
        state=KLMidSurfaceState(
            exact.eps_x,exact.eps_s,exact.gamma_xs,
            exact.kappa_x,exact.kappa_s,exact.kappa_xs,
        )
    else:
        state=make_mid_surface_state(
            ux,us,vx,vs,wx,ws,w,wxx,wss,wxs,
            R=ops.grid.radius,geometry=mode,v=v
        )
    return KLFieldKinematics(
        U,ux,us,vx,vs,wx,ws,wxx,wss,wxs,state,exact
    )


def zero_surface_load(ops):
    return np.zeros((3,ops.grid.ns,ops.grid.nx),float)


def strong_residual_from_resultants(
    kin,resultants,ops,*,geometry,generalized_load
):
    p=np.asarray(generalized_load,float)
    expected=(3,ops.grid.ns,ops.grid.nx)
    if p.shape!=expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    mode=str(geometry).upper()
    if mode=="LINEAR":
        Ru=-ops.Dx(resultants.Nx)-ops.Ds(resultants.Nxs)-p[0]
        Rv=-ops.Dx(resultants.Nxs)-ops.Ds(resultants.Ns)-p[1]
        membrane_w=np.zeros_like(resultants.Ns)
        zeroth_w=resultants.Ns/ops.grid.radius
    elif mode=="VON_KARMAN":
        Ru=-ops.Dx(resultants.Nx)-ops.Ds(resultants.Nxs)-p[0]
        Rv=-ops.Dx(resultants.Nxs)-ops.Ds(resultants.Ns)-p[1]
        ax=resultants.Nx*kin.wx+resultants.Nxs*kin.ws
        ass=resultants.Ns*kin.ws+resultants.Nxs*kin.wx
        membrane_w=-ops.Dx(ax)-ops.Ds(ass)
        zeroth_w=resultants.Ns/ops.grid.radius
    elif mode=="GREEN_LAGRANGE":
        R=ops.grid.radius
        a=1.0+kin.vs+kin.U[2]/R
        b=kin.ws-kin.U[1]/R
        fx_u=resultants.Nx*(1.0+kin.ux)+resultants.Nxs*kin.us
        fs_u=resultants.Ns*kin.us+resultants.Nxs*(1.0+kin.ux)
        fx_v=resultants.Nx*kin.vx+resultants.Nxs*a
        fs_v=resultants.Ns*a+resultants.Nxs*kin.vx
        fx_w=resultants.Nx*kin.wx+resultants.Nxs*b
        fs_w=resultants.Ns*b+resultants.Nxs*kin.wx
        Ru=-ops.Dx(fx_u)-ops.Ds(fs_u)-p[0]
        Rv=(-ops.Dx(fx_v)-ops.Ds(fs_v)
            -(resultants.Ns*b+resultants.Nxs*kin.wx)/R-p[1])
        membrane_w=-ops.Dx(fx_w)-ops.Ds(fs_w)
        zeroth_w=(resultants.Ns*a+resultants.Nxs*kin.vx)/R
    else:
        raise ValueError("geometry must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")

    bend=(
        -ops.Dxx(resultants.Mx)
        -2.0*ops.Ds(ops.Dx(resultants.Mxs))
        -ops.Dss(resultants.Ms)
    )
    Rw=membrane_w+zeroth_w+bend-p[2]
    return np.stack([Ru,Rv,Rw],axis=0)


def elastic_strong_residual(
    U,ops,*,E,nu,h,geometry,generalized_load=None,
    resultants_method="ANALYTIC",quadrature=None
):
    kin=field_kinematics(U,ops,geometry=geometry)
    constants=KLElasticConstants(E,nu,h)
    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        res=analytic_elastic_resultants(kin.state,constants)
    elif method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        res=quadrature_elastic_resultants(kin.state,constants,quadrature)
    else:
        raise ValueError("resultants_method must be ANALYTIC or QUADRATURE")
    p=zero_surface_load(ops) if generalized_load is None else generalized_load
    R=strong_residual_from_resultants(
        kin,res,ops,geometry=geometry,generalized_load=p
    )
    return R,res,kin
