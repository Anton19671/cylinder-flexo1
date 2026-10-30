#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact finite-rotation Kirchhoff-Love shell inertia in a cylindrical frame.

Reference midsurface material coordinates: x, s=R*theta. The deformed
director n=(A cross B)/|A cross B|, where
 A=e_x+q_x,
 B=(q_s0, 1+q_s1+q2/R, q_s2-q1/R).
The three-component physical placement is
 u(x,s,z,t)=q(x,s,t)+z*(n(x,s,t)-e_r).

The material acceleration therefore is
 u_tt=q_tt+z*(Dn(q)[q_tt]+D2n(q)[q_t,q_t]).
The second directional derivative is NOT negligible in finite rotations.
The exact acceleration virtual work is
 integral_A0 integral_-h/2^h/2 rho*(1+z/R) u_tt . delta u dz dA0,
 delta u=delta q+z Dn(q)[delta q].
This code evaluates that weak form analytically; the curved thickness
moments rho*h, rho*h^3/(12 R), rho*h^3/12 are exact.

No electrical inertia; no small-rotation/constant-mass approximation.
"""
from __future__ import annotations

import math
import numpy as np
from cylinder_surface_linear_algebra import (
    is_sparse_matrix, apply_axial, apply_periodic,
)
from cylinder_kinematic_models import normalized_model


def _state(value, expected, label):
    a=np.asarray(value,dtype=float)
    if a.shape!=expected or not np.isfinite(a).all():
        raise ValueError(f"{label}: finite shape {expected} required")
    return a


def _normal_data(q, v, a, Dx, Ds, R):
    """Return n, n_t, n_tt, and A/B at all surface quadrature nodes."""
    qx=apply_axial(q,Dx)
    qs=apply_periodic(q,Ds)
    vx=apply_axial(v,Dx)
    vs=apply_periodic(v,Ds)
    ax=apply_axial(a,Dx)
    ass=apply_periodic(a,Ds)

    def axial(t,with_identity):
        return np.stack((t[0]+(1. if with_identity else 0.),t[1],t[2]),axis=-1)

    def circum(t,der,with_identity):
        return np.stack((der[0],der[1]+t[2]/R+
                         (1. if with_identity else 0.),
                         der[2]-t[1]/R),axis=-1)

    A=axial(qx,True)
    B=circum(q,qs,True)
    At=axial(vx,False)
    Bt=circum(v,vs,False)
    Att=axial(ax,False)
    Btt=circum(a,ass,False)
    N=np.cross(A,B)
    mag=np.linalg.norm(N,axis=-1)
    if not np.isfinite(mag).all() or np.any(mag<=1.e-10):
        raise ValueError("INVALID_KL_MIDSURFACE_NORMAL: degenerate finite-rotation metric")
    n=N/mag[...,None]
    Nt=np.cross(At,B)+np.cross(A,Bt)
    nproj=np.sum(n*Nt,axis=-1)
    nt=(Nt-n*nproj[...,None])/mag[...,None]
    Ntt=(np.cross(Att,B)+2.*np.cross(At,Bt)+np.cross(A,Btt))
    ntt=((Ntt-n*np.sum(n*Ntt,axis=-1)[...,None])/mag[...,None]
         -n*np.sum(nt*nt,axis=-1)[...,None]
         -2.*(nproj/mag)[...,None]*nt)
    if not np.isfinite(ntt).all():
        raise ValueError("INVALID_KL_DIRECTOR_ACCELERATION")
    return n,nt,ntt,A,B,mag


def kl_inertial_jet_coefficients(mechanical, velocity, acceleration, *,
                                  Dx, Ds, wx, ws, radius_m, thickness_m,
                                  density_kg_m3,geometry_model="GREEN_LAGRANGE"):
    """Return KL inertial virtual-work coefficients before surface IBP.

    Order: (q,q_x,q_s), each coefficient (3,nx,ns). They are the
    *unweighted* local density per unit reference midsurface area.
    Thus they can be used both in the exact discrete adjoint and in
    the continuum-form natural-end diagnostic. Includes the
    D2n(q)[qdot,qdot] (gyroscopic/geometric) acceleration term.
    """
    Dx=Dx if is_sparse_matrix(Dx) else np.asarray(Dx,float)
    Ds=Ds if is_sparse_matrix(Ds) else np.asarray(Ds,float)
    wx=np.asarray(wx,float)
    ws=np.asarray(ws,float)
    if Dx.ndim!=2 or Ds.ndim!=2 or Dx.shape[0]!=Dx.shape[1] or Ds.shape[0]!=Ds.shape[1]:
        raise ValueError("KL inertia derivative matrices must be square")
    nx,ns=Dx.shape[0],Ds.shape[0]
    if wx.shape!=(nx,) or ws.shape!=(ns,) or not (
        np.isfinite(Dx.data if is_sparse_matrix(Dx) else Dx).all() and
        np.isfinite(Ds.data if is_sparse_matrix(Ds) else Ds).all() and
        np.isfinite(wx).all() and np.isfinite(ws).all()):
        raise ValueError("KL inertia has invalid finite quadrature/derivative matrices")
    if np.any(wx<=0) or np.any(ws<=0):
        raise ValueError("KL inertia requires positive surface quadrature")
    R,h,rho=map(float,(radius_m,thickness_m,density_kg_m3))
    if not all(map(math.isfinite,(R,h,rho))) or not (R>h/2>0 and rho>0):
        raise ValueError("KL inertia requires R>h/2>0 and rho>0")
    shape=(3,nx,ns)
    q=_state(mechanical,shape,"KL displacement")
    v=_state(velocity,shape,"KL velocity")
    a=_state(acceleration,shape,"KL acceleration")
    model=normalized_model(geometry_model)
    a_vec=np.moveaxis(a,0,-1)
    m0=rho*h
    m1=rho*h**3/(12.*R)
    m2=rho*h**3/12.
    if model!="GREEN_LAGRANGE":
        # The LINEAR / von Karman KL director is the small-rotation map
        # n=(-w_x,v/R-w_s,1); its acceleration has no quadratic velocity term.
        ax=apply_axial(a,Dx)
        ass=apply_periodic(a,Ds)
        ntt=np.stack((-ax[2],a[1]/R-ass[2],np.zeros_like(a[0])),axis=-1)
        g0=m0*a_vec+m1*ntt
        gn=m1*a_vec+m2*ntt
        gq=g0.copy()
        gq[...,1]+=gn[...,1]/R
        gx=np.zeros_like(g0)
        gs=np.zeros_like(g0)
        gx[...,2]=-gn[...,0]
        gs[...,2]=-gn[...,1]
        out=np.stack((np.moveaxis(gq,-1,0),
                      np.moveaxis(gx,-1,0),
                      np.moveaxis(gs,-1,0)),axis=0)
        if not np.isfinite(out).all():
            raise ValueError("INVALID_KL_SMALL_ROTATION_INERTIA")
        return out
    n,nt,ntt,A,B,mag=_normal_data(q,v,a,Dx,Ds,R)
    g0=m0*a_vec+m1*ntt
    gn=m1*a_vec+m2*ntt
    projected=(gn-n*np.sum(n*gn,axis=-1)[...,None])/mag[...,None]
    gx=np.cross(B,projected)
    gs=np.cross(projected,A)
    gq=g0.copy()
    gq[...,1]-=gs[...,2]/R
    gq[...,2]+=gs[...,1]/R
    out=np.stack((np.moveaxis(gq,-1,0),
                  np.moveaxis(gx,-1,0),
                  np.moveaxis(gs,-1,0)),axis=0)
    if not np.isfinite(out).all():
        raise ValueError("INVALID_KL_INERTIAL_WEAK_COEFFICIENTS")
    return out


def kl_inertial_weak_action(mechanical, velocity, acceleration, *,
                            Dx, Ds, wx, ws, radius_m, thickness_m,
                            density_kg_m3,geometry_model="GREEN_LAGRANGE"):
    """Assemble KL inertial virtual work by the selected director map.

    This and natural end diagnostics share one exact local inertial
    coefficient map: no separately approximated KL inertia is introduced.
    """
    coefficients=kl_inertial_jet_coefficients(
        mechanical,velocity,acceleration,Dx=Dx,Ds=Ds,wx=wx,ws=ws,
        radius_m=radius_m,thickness_m=thickness_m,
        density_kg_m3=density_kg_m3,geometry_model=geometry_model)
    W=np.asarray(wx,float)[:,None]*np.asarray(ws,float)[None,:]
    Dx=Dx if is_sparse_matrix(Dx) else np.asarray(Dx,float)
    Ds=Ds if is_sparse_matrix(Ds) else np.asarray(Ds,float)
    out=(coefficients[0]*W[None,...]
         +apply_axial(coefficients[1]*W[None,...],Dx.T)
         +apply_periodic(coefficients[2]*W[None,...],Ds.T))
    if not np.isfinite(out).all():
        raise ValueError("INVALID_KL_INERTIAL_WEAK_RESIDUAL")
    return out


def kl_kinetic_energy(mechanical, velocity, *, Dx, Ds, wx, ws,
                      radius_m,thickness_m,density_kg_m3,
                      geometry_model="GREEN_LAGRANGE"):
    """KL kinetic energy using the same selected director kinematics."""
    q=np.asarray(mechanical,float);v=np.asarray(velocity,float)
    zeros=np.zeros_like(v)
    # Reuse the validated map and return an integral scalar in joules.
    Dx=Dx if is_sparse_matrix(Dx) else np.asarray(Dx,float)
    Ds=Ds if is_sparse_matrix(Ds) else np.asarray(Ds,float)
    wx=np.asarray(wx,float);ws=np.asarray(ws,float)
    R,h,rho=map(float,(radius_m,thickness_m,density_kg_m3))
    if not (np.isfinite([R,h,rho]).all() and R>h/2>0 and rho>0):
        raise ValueError("invalid KL kinetic geometry/density")
    shape=(3,len(wx),len(ws))
    q=_state(q,shape,"KL displacement")
    v=_state(v,shape,"KL velocity")
    z=_state(zeros,shape,"KL zero acceleration")
    model=normalized_model(geometry_model)
    if model=="GREEN_LAGRANGE":
        _,nt,_,_,_,_=_normal_data(q,v,z,Dx,Ds,R)
    else:
        vx=apply_axial(v,Dx);vs=apply_periodic(v,Ds)
        nt=np.stack((-vx[2],v[1]/R-vs[2],np.zeros_like(v[0])),axis=-1)
    vec=np.moveaxis(v,0,-1)
    m0,m1,m2=rho*h,rho*h**3/(12.*R),rho*h**3/12.
    local=.5*(m0*np.sum(vec*vec,axis=-1)+
              2*m1*np.sum(vec*nt,axis=-1)+m2*np.sum(nt*nt,axis=-1))
    total=float(np.sum(local*wx[:,None]*ws[None,:]))
    if not math.isfinite(total):
        raise ValueError("INVALID_KL_KINETIC_ENERGY")
    return total


__all__=("kl_inertial_weak_action","kl_inertial_jet_coefficients",
         "kl_kinetic_energy")
