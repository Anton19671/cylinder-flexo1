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


def _state(value, expected, label):
    a=np.asarray(value,dtype=float)
    if a.shape!=expected or not np.isfinite(a).all():
        raise ValueError(f"{label}: finite shape {expected} required")
    return a


def _normal_data(q, v, a, Dx, Ds, R):
    """Return n, n_t, n_tt, and A/B at all surface quadrature nodes."""
    qx=np.einsum("ij,cjk->cik",Dx,q,optimize=True)
    qs=np.einsum("ij,ckj->cki",Ds,q,optimize=True)
    vx=np.einsum("ij,cjk->cik",Dx,v,optimize=True)
    vs=np.einsum("ij,ckj->cki",Ds,v,optimize=True)
    ax=np.einsum("ij,cjk->cik",Dx,a,optimize=True)
    ass=np.einsum("ij,ckj->cki",Ds,a,optimize=True)

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


def kl_inertial_weak_action(mechanical, velocity, acceleration, *,
                            Dx, Ds, wx, ws, radius_m, thickness_m,
                            density_kg_m3):
    """Exact nonlinear inertia residual with surface derivatives assembled by IBP.

    Inputs and return: (3,nx,ns); density in kg/m^3. Inertia is
    configuration- and velocity-dependent:
      I(q,v,a)=M(q)a+G(q)[v,v].
    The same physical normal n is used by the KL Green-Lagrange enthalpy.
    """
    Dx=np.asarray(Dx,float)
    Ds=np.asarray(Ds,float)
    wx=np.asarray(wx,float)
    ws=np.asarray(ws,float)
    if Dx.ndim!=2 or Ds.ndim!=2 or Dx.shape[0]!=Dx.shape[1] or Ds.shape[0]!=Ds.shape[1]:
        raise ValueError("KL inertia derivative matrices must be square")
    nx,ns=Dx.shape[0],Ds.shape[0]
    if wx.shape!=(nx,) or ws.shape!=(ns,) or not (
        np.isfinite(Dx).all() and np.isfinite(Ds).all() and
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
    n,nt,ntt,A,B,mag=_normal_data(q,v,a,Dx,Ds,R)
    a_vec=np.moveaxis(a,0,-1)
    # Exact curved thickness moments of rho H(z).
    m0=rho*h
    m1=rho*h**3/(12.*R)
    m2=rho*h**3/12.
    g0=m0*a_vec+m1*ntt
    gn=m1*a_vec+m2*ntt
    # d n=P_n dN/|N|; the adjoint converts a normal virtual
    # variation into independent variations of q, q_x, q_s.
    projected=(gn-n*np.sum(n*gn,axis=-1)[...,None])/mag[...,None]
    gx=np.cross(B,projected)
    gs=np.cross(projected,A)
    gq=g0.copy()
    gq[...,1]-=gs[...,2]/R
    gq[...,2]+=gs[...,1]/R
    weights=wx[:,None]*ws[None,:]
    # Discrete surface adjoints match the exact variation of the
    # trapezoidal weak integral and include natural edge work.
    gq_c=np.moveaxis(gq*weights[...,None],-1,0)
    gx_c=np.moveaxis(gx*weights[...,None],-1,0)
    gs_c=np.moveaxis(gs*weights[...,None],-1,0)
    out=(gq_c+np.einsum("ki,ckj->cij",Dx,gx_c,optimize=True)
         +np.einsum("kj,cik->cij",Ds,gs_c,optimize=True))
    if not np.isfinite(out).all():
        raise ValueError("INVALID_KL_INERTIAL_WEAK_RESIDUAL")
    return out


def kl_kinetic_energy(mechanical, velocity, *, Dx, Ds, wx, ws,
                      radius_m,thickness_m,density_kg_m3):
    """Exact KL finite-rotation kinetic energy from the SAME director map."""
    q=np.asarray(mechanical,float);v=np.asarray(velocity,float)
    zeros=np.zeros_like(v)
    # Reuse the validated map and return an integral scalar in joules.
    Dx=np.asarray(Dx,float);Ds=np.asarray(Ds,float)
    wx=np.asarray(wx,float);ws=np.asarray(ws,float)
    R,h,rho=map(float,(radius_m,thickness_m,density_kg_m3))
    if not (np.isfinite([R,h,rho]).all() and R>h/2>0 and rho>0):
        raise ValueError("invalid KL kinetic geometry/density")
    shape=(3,len(wx),len(ws))
    q=_state(q,shape,"KL displacement")
    v=_state(v,shape,"KL velocity")
    z=_state(zeros,shape,"KL zero acceleration")
    _,nt,_,_,_,_=_normal_data(q,v,z,Dx,Ds,R)
    vec=np.moveaxis(v,0,-1)
    m0,m1,m2=rho*h,rho*h**3/(12.*R),rho*h**3/12.
    local=.5*(m0*np.sum(vec*vec,axis=-1)+
              2*m1*np.sum(vec*nt,axis=-1)+m2*np.sum(nt*nt,axis=-1))
    total=float(np.sum(local*wx[:,None]*ws[None,:]))
    if not math.isfinite(total):
        raise ValueError("INVALID_KL_KINETIC_ENERGY")
    return total


__all__=("kl_inertial_weak_action","kl_kinetic_energy")
