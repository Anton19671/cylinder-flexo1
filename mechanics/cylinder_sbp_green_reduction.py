#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact, weighted SBP Green reduction of G3D/flexoelectric weak jets.

For a periodic cylinder surface and independent mechanical jets q_(x^a s^b),
the DISCRETE weak variation is

  sum_{a,b} <A_ab, Dx^a Ds^b delta q>_W.

When Dx^T Wx + Wx Dx = Bx and Ds^T Ws + Ws Ds = 0,
it equals, to floating-point roundoff,

  < sum (-1)^(a+b) Dx^a Ds^b A_ab, delta q>_W
  + sum_{ends} < t,delta q>_Ws
             + <m_n, D_n delta q>_Ws
             + <h_nn,D_nn delta q>_Ws,

where (t,m_n,h_nn) are the SAME axial natural end fluxes derived
from the constitutive weak coefficients. Electric phi_x contributes
the electrical natural end charge flux.

This module does not impose strong natural BCs, approximate 3D corner
conditions or change material equations. It gives an independently
auditable exact decomposition of the existing discrete variational
nodal residual (for the selected SBP grid only). A user-supplied
non-SBP pair is rejected, rather than silently using continuum fluxes.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cylinder_sbp_derivatives import require_green_identity
from cylinder_g3d_natural_edges import axial_natural_end_tractions


@dataclass(frozen=True)
class GreenFieldSplit:
    """All arrays have nodal force units; not pointwise stress units."""
    bulk: np.ndarray
    boundary: np.ndarray
    reconstructed: np.ndarray
    direct_weak: np.ndarray
    absolute_defect: float
    relative_defect: float


@dataclass(frozen=True)
class CoupledGreenSplit:
    mechanical: GreenFieldSplit
    electrical: GreenFieldSplit | None
    natural_ends: object
    green_identity: dict
    status: str = "EXACT_DISCRETE_SBP_GREEN_REDUCTION_NOT_MESH_VERIFIED"


def _finite_coeff(coefficients, nx, ns, label):
    a=np.asarray(coefficients,float)
    if (a.ndim!=4 or a.shape[1]<1 or a.shape[2:]!=(nx,ns)
        or not np.isfinite(a).all()):
        raise ValueError(f"{label}: finite coefficients (njet,nfield,nx,ns) required")
    return a


def _make_field_split(coefficient, powers, W, ws, ends, electric=False):
    """Evaluate direct adjoint, strong bulk and physical end virtual work."""
    njet,nfield,nx,ns=coefficient.shape
    dx,Dxx= powers["Dx"],powers["Dx2"]
    xp,sp=powers["xp"],powers["sp"]
    direct=np.zeros((nfield,nx,ns),float)
    strong=np.zeros_like(direct)
    for idx,(ax,az) in enumerate(powers["orders"]):
        tensor=coefficient[idx]
        d_x=xp[ax];d_s=sp[az]
        direct+=np.einsum(
            "ki,ckj,lj->cil",d_x,W[None,...]*tensor,d_s,
            optimize=True)
        strong+=(-1.)**(ax+az)*np.einsum(
            "ik,ckj,lj->cil",d_x,tensor,d_s,
            optimize=True)
    bulk=W[None,...]*strong
    boundary=np.zeros_like(bulk)
    for end in (ends.left,ends.right):
        i=0 if end.side=="left" else nx-1
        if electric:
            charge=end.electric_charge_flux
            if charge is None:
                raise ValueError("electric flux missing in electrical Green reduction")
            boundary[:,i,:]+=ws[None,:]*charge
        else:
            boundary[:,i,:]+=ws[None,:]*end.traction
            # Virtual variations d_n q=n_x D_x q and
            # d_nn q=D_x^2 q for either end.
            boundary+=(
                (end.n_x*dx[i,:])[None,:,None]
                *end.normal_double_traction[:,None,:]*ws[None,None,:])
            boundary+=(
                Dxx[i,:][None,:,None]
                *end.normal_triple_traction[:,None,:]*ws[None,None,:])
    reconstructed=bulk+boundary
    absolute=float(np.max(np.abs(direct-reconstructed),initial=0.))
    size=max(1.,float(np.max(np.abs(direct),initial=0.)),
             float(np.max(np.abs(bulk),initial=0.)),
             float(np.max(np.abs(boundary),initial=0.)))
    relative=absolute/size
    if any(not np.isfinite(z).all() for z in (
        direct,bulk,boundary,reconstructed)):
        raise ValueError("NONFINITE_SBP_GREEN_SPLIT")
    return GreenFieldSplit(bulk,boundary,reconstructed,direct,
                           absolute,relative)


def split_coupled_sbp_residual(operators, mechanical_coefficients,
                               mechanical_derivatives,
                               electrical_coefficients=None,
                               *, relative_tolerance=2.e-10):
    """Split exact 2D SBP weak residual of all model-specific weak jets.

    The electric order is [phi, phi_x, phi_s]. This is the same
    three-jet electric potential approximation used by the coupled
    G3D/flexoelectric functional for every TE potential order.
    """
    dx=np.asarray(operators.Dx,float)
    ds=np.asarray(operators.Ds,float)
    wx=np.asarray(operators.wx,float)
    ws=np.asarray(operators.ws,float)
    report=require_green_identity(dx,ds,wx,ws)
    nx,ns=len(wx),len(ws)
    mc=_finite_coeff(mechanical_coefficients,nx,ns,"mechanical G3D")
    orders=[]
    for item in mechanical_derivatives:
        if len(item)!=3:
            raise ValueError("mechanical derivative tuples must have length three")
        _,ax,az=item
        if type(ax) is not int or type(az) is not int or ax<0 or az<0 or ax+az>3:
            raise ValueError("unsupported G3D derivative order")
        orders.append((ax,az))
    if len(orders)!=len(mc):
        raise ValueError("G3D weak coefficients and derivatives disagree")
    ec=None
    if electrical_coefficients is not None:
        ec=_finite_coeff(electrical_coefficients,nx,ns,"electrical G3D")
        if ec.shape[0]!=3:
            raise ValueError("electrical jets require [phi,phi_x,phi_s]")
    ends=axial_natural_end_tractions(
        mc,mechanical_derivatives,dx,ds,electrical_coefficients=ec)
    xp=[np.linalg.matrix_power(dx,k) for k in range(4)]
    sp=[np.linalg.matrix_power(ds,k) for k in range(4)]
    W=np.outer(wx,ws)
    common={"xp":xp,"sp":sp,"Dx":dx,"Dx2":xp[2],"orders":orders}
    m=_make_field_split(mc,common,W,ws,ends)
    electric=None
    if ec is not None:
        electric=_make_field_split(
            ec,{**common,"orders":[(0,0),(1,0),(0,1)]},
            W,ws,ends,electric=True)
    if (isinstance(relative_tolerance,(bool,np.bool_)) or
        not np.isfinite(float(relative_tolerance)) or relative_tolerance<=0):
        raise ValueError("positive Green reduction tolerance required")
    tol=float(relative_tolerance)
    maximum=max(m.relative_defect,
                0. if electric is None else electric.relative_defect)
    if maximum>tol:
        raise ValueError("SBP_GREEN_DECOMPOSITION_DRIFT: "
                         f"relative={maximum:.4e}, tolerance={tol:.4e}")
    return CoupledGreenSplit(m,electric,ends,report)


__all__=("GreenFieldSplit","CoupledGreenSplit","split_coupled_sbp_residual")
