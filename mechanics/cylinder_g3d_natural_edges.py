#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuum-form natural axial-end tractions derived from the G3D weak jets.

For a thickness-integrated *surface* energy density
  L(q, q_x, q_s, q_xx, q_xs, q_ss, q_xxx, q_xxs, q_xss, q_sss, phi...),
write A_ab = partial L / partial q_(x^a s^b). All entries are the
ACTUAL constitutive weak coefficients produced by KL, FSDT, Reddy or
CUF TE1..TE4, not simplified classical membrane resultants.

After integrating by parts in axial x and periodic s, the axial
boundary variation is

  [ int_s (B0 . delta q + B1 . delta q_x
                         + B2 . delta q_xx
                         + P_x . delta phi) ds ]_{x=0}^{x=L},

where
  B0 = A_10 - dx A_20 - ds A_11
                 + dxx A_30 + dxds A_21 + dss A_12
  B1 = A_20 - dx A_30 - ds A_21
  B2 = A_30.

These formulas are summed ONLY over derivatives actually present in
the model. For outward normal n_x = -1 on left, +1 on right:

   physical traction          = n_x * B0
   physical double traction   =       B1  (conjugate to d_n q)
   physical triple traction   = n_x * B2  (conjugate to d_nn q)
   electrical natural flux    = n_x * P_x

No factor of two is introduced for a mixed q_xs jet. Electric flux
is the enthalpy derivative with respect to phi_x and therefore has
the correct D=-dH/dElectric sign without a separate constitutive law.

IMPORTANT NUMERICAL LIMIT: using non-SBP finite-difference Dx and
periodic one-sided Ds to differentiate A_ab approximates CONTINUUM
boundary fluxes. It is not an exact discrete integration-by-parts
identity for the existing reference grid and MUST NOT be used to
claim certified microfree conditions, enforce new BC, or accept steps.
The full discrete weak residual with natural variations remains the
equilibrium equation, and is checked separately by MVI/FDM.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from cylinder_surface_linear_algebra import (
    is_sparse_matrix, apply_biaxial, operator_power,
)


@dataclass(frozen=True)
class NaturalEnd:
    """Fields sampled on a physical axial end, with lengthwise density units."""
    side: str
    n_x: int
    traction: np.ndarray
    normal_double_traction: np.ndarray
    normal_triple_traction: np.ndarray
    electric_charge_flux: np.ndarray | None


@dataclass(frozen=True)
class NaturalEnds:
    left: NaturalEnd
    right: NaturalEnd
    status: str = "CONTINUUM_FORM_SAMPLED_ON_NON_SBP_GRID_NOT_STRONG_BC_CERTIFIED"

    def npz_arrays(self):
        result={}
        for end in (self.left,self.right):
            p=end.side+"_g3d_natural_"
            result[p+"traction"]=end.traction
            result[p+"normal_double_traction"]=end.normal_double_traction
            result[p+"normal_triple_traction"]=end.normal_triple_traction
            if end.electric_charge_flux is not None:
                result[p+"electric_charge_flux"]=end.electric_charge_flux
        return result

    def magnitudes(self):
        """Unnormalized physical flux magnitudes; not convergence criteria."""
        summary={}
        for end in (self.left,self.right):
            q={
                "traction":end.traction,
                "normal_double_traction":end.normal_double_traction,
                "normal_triple_traction":end.normal_triple_traction,
            }
            if end.electric_charge_flux is not None:
                q["electric_charge_flux"]=end.electric_charge_flux
            summary[end.side]={key:float(np.max(np.abs(v),initial=0))
                               for key,v in q.items()}
        return summary


def axial_natural_end_tractions(mechanical_coefficients, derivative_orders,
                                Dx, Ds, electrical_coefficients=None):
    """Return continuum-form end fluxes for any 2D energy up to third order.

    mechanical_coefficients: (nderiv,ncomponent,nx,ns), unweighted
       thickness-integrated coefficients per midsurface area.
    derivative_orders: sequence of (name, a, b) for q_(x^a s^b).
    Dx,Ds: physical-coordinate spatial differentiation matrices.
    electrical_coefficients: optional (3,npotential,nx,ns), order
       [phi, phi_x, phi_s] (no electric strain gradient postulated).
    """
    a=np.asarray(mechanical_coefficients,dtype=float)
    dx=Dx if is_sparse_matrix(Dx) else np.asarray(Dx,dtype=float)
    ds=Ds if is_sparse_matrix(Ds) else np.asarray(Ds,dtype=float)
    if (a.ndim!=4 or not a.size or not np.isfinite(a).all() or
        a.shape[1]<1 or a.shape[2]<2 or a.shape[3]<2):
        raise ValueError("invalid finite G3D mechanical jet coefficient tensor")
    nder,ncomp,nx,ns=a.shape
    if (dx.shape!=(nx,nx) or ds.shape!=(ns,ns) or
        not np.isfinite(dx.data if is_sparse_matrix(dx) else dx).all() or
        not np.isfinite(ds.data if is_sparse_matrix(ds) else ds).all()):
        raise ValueError("G3D end derivatives require matching finite Dx and Ds")
    if len(derivative_orders)!=nder:
        raise ValueError("natural end jet order count does not match coefficients")
    seen=set()
    orders=[]
    for derivative in derivative_orders:
        if not isinstance(derivative,(list,tuple)) or len(derivative)!=3:
            raise ValueError("natural end orders require (name, axial, tangential)")
        _,x_order,s_order=derivative
        if (type(x_order) is not int or type(s_order) is not int or
            x_order<0 or s_order<0 or x_order+s_order>3):
            raise ValueError("invalid G3D axial/tangential derivative order")
        if (x_order,s_order) in seen:
            raise ValueError("duplicate G3D mixed derivative: use one q_xs jet")
        seen.add((x_order,s_order))
        orders.append((x_order,s_order))
    if (0,0) not in seen:
        raise ValueError("missing base displacement jet")
    e=None
    if electrical_coefficients is not None:
        e=np.asarray(electrical_coefficients,dtype=float)
        if e.ndim!=4 or e.shape[0]!=3 or e.shape[2:]!=(nx,ns) or (
            e.shape[1]<1 or not np.isfinite(e).all()):
            raise ValueError("G3D electric flux needs (3,npotential,nx,ns)")
    xpow=[operator_power(dx,k) for k in range(3)]
    spow=[operator_power(ds,k) for k in range(3)]
    flux=[np.zeros((ncomp,nx,ns),float) for _ in range(3)]
    # Exact algebraic signs from axial and periodic integration by parts.
    # q_xxs enters two different natural fluxes and MUST NOT be doubled.
    for idx,(ax,az) in enumerate(orders):
        for k in range(min(3,ax)):
            da=ax-1-k
            sign=-1. if (da+az)%2 else 1.
            derivative=apply_biaxial(a[idx],xpow[da],spow[az])
            flux[k]+=sign*derivative
    if any(not np.isfinite(f).all() for f in flux):
        raise ValueError("nonfinite G3D natural end flux")
    result=[]
    for side,node,n_x in (("left",0,-1),("right",-1,1)):
        electric=None if e is None else n_x*e[1,:,node,:].copy()
        result.append(NaturalEnd(side,n_x,n_x*flux[0][:,node,:].copy(),
            flux[1][:,node,:].copy(),n_x*flux[2][:,node,:].copy(),electric))
    return NaturalEnds(*result)


__all__=("NaturalEnd","NaturalEnds","axial_natural_end_tractions")
