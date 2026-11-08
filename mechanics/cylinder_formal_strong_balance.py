"""Formal coupled strong interior equations for all cylinder shell theories.

For each independent mechanical surface jet q_(x^a s^b), let A_ab be its
thickness-integrated first-variation coefficient. The exact continuum
Euler expression is sum (-1)^(a+b) d_x^a d_s^b A_ab. Electric potential
jets have the corresponding zeroth and first derivative terms.

The caller supplies a derivative of smooth coefficient fields. This is
continuum algebra for §§22--29 of Krysko--Zhong, not a choice of strong
finite-difference stencil, a boundary closure, or a numerical solver.
In particular, applying a collocated centered first difference twice on an
even periodic grid would introduce a spurious checkerboard mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FormalCoupledBalance:
    mechanical: np.ndarray
    electrical: np.ndarray | None


def _derivative(action, field, axial, periodic):
    if axial==periodic==0:
        return field
    result=np.asarray(action(field,axial,periodic),float)
    if result.shape!=field.shape or not np.isfinite(result).all():
        raise ValueError("strong derivative must retain finite coefficient shape")
    return result


def formal_coupled_balance(mechanical_coefficients, derivative_orders,
                           derivative, electrical_coefficients=None,
                           *, mechanical_load=None, electrical_charge=None):
    """Apply the exact Euler derivative signs to unweighted weak jet fields.

    Mechanical coefficients are (njet,nfield,nx,ns) and the derivative
    orders are (name, axial_order, periodic_order), each jet occurring once.
    Electrical coefficients, when supplied, are (3,npotential,nx,ns) in
    [psi,psi_x,psi_s] order. Distributed loads and charges are densities in
    the same midsurface measure. Mechanical force is subtracted; physical
    free charge is added, consistently with +integral(rho*phi) in the
    stationary electric enthalpy and div(D)=rho.
    """
    if not callable(derivative):
        raise TypeError("a smooth-field derivative callback is required")
    mc=np.asarray(mechanical_coefficients,float)
    if (mc.ndim!=4 or min(mc.shape)<1 or len(derivative_orders)!=len(mc)
            or not np.isfinite(mc).all()):
        raise ValueError("finite mechanical coefficients (njet,nfield,nx,ns) required")
    seen=set()
    mechanical=np.zeros_like(mc[0])
    for k,item in enumerate(derivative_orders):
        if not isinstance(item,(tuple,list)) or len(item)!=3:
            raise ValueError("each mechanical jet requires (name,x_order,s_order)")
        _,ax,az=item
        if (type(ax) is not int or type(az) is not int or ax<0 or az<0
                or ax+az>3 or (ax,az) in seen):
            raise ValueError("duplicate or unsupported mechanical surface jet")
        seen.add((ax,az))
        mechanical+=(-1)**(ax+az)*_derivative(derivative,mc[k],ax,az)
    if (0,0) not in seen:
        raise ValueError("base mechanical displacement jet is required")
    if mechanical_load is not None:
        load=np.asarray(mechanical_load,float)
        if load.shape!=mechanical.shape or not np.isfinite(load).all():
            raise ValueError("finite mechanical load density has wrong shape")
        mechanical-=load

    electric=None
    if electrical_coefficients is not None:
        ec=np.asarray(electrical_coefficients,float)
        if (ec.ndim!=4 or ec.shape[0]!=3 or ec.shape[1]<1
                or ec.shape[2:]!=mc.shape[2:] or not np.isfinite(ec).all()):
            raise ValueError("electric coefficients require (3,npotential,nx,ns)")
        electric=(ec[0]-_derivative(derivative,ec[1],1,0)
                  -_derivative(derivative,ec[2],0,1))
        if electrical_charge is not None:
            charge=np.asarray(electrical_charge,float)
            if charge.shape!=electric.shape or not np.isfinite(charge).all():
                raise ValueError("finite charge density has wrong shape")
            electric+=charge
    elif electrical_charge is not None:
        raise ValueError("charge requires active electrical coefficients")
    return FormalCoupledBalance(mechanical,electric)


__all__=("FormalCoupledBalance","formal_coupled_balance")
