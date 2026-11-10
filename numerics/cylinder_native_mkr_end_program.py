"""Physical strong-MKR end program for stationary parallel cylindrical sections.

This is the OWN strong (not MVI weak/SBP) closure. The current scope fixes
mechanical q values at x=0 and x=L. The other axial rows are natural
generalized normal actions derived from local G3D/Gz physical weak jets.
Their number is supplied by explicit fieldwise differential half-orders;
the number of stored kinematic jets is NOT used to infer PDE order.

Optional generalized end loads have the same physical normal conventions
as the separately implemented MVI input, without sharing its numerics.
The source continuum microfree edge/face model is not certified.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import cuf_mean_potential_gauge


@dataclass(frozen=True)
class MKREndProgram:
    axial_orders: tuple[int,...]
    mechanical_ends: dict
    electrical_ends: dict | None
    electric_gauge: object | None
    status: str="EXPLICIT_MKR_STRONG_END_ROWS_UNCERTIFIED_MICROFREE"


def _profile(value,n,ns,name):
    if value is None or isinstance(value,(bool,np.bool_)):
        raise ValueError("MKR_END_"+name+"_FINITE_PROFILE_REQUIRED")
    try:
        v=np.asarray(value,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("MKR_END_"+name+"_NONNUMERIC_PROFILE") from exc
    if v.ndim==0:
        v=np.full((n,ns),float(v))
    if v.shape!=(n,ns) or not np.isfinite(v).all():
        raise ValueError("MKR_END_"+name+"_SHAPE_INVALID")
    return v


def compile_native_mkr_end_program(
        model:DirectStrongShell,*,axial_orders,mechanical,
        electric,potential_order,mechanical_end_work=None,
        potential_scale_v=1.):
    if not isinstance(model,DirectStrongShell):
        raise TypeError("native DirectStrongShell required")
    n=model.n_mechanical
    if (not isinstance(axial_orders,(tuple,list)) or len(axial_orders)!=n or
            any(type(m) is not int or not 1<=m<=3 for m in axial_orders)):
        raise ValueError("MKR_END_EXPLICIT_FIELD_HALF_ORDERS_REQUIRED")
    orders=tuple(axial_orders)
    if (model.theory.startswith("CUF_TE") and max(orders)>2):
        raise ValueError("MKR_END_CUF_MAXIMUM_HALF_ORDER_TWO")
    if max(orders)>max(ax for _,ax,_ in model.derivative_orders):
        raise ValueError("MKR_END_ORDER_EXCEEDS_CONSTITUTIVE_AXIAL_JETS")
    if 2*max(orders)>model.grid.nx:
        raise ValueError("MKR_END_GRID_TOO_SMALL_FOR_PHYSICAL_END_ROWS")
    if mechanical!="IMMOBILE_PARALLEL_ENDS":
        raise ValueError("MKR_END_ONLY_IMMOBILE_PARALLEL_MECHANICAL")
    bounds={side:[[{"kind":"ESSENTIAL","value":0.}]+
                   [{"kind":"NATURAL","value":0.} for k in range(m-1)]
                   for m in orders] for side in ("left","right")}
    if mechanical_end_work is not None:
        if (not isinstance(mechanical_end_work,dict) or
                set(mechanical_end_work)!={"left","right"}):
            raise ValueError("MKR_END_WORK_REQUIRES_LEFT_RIGHT")
        names={"traction","double_traction","normal_double_traction",
               "triple_traction","normal_triple_traction"}
        for side,normal in (("left",-1),("right",1)):
            payload=mechanical_end_work[side]
            if payload is None:
                continue
            if not isinstance(payload,dict) or set(payload)-names:
                raise ValueError("MKR_END_WORK_UNKNOWN_ACTION")
            if ({"double_traction","normal_double_traction"}<=set(payload)
                    or {"triple_traction","normal_triple_traction"}<=set(payload)):
                raise ValueError("MKR_END_WORK_AMBIGUOUS_ORIENTATION")
            for kind,value in payload.items():
                data=_profile(value,n,model.grid.ns,kind.upper())
                if kind=="traction":
                    if np.any(data):
                        raise ValueError("MKR_END_FIXED_Q_TRACTION_IS_REACTION")
                    continue
                order=1 if "double" in kind else 2
                if any(m<=order for m in orders):
                    for c in range(n):
                        if orders[c]<=order and np.any(data[c]):
                            raise ValueError("MKR_END_WORK_FIELD_TRACE_UNAVAILABLE")
                factor=(normal if kind=="double_traction" else 1.)
                for c in range(n):
                    if orders[c]>order:
                        bounds[side][c][order]={
                            "kind":"NATURAL","value":factor*data[c].copy()}
    if not model.electrical_on:
        if electric!="OFF" or potential_order is not None:
            raise ValueError("MKR_END_ELECTRICITY_OFF_REQUIRES_OFF")
        e=None
        gauge=None
    else:
        if type(potential_order) is not int or not 0<=potential_order<=4:
            raise ValueError("MKR_END_ELECTRIC_ORDER_REQUIRED")
        if electric not in ("GROUNDED_LEFT","GROUNDED_BOTH",
                            "INSULATED_FLOATING"):
            raise ValueError("MKR_END_ELECTRIC_MODE_UNSUPPORTED")
        e={}
        for side in ("left","right"):
            ground=(electric=="GROUNDED_BOTH" or
                    electric=="GROUNDED_LEFT" and side=="left")
            e[side]=[{"kind":"ESSENTIAL" if ground else "NATURAL",
                      "value":0.} for k in range(potential_order+1)]
        gauge=(cuf_mean_potential_gauge(
            model.area_weights,radius_m=model.grid.radius_m,
            thickness_m=model.thickness_m,potential_order=potential_order,
            potential_scale_v=potential_scale_v)
               if electric=="INSULATED_FLOATING" else None)
    return MKREndProgram(orders,bounds,e,gauge)


__all__=("MKREndProgram","compile_native_mkr_end_program")
