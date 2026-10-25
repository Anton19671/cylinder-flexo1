#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variational DEAD_REFERENCE radial loads for the coupled shell functional.

Reference midsurface area: dA0=dx ds, s=R*theta.
Physical cylindrical face at z_f: dAf=(1+z_f/R)dA0.

Virtual external work (positive traction points to +e_r):
  delta W = integral_A0 [H_f p(x,s) delta u_r(x,s,z_f)] dA0.
CUF TE-N: delta u_r=sum_{k=0}^N (2*z_f/h)^k delta w_k.
FSDT/Reddy: delta u_r=delta w_0 (radial component has no warping).
KL: only z_f=0 is accepted. At an offset face the exact finite-rotation
normal produces additional q_x/q_s variations; dropping them is NOT valid.

Only fixed reference-direction forces are conservative in the above form.
Follower pressure, current-area loads and true delta-like singular point
forces require different consistent weak forms and are NOT silently mapped.

Concentrated line/point inputs are explicitly spread over a finite PHYSICAL
band with C1 support, and normalized using the identical quadrature weights
as the weak solver. Regularization widths are properties of the problem,
not changing hidden functions of grid size.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

RADIAL_LOAD_TYPES = (
    "UNIFORM_RADIAL_FACE", "HARMONIC_RADIAL_FACE", "PATCH_RADIAL_FACE",
    "LINE_RADIAL_GENERATOR", "POINT_RADIAL_FORCE",
)
FACES = ("MIDDLE", "INNER", "OUTER")

_COMMON = {"type","kinematics","surface","face","load_factors",
           "direction","x1","x2","c1","c2","full_circumference",
           "circumferential_coordinate","edge_width_x","edge_width_c"}
_SPECIFIC = {
    "UNIFORM_RADIAL_FACE": {"q"},
    "HARMONIC_RADIAL_FACE": {"q","m","n","phase_rad"},
    "PATCH_RADIAL_FACE": {"q"},
    "LINE_RADIAL_GENERATOR": {"p_line","c0","band_width"},
    "POINT_RADIAL_FORCE": {
        "P","force","x0","theta0","s0","radius_x_m","radius_s_m",
    },
}


@dataclass(frozen=True)
class RadialLoadProjection:
    generalized: np.ndarray
    face_pressure_pa: np.ndarray
    metadata: dict


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{name}: explicit finite number required")
    try: result=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"{name}: finite number required") from exc
    if not math.isfinite(result) or (positive and result<=0):
        raise ValueError(f"{name}: invalid numeric value or sign")
    return result


def _flag(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name}: boolean required")
    return value


def _finite_integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")
    return value


def _smooth(d, width):
    """C1 top-hat front: zero at d<=-width/2, unity at d>=width/2."""
    t=np.clip(np.asarray(d,dtype=float)/width+0.5,0.,1.)
    return np.sin(.5*np.pi*t)**2


def _bump(distance, radius):
    """Compact C1 cosine-squared bump, distance measured in metres."""
    t=np.asarray(distance,dtype=float)/radius
    return np.where(np.abs(t)<1.,np.cos(.5*np.pi*np.clip(t,-1.,1.))**2,0.)


def _axial_window(x, x1, x2, L, transition):
    if not (0<=x1<x2<=L):
        raise ValueError("axial load interval requires 0 <= x1 < x2 <= L")
    if transition> x2-x1:
        raise ValueError("edge_width_x exceeds axial loaded length")
    ans=np.ones_like(x)
    if x1>0:ans*=_smooth(x-x1,transition)
    if x2<L:ans*=_smooth(x2-x,transition)
    return ans


def _circumferential_window(s, c1, c2, R, coord_mode, transition):
    period=2*np.pi*R
    if coord_mode=="THETA":
        a,b=c1*R,c2*R
        transition*=R
    else:a,b=c1,c2
    start=a%period
    span=(b-a)%period
    if span<=1.e-12*period:
        raise ValueError("partial circumferential interval has zero physical span")
    if transition>span:
        raise ValueError("edge_width_c exceeds physical circumferential span")
    center=(start+span/2)%period
    delta=(s-center+period/2)%period-period/2
    signed=span/2-np.abs(delta)
    return _smooth(signed,transition)


def _geometry(functional, face):
    R,h=float(functional.R),float(functional.h)
    z={"OUTER":h/2,"MIDDLE":0.,"INNER":-h/2}[face]
    H=1.+z/R
    if H<=0:
        raise ValueError("loaded face has non-positive reference area metric")
    return z,H


def _component_projection(functional, z, H, face, pressure):
    nx,ns=functional.ops.shape
    n=int(functional.n_mechanical)
    out=np.zeros((n,nx,ns),float)
    theory=str(functional.theory).upper()
    if theory.startswith("CUF_TE"):
        order=int(theory[-1]);m=order+1
        if n!=3*m:raise ValueError("CUF field order differs from TE ansatz")
        xi=2*z/functional.h
        for k in range(m):
            out[2*m+k]=H*pressure*xi**k
    elif theory in ("FSDT","REDDY"):
        if n!=5:raise ValueError("director model must have five fields")
        out[2]=H*pressure
    elif theory=="KL":
        if face!="MIDDLE":
            raise ValueError(
                "KL offset-face radial traction requires nonlinear director "
                "variations and cannot be represented by nodal dead w load")
        if n!=3:raise ValueError("KL must have three fields")
        out[2]=pressure
    else:
        raise ValueError("unknown variational shell theory")
    return out


def project_radial_load(functional, definition):
    """Return generalized nodal load f_k in the SAME Π_h as MVI and FDM.

    f_k is energy-conjugate per midsurface area to q_k and is thus passed
    directly to CylinderFunctional.evaluate(dead_load=...).
    """
    if not isinstance(definition,dict):
        raise ValueError("load_definition must be an object")
    typ=str(definition.get("type","")).upper()
    if typ not in RADIAL_LOAD_TYPES:
        raise ValueError(f"radial variational type must be one of {RADIAL_LOAD_TYPES}")
    extras=set(definition)-(_COMMON|_SPECIFIC[typ])
    if extras:
        raise ValueError(f"unsupported {typ} fields: {sorted(extras)}")
    if str(definition.get("kinematics","")).upper()!="DEAD_REFERENCE":
        raise ValueError("reference variational load requires DEAD_REFERENCE, not follower")
    has_face="face" in definition
    has_surface="surface" in definition
    if has_face and has_surface and str(definition["face"]).upper()!=str(definition["surface"]).upper():
        raise ValueError("conflicting pressure face/surface")
    face=str(definition.get("face",definition.get("surface",""))).upper()
    if face not in FACES:
        raise ValueError(f"explicit physical face required: {FACES}")
    z,H=_geometry(functional,face)
    ops=functional.ops
    nx,ns=ops.shape
    area=ops.area_weights
    L=float(np.sum(ops.wx))
    R=float(functional.R)
    period=2*np.pi*R
    x=np.linspace(0.,L,nx)
    s=np.arange(ns,dtype=float)*(period/ns)
    step_x=L/(nx-1)
    step_s=period/ns
    direction=str(definition.get("direction","RADIAL_OUTWARD")).upper()
    if direction not in ("RADIAL_OUTWARD","RADIAL_INWARD"):
        raise ValueError("pressure direction must be RADIAL_OUTWARD or RADIAL_INWARD")
    sign=1. if direction=="RADIAL_OUTWARD" else -1.
    coord_mode=str(definition.get("circumferential_coordinate","THETA")).upper()
    if coord_mode not in ("THETA","S"):
        raise ValueError("circumferential_coordinate must be THETA or S")
    x1=_number(definition.get("x1",0.),"x1")
    x2=_number(definition.get("x2",L),"x2")
    # For exact force-normalized line/point loads, a geometric sampling width
    # is explicit, not derived from the mesh.
    if typ=="POINT_RADIAL_FORCE":
        if "x1" in definition or "x2" in definition or "c1" in definition or "c2" in definition:
            raise ValueError("POINT uses x0 and theta0/s0, not patch bounds")
        x0=_number(definition.get("x0"),"x0")
        if not 0<=x0<=L:raise ValueError("point x0 outside cylinder")
        if "theta0" in definition and "s0" in definition:
            raise ValueError("choose theta0 or s0, not both")
        theta0=_number(definition.get("theta0",0.),"theta0")
        s0=_number(definition.get("s0",theta0*R),"s0")%period
        rx=_number(definition.get("radius_x_m"),"radius_x_m",positive=True)
        rs=_number(definition.get("radius_s_m"),"radius_s_m",positive=True)
        if rx<step_x or rs<step_s:
            raise ValueError("point regularization underresolved; radii must "
                             "be >= one physical surface grid step")
        distance=np.abs(s-s0)
        distance=np.minimum(distance,period-distance)
        indicator=np.outer(_bump(x-x0,rx),_bump(distance,rs))
        target=sign*_number(definition.get("P",definition.get("force")),"P")
        raw=float(np.sum(area*H*indicator))
        if raw<=0:raise ValueError("point force regularization has zero quadrature support")
        pressure=target*indicator/raw
        expected=target
        intensity="N"
        normalized=True
    else:
        front_x=_number(definition.get("edge_width_x",min(2*step_x,x2-x1)),
                        "edge_width_x",positive=True)
        x_window=_axial_window(x,x1,x2,L,front_x)
        full=_flag(definition.get("full_circumference",typ in (
            "UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE")),
                   "full_circumference")
        if full:
            if "c1" in definition or "c2" in definition:
                raise ValueError("full_circumference cannot also set c1/c2")
            c_window=np.ones(ns,float)
        else:
            c1=_number(definition.get("c1"),"c1")
            c2=_number(definition.get("c2"),"c2")
            edge_c=_number(definition.get("edge_width_c",min(
                2*step_s/R if coord_mode=="THETA" else 2*step_s,
                (c2-c1)%(2*np.pi if coord_mode=="THETA" else period))),
                "edge_width_c",positive=True)
            c_window=_circumferential_window(s,c1,c2,R,coord_mode,edge_c)
        indicator=np.outer(x_window,c_window)
        expected=None;normalized=False;intensity="Pa"
        if typ in ("UNIFORM_RADIAL_FACE","PATCH_RADIAL_FACE"):
            if typ=="PATCH_RADIAL_FACE" and full and x1==0 and x2==L:
                raise ValueError("PATCH must specify a finite axial or circumferential region")
            pressure=sign*_number(definition.get("q"),"q")*indicator
        elif typ=="HARMONIC_RADIAL_FACE":
            m=_finite_integer(definition.get("m"),"m",1)
            n=_finite_integer(definition.get("n"),"n",0)
            phase=_number(definition.get("phase_rad",0.),"phase_rad")
            pressure=sign*_number(definition.get("q"),"q")*indicator*(
                np.sin(m*np.pi*x/L)[:,None]*np.cos(n*s/R+phase)[None,:])
        else:
            line_c=_number(definition.get("c0"),"c0")
            width=_number(definition.get("band_width"),"band_width",positive=True)
            if width<2*step_s or width>period:
                raise ValueError("generator band_width must lie in [2*ds,2*pi*R]")
            if x2-x1<2*step_x:
                raise ValueError("generator axial support must span >= 2*dx")
            c0=line_c*R if coord_mode=="THETA" else line_c
            distance=np.abs(s-(c0%period))
            distance=np.minimum(distance,period-distance)
            bump=_smooth(.5*width-distance,min(width,2*step_s))
            indicator=np.outer(x_window,bump)
            line_force=_number(definition.get("p_line"),"p_line")
            expected=sign*line_force*(x2-x1)
            raw=float(np.sum(area*H*indicator))
            if raw<=0:raise ValueError("generator band has zero quadrature support")
            pressure=expected*indicator/raw
            intensity="N/m"
            normalized=True
    if pressure.shape!=(nx,ns) or not np.isfinite(pressure).all():
        raise ValueError("nonfinite physical face pressure")
    generalized=_component_projection(functional,z,H,face,pressure)
    actual=float(np.sum(area*H*pressure))
    if normalized:
        tolerance=5.e-13*max(1.,abs(expected))
        if abs(actual-expected)>tolerance:
            raise ValueError("force-normalized radial load lost physical resultant")
    info={
        "type":typ,"face":face,"reference_face_offset_z_m":float(z),
        "reference_face_area_factor":float(H),
        "direction":direction,"kinematics":"DEAD_REFERENCE",
        "normalization":"RESULTANT" if normalized else "PRESCRIBED_FACE_PRESSURE",
        "input_unit":intensity,
        "face_force_radial_n":actual,
        "target_force_radial_n":None if expected is None else float(expected),
        "surface_nodes":int(nx*ns),
        "support_nodes":int(np.count_nonzero(pressure)),
        "grid_study_verified":False,
        "load_is_follower":False,
    }
    return RadialLoadProjection(generalized,pressure,info)


__all__=("RADIAL_LOAD_TYPES","FACES","RadialLoadProjection","project_radial_load")
