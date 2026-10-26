#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit load builder for the final cylinder interface.

This is an input-generation layer only; it does not alter shell mechanics.
Supported physical loads are radial face tractions, because this projection is
already explicitly implemented in the verified general-CUF mechanics.

V1/V11 safety rules:
- LINE_RADIAL_GENERATOR is represented by a finite *physical* circumferential
  band. Its width is independent of mesh spacing; the old p_line/ds point
  collocation is intentionally not available.
- PATCH_RADIAL_FACE keeps fixed physical bounds and uses a C1 cos^2 transition
  at internal patch fronts. The automatic transition spans the half-width of
  the active finite-difference stencil (3 cells for a 7-point stencil), so the
  front is resolved without changing the nominal physical patch dimensions.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

import cylinder_te1_mvi_core as mvi
import cylinder_cuf_stage7a_general_mechanics as g7

LOAD_TYPES=(
    "UNIFORM_RADIAL_FACE",
    "HARMONIC_RADIAL_FACE",
    "PATCH_RADIAL_FACE",
    "LINE_RADIAL_GENERATOR",
    "POINT_RADIAL_FORCE",
    "EXTERNAL_NPZ",
)
FACES=("OUTER","MIDDLE","INNER")
COORD_MODES=("THETA","S")
LOAD_KINEMATICS=("DEAD_REFERENCE",)

class LoadDefinitionError(ValueError): pass


def normalize_load_kinematics(spec):
    """V8: current verified load arrays are nominal/dead in the reference configuration."""
    if not isinstance(spec,dict):
        raise LoadDefinitionError("load definition must be an object")
    kin=str(spec.get("kinematics","DEAD_REFERENCE")).strip().upper()
    aliases={"DEAD":"DEAD_REFERENCE","REFERENCE":"DEAD_REFERENCE","NOMINAL":"DEAD_REFERENCE"}
    kin=aliases.get(kin,kin)
    if kin not in LOAD_KINEMATICS:
        raise LoadDefinitionError(
            "load kinematics must be DEAD_REFERENCE; follower/current-normal load is not implemented "
            "because it requires deformation-dependent load direction/area and its consistent tangent"
        )
    return kin


def _finite(v,name):
    if v is None:
        raise LoadDefinitionError(f"{name} must be supplied")
    try:
        x=float(v)
    except Exception as exc:
        raise LoadDefinitionError(f"{name} must be a finite number") from exc
    if not np.isfinite(x): raise LoadDefinitionError(f"{name} must be finite")
    return x


def _positive(v,name):
    x=_finite(v,name)
    if x<=0.0: raise LoadDefinitionError(f"{name} must be > 0")
    return x


def _factors(spec):
    a=spec.get("load_factors")
    if a is None: raise LoadDefinitionError("load_factors must be supplied explicitly")
    a=np.asarray(a,float)
    if a.ndim!=1 or a.size<1 or not np.all(np.isfinite(a)):
        raise LoadDefinitionError("load_factors must be a finite non-empty 1D array")
    return a


def _periodic_mask(values,a,b,period):
    v=np.mod(np.asarray(values,float),period)
    aa=float(a)%period; bb=float(b)%period
    if aa <= bb:
        return (v >= aa-1e-14) & (v <= bb+1e-14)
    return (v >= aa-1e-14) | (v <= bb+1e-14)


def _periodic_span(a,b,period):
    aa=float(a)%period; bb=float(b)%period
    span=(bb-aa)%period
    return aa,bb,span


def _periodic_signed_distance_to_interval(values,a,b,period):
    """Positive inside periodic interval a->b, negative outside."""
    v=np.mod(np.asarray(values,float),period)
    aa,_,span=_periodic_span(a,b,period)
    if span<=1e-14*max(1.0,period):
        raise LoadDefinitionError(
            "PATCH circumferential endpoints coincide; use full_circumference=true for a full ring"
        )
    center=(aa+0.5*span)%period
    delta=(v-center+0.5*period)%period-0.5*period
    return 0.5*span-np.abs(delta),span


def _cos2_front_from_signed_distance(d,width):
    """C1 transition: 0 at d<=-w/2, 1 at d>=+w/2."""
    d=np.asarray(d,float)
    w=float(width)
    if w<=0.0:
        return (d>=0.0).astype(float)
    t=np.clip((d+0.5*w)/w,0.0,1.0)
    return np.sin(0.5*np.pi*t)**2


def _smooth_nonperiodic_interval(values,a,b,domain_a,domain_b,width):
    """Smooth top-hat; do not taper at a physical domain boundary."""
    v=np.asarray(values,float)
    w=float(width)
    out=np.ones_like(v,float)
    tol=1e-14*max(1.0,abs(domain_b-domain_a))
    if a>domain_a+tol:
        out*=_cos2_front_from_signed_distance(v-a,w)
    else:
        out*=np.where(v>=domain_a-tol,1.0,0.0)
    if b<domain_b-tol:
        out*=_cos2_front_from_signed_distance(b-v,w)
    else:
        out*=np.where(v<=domain_b+tol,1.0,0.0)
    return out


def _surface_integral(q,grid):
    wx=np.full(grid.nx,grid.dx,float)
    wx[0]*=0.5; wx[-1]*=0.5
    return float(np.sum(np.asarray(q,float)*grid.ds*wx[None,:]))


def _face_geometry(face,R,h):
    """Reference offset and area metric of a concentric cylinder surface."""
    face=str(face).strip().upper()
    if face not in FACES:
        raise LoadDefinitionError(f"surface must be one of {FACES}")
    R=float(R); h=float(h)
    if not np.isfinite(R) or R<=0.0:
        raise LoadDefinitionError("R must be finite and >0")
    if not np.isfinite(h) or h<=0.0:
        raise LoadDefinitionError("h must be finite and >0")
    zq={"OUTER":0.5*h,"MIDDLE":0.0,"INNER":-0.5*h}[face]
    Hq=1.0+zq/R
    if Hq<=0.0:
        raise LoadDefinitionError(
            "inner loaded surface has non-positive radius; require h < 2R"
        )
    return face,zq,Hq


def _auto_front_nodes(stencil):
    st=int(stencil)
    if st<3 or st%2!=1:
        raise LoadDefinitionError("stencil must be an odd integer >= 3")
    return (st-1)//2


def _patch_field(spec,grid,coord,period,cm,mx_bounds,stencil):
    qval=_finite(spec.get("q"),"q")
    x1,x2=mx_bounds
    full=bool(spec.get("full_circumference",False))
    nfront=_auto_front_nodes(stencil)

    xspan=x2-x1
    if xspan<=0.0:
        raise LoadDefinitionError("PATCH x2 must be greater than x1")
    wx_user=spec.get("edge_width_x")
    wx=(nfront*grid.dx if wx_user is None else _positive(wx_user,"edge_width_x"))
    # A transition wider than the patch itself suppresses the nominal plateau.
    # AUTO is capped safely; an explicit incompatible request is rejected.
    if wx_user is None:
        wx=min(wx,xspan)
    elif wx>xspan*(1.0+1e-12):
        raise LoadDefinitionError("edge_width_x must not exceed PATCH axial width x2-x1")
    rx=_smooth_nonperiodic_interval(grid.x,x1,x2,0.0,grid.length_x,wx)

    if full:
        rs=np.ones(grid.ns,float)
        cspan=period
        wc=0.0
    else:
        a=_finite(spec.get("c1"),"c1"); b=_finite(spec.get("c2"),"c2")
        signed,cspan=_periodic_signed_distance_to_interval(coord,a,b,period)
        wc_user=spec.get("edge_width_c",spec.get("edge_width"))
        dc=(grid.ds/grid.radius) if cm=="THETA" else grid.ds
        wc=(nfront*dc if wc_user is None else _positive(wc_user,"edge_width_c"))
        if wc_user is None:
            wc=min(wc,cspan)
        elif wc>cspan*(1.0+1e-12):
            raise LoadDefinitionError("edge_width_c must not exceed PATCH circumferential width")
        rs=_cos2_front_from_signed_distance(signed,wc)

    weight=np.outer(rs,rx)
    q=qval*weight
    front=(weight>1e-12)&(weight<1.0-1e-12)
    support=weight>1e-12

    arc_span=(cspan*grid.radius) if cm=="THETA" else cspan
    target_area=xspan*(grid.circumference if full else arc_span)
    target_resultant=qval*target_area
    discrete_resultant=_surface_integral(q,grid)
    meta={
        "selected_surface_nodes":int(np.count_nonzero(support)),
        "load_front_surface_nodes":int(np.count_nonzero(front)),
        "patch_nominal_area":float(target_area),
        "patch_target_resultant":float(target_resultant),
        "patch_discrete_resultant":float(discrete_resultant),
        "patch_discrete_resultant_rel_error":float(
            abs(discrete_resultant-target_resultant)/max(abs(target_resultant),1e-300)
        ) if target_resultant!=0.0 else 0.0,
        "front_smoothing":"COS2_C1",
        "front_half_stencil_nodes":int(nfront),
        "edge_width_x_actual":float(wx),
        "edge_width_c_actual":float(wc),
    }
    return q,meta,{"front_surface_mask":front,"support_surface_mask":support}


def _point_radial_field(spec, grid):
    """Concentrated radial force distributed conservatively to nearest nodes."""
    # This legacy generator uses a conservative nearest-node stencil.
    # Physical C1 radius inputs belong to the variational G3D generator;
    # rejecting them here prevents silently changing the requested load.
    unsupported={"s0","radius_x_m","radius_s_m"} & set(spec)
    if unsupported:
        raise LoadDefinitionError(
            f"legacy point-load stencil cannot use {sorted(unsupported)}; "
            "use variational G3D physical C1 point regularization")
    if "P" in spec and "force" in spec:
        raise LoadDefinitionError("select P or force, not both")
    P=_finite(spec.get("P", spec.get("force")), "P")
    x0=_finite(spec.get("x0"), "x0")
    th0=_finite(spec.get("theta0",0.0), "theta0")
    if x0 < -1e-12 or x0 > grid.length_x+1e-12:
        raise LoadDefinitionError("x0 must be inside [0,L]")
    th=np.mod(th0,2.0*math.pi)
    # nearest node in x and periodic circumferential coordinate
    ix=int(np.argmin(np.abs(grid.x-x0)))
    dsang=grid.s/grid.radius
    d=np.abs(dsang-th); d=np.minimum(d,2.0*math.pi-d)
    js=int(np.argmin(d))
    weights=np.zeros((grid.ns,grid.nx),float)
    ids=[]
    for dj in (-1,0,1):
        for di in (-1,0,1):
            j=(js+dj)%grid.ns; i=ix+di
            if 0<=i<grid.nx:
                ids.append((j,i))
    if not ids:
        raise LoadDefinitionError("point load stencil is empty")
    w=1.0/len(ids)
    for j,i in ids:
        weights[j,i]=w

    # `weights` currently sums to 1 as a plain nodal indicator (a force
    # SHARE per node). Downstream (KL weak-projected residual and CUF
    # radial_nominal_face_load) treats the returned field as a PRESSURE
    # and integrates it against the physical area weights W=ds*dx (see
    # surface_weights()/_surface_integral()). Passing weights*P straight
    # through therefore silently rescales the requested force P by the
    # local nodal area (mesh-dependent, and wrong by orders of magnitude
    # for any real grid). Convert to an equivalent nodal pressure so the
    # SAME area integral the solver performs reproduces exactly P,
    # independent of mesh resolution.
    area_under_weights=_surface_integral(weights,grid)
    if area_under_weights<=0.0:
        raise LoadDefinitionError("point load stencil has zero effective area")
    q=weights*(P/area_under_weights)

    discrete_resultant=_surface_integral(q,grid)
    meta={
        "point_force":float(P),"point_x0":float(x0),"point_theta0":float(th),
        "P_magnitude":float(abs(P)),
        "point_target_resultant":float(P),
        "point_discrete_resultant":float(discrete_resultant),
        "point_discrete_resultant_rel_error":float(
            abs(discrete_resultant-P)/max(abs(P),1e-300)
        ) if P!=0.0 else 0.0,
        "point_nodal_area_effective":float(area_under_weights),
    }
    support=weights>0.0
    return q,meta,{"front_surface_mask":np.zeros_like(support,dtype=bool),"support_surface_mask":support}


def _surface_radial_field(spec,grid,*,stencil=7):
    typ=str(spec.get("type","")).upper()
    if typ=="POINT_RADIAL_FORCE":
        return _point_radial_field(spec, grid)
    if typ not in LOAD_TYPES[:-1]:
        raise LoadDefinitionError(f"unsupported generated load type: {typ}")
    q=np.zeros((grid.ns,grid.nx),float)
    x=grid.x; s=grid.s

    x1=_finite(spec.get("x1",0.0),"x1") if "x1" in spec else 0.0
    x2=_finite(spec.get("x2",grid.length_x),"x2") if "x2" in spec else grid.length_x
    if x1>x2: x1,x2=x2,x1
    if x1 < -1e-14 or x2 > grid.length_x+1e-14:
        raise LoadDefinitionError("x interval must lie inside [0,L]")
    mx=(x>=x1-1e-14)&(x<=x2+1e-14)

    if typ=="UNIFORM_RADIAL_FACE":
        qval=_finite(spec.get("q"),"q")
        q[:,mx]=qval
        support=np.broadcast_to(mx,(grid.ns,grid.nx)).copy()
        front=np.zeros_like(support,dtype=bool)
        meta={"selected_surface_nodes":int(np.count_nonzero(support)),"load_front_surface_nodes":0,"front_smoothing":"NONE"}
        return q,meta,{"front_surface_mask":front,"support_surface_mask":support}

    if typ=="HARMONIC_RADIAL_FACE":
        # Analytic harmonic traction optionally restricted to a physical
        # axial/circumferential region.  The harmonic law itself is unchanged;
        # the region is a C1 window using the same front treatment as PATCH.
        q0=_finite(spec.get("q"),"q")
        try:
            mh=float(spec.get("m"))
            nh=float(spec.get("n"))
        except Exception as exc:
            raise LoadDefinitionError("HARMONIC_RADIAL_FACE requires integer m and n") from exc
        if not np.isfinite(mh) or not np.isfinite(nh) or int(mh)!=mh or int(nh)!=nh:
            raise LoadDefinitionError("HARMONIC_RADIAL_FACE m and n must be integers")
        mh=int(mh); nh=int(nh)
        if mh < 1:
            raise LoadDefinitionError("HARMONIC_RADIAL_FACE m must be >= 1")
        if nh < 0:
            raise LoadDefinitionError("HARMONIC_RADIAL_FACE n must be >= 0")

        theta=s/grid.radius
        axial=np.sin(mh*math.pi*x/grid.length_x)
        circum=np.cos(nh*theta)
        analytic=np.outer(circum,axial)

        nfront=_auto_front_nodes(stencil)
        xspan=x2-x1
        if xspan<=0.0:
            raise LoadDefinitionError("HARMONIC x2 must be greater than x1")
        wx_user=spec.get("edge_width_x")
        wx=(nfront*grid.dx if wx_user is None else _positive(wx_user,"edge_width_x"))
        if wx_user is None:
            wx=min(wx,xspan)
        elif wx>xspan*(1.0+1e-12):
            raise LoadDefinitionError("edge_width_x must not exceed HARMONIC axial width x2-x1")
        rx=_smooth_nonperiodic_interval(grid.x,x1,x2,0.0,grid.length_x,wx)

        full=bool(spec.get("full_circumference",True))
        cm=str(spec.get("circumferential_coordinate","THETA")).upper()
        if cm not in COORD_MODES:
            raise LoadDefinitionError(f"circumferential_coordinate must be one of {COORD_MODES}")
        period=2*math.pi if cm=="THETA" else grid.circumference
        coord=s/grid.radius if cm=="THETA" else s
        if full:
            rs=np.ones(grid.ns,float); cspan=period; wc=0.0
        else:
            a=_finite(spec.get("c1"),"c1"); bb=_finite(spec.get("c2"),"c2")
            signed,cspan=_periodic_signed_distance_to_interval(coord,a,bb,period)
            wc_user=spec.get("edge_width_c",spec.get("edge_width"))
            dc=(grid.ds/grid.radius) if cm=="THETA" else grid.ds
            wc=(nfront*dc if wc_user is None else _positive(wc_user,"edge_width_c"))
            if wc_user is None:
                wc=min(wc,cspan)
            elif wc>cspan*(1.0+1e-12):
                raise LoadDefinitionError("edge_width_c must not exceed HARMONIC circumferential width")
            rs=_cos2_front_from_signed_distance(signed,wc)

        window=np.outer(rs,rx)
        q=q0*analytic*window
        support=window>1e-12
        front=(window>1e-12)&(window<1.0-1e-12)
        arc_span=(cspan*grid.radius) if cm=="THETA" else cspan
        meta={
            "selected_surface_nodes":int(np.count_nonzero(support)),
            "load_front_surface_nodes":int(np.count_nonzero(front)),
            "front_smoothing":"ANALYTIC_HARMONIC_X_COS2_WINDOW" if not full or x1>0.0 or x2<grid.length_x else "ANALYTIC_HARMONIC",
            "harmonic_q0":float(q0),
            "harmonic_m":int(mh),
            "harmonic_n":int(nh),
            "harmonic_formula":"q*cos(n*s/R)*sin(m*pi*x/L)",
            "harmonic_region_x1":float(x1),
            "harmonic_region_x2":float(x2),
            "harmonic_full_circumference":bool(full),
            "harmonic_region_arc_width":float(grid.circumference if full else arc_span),
            "edge_width_x_actual":float(wx),
            "edge_width_c_actual":float(wc),
        }
        return q,meta,{"front_surface_mask":front,"support_surface_mask":support}

    cm=str(spec.get("circumferential_coordinate","THETA")).upper()
    if cm not in COORD_MODES: raise LoadDefinitionError(f"circumferential_coordinate must be one of {COORD_MODES}")
    period=2*math.pi if cm=="THETA" else grid.circumference
    coord=s/grid.radius if cm=="THETA" else s

    if typ=="PATCH_RADIAL_FACE":
        return _patch_field(spec,grid,coord,period,cm,(x1,x2),stencil)

    # V1: a line intensity is represented by a finite physical band.
    # The old q=p_line/ds single-node collocation is intentionally removed:
    # it has no grid-convergent strong-form limit.
    p_line=_finite(spec.get("p_line"),"p_line")
    c0=_finite(spec.get("c0"),"c0")%period
    b=_positive(spec.get("band_width"),"band_width")
    if b>grid.circumference*(1.0+1e-12):
        raise LoadDefinitionError("band_width must not exceed cylinder circumference")
    if b < 2.0*grid.ds*(1.0-1e-12):
        ns_min=int(math.ceil(2.0*grid.circumference/b))
        raise LoadDefinitionError(
            f"band_width={b:.6g} is smaller than two circumferential grid steps "
            f"2*ds={2.0*grid.ds:.6g}; increase ns to at least {ns_min}"
        )
    d=np.abs(coord-c0)
    d=np.minimum(d,period-d)
    d_phys=d*grid.radius if cm=="THETA" else d
    # The finite band also gets the same C1 front treatment as PATCH.  The
    # nominal band width remains physical and mesh-independent; only the
    # transition layer shrinks with refinement.
    nfront=_auto_front_nodes(stencil)
    wband=min(nfront*grid.ds,b)
    rs=_cos2_front_from_signed_distance(0.5*b-d_phys,wband)
    band_measure=float(np.sum(rs)*grid.ds)
    if band_measure<=0.0:
        raise LoadDefinitionError("line-load band has zero discrete measure")
    q[:,mx]=(p_line/band_measure)*rs[:,None]

    # Preserve the requested total force p_line*(x2-x1) exactly under the
    # same trapezoidal surface integration used by the solver.  This matters
    # when an axial endpoint lies between nodes.  For the usual full-length
    # generator the factor is exactly one.
    xspan=float(x2-x1)
    if xspan<=0.0:
        raise LoadDefinitionError("LINE x2 must be greater than x1")
    target_resultant=float(p_line*xspan)
    raw_resultant=_surface_integral(q,grid)
    if target_resultant!=0.0:
        if raw_resultant==0.0:
            raise LoadDefinitionError("line load has zero discrete resultant")
        q*=target_resultant/raw_resultant
    qpeak=float(np.max(np.abs(q))) if q.size else 0.0
    support=rs>1e-12
    front=(rs>1e-12)&(rs<1.0-1e-12)
    # Report a collocation-consistent diagnostic, without implying that a
    # weak-form integral proves point-load convergence.
    if np.any(mx):
        vals=q[:,np.flatnonzero(mx)[0]]
        discrete_intensity=float(np.sum(vals)*grid.ds)
    else:
        discrete_intensity=0.0
    meta={
        "selected_surface_nodes":int(np.count_nonzero(support)*np.count_nonzero(mx)),
        "load_front_surface_nodes":int(np.count_nonzero(front)*np.count_nonzero(mx)),
        "line_coordinate_requested":float(c0),
        "band_width_physical":float(b),
        "band_nodes":int(np.count_nonzero(support)),
        "band_peak_pressure_nominal":float(abs(p_line)/b),
        "band_peak_pressure_discrete":float(np.max(np.abs(q)) if q.size else 0.0),
        "band_edge_width_actual":float(wband),
        "front_smoothing":"COS2_C1",
        "line_target_intensity":float(p_line),
        "line_discrete_intensity_estimate":float(discrete_intensity),
        "line_discrete_intensity_rel_error":float(
            abs(discrete_intensity-p_line)/max(abs(p_line),1e-300)
        ) if p_line!=0.0 else 0.0,
        "line_target_resultant":target_resultant,
        "line_discrete_resultant":float(_surface_integral(q,grid)),
        "line_discrete_peak_pressure_abs":qpeak,
    }
    support2=np.broadcast_to(support[:,None],(grid.ns,grid.nx)).copy() & np.broadcast_to(mx,(grid.ns,grid.nx))
    front2=np.broadcast_to(front[:,None],(grid.ns,grid.nx)).copy() & np.broadcast_to(mx,(grid.ns,grid.nx))
    return q,meta,{"front_surface_mask":front2,"support_surface_mask":support2}


def build_generated_loads(spec,*,theory,L,R,h,nx,ns,stencil=7,basis=None,return_masks=False):
    if not isinstance(spec,dict): raise LoadDefinitionError("load_definition must be an object")
    typ=str(spec.get("type","")).upper()
    if typ=="EXTERNAL_NPZ": raise LoadDefinitionError("EXTERNAL_NPZ is loaded by the dispatcher")
    if typ not in LOAD_TYPES: raise LoadDefinitionError(f"type must be one of {LOAD_TYPES}")
    load_kinematics=normalize_load_kinematics(spec)
    theory=str(theory).upper()
    grid=mvi.CylinderGrid(float(L),float(R),int(nx),int(ns))
    radial,meta,masks=_surface_radial_field(spec,grid,stencil=int(stencil))
    factors=_factors(spec)
    if theory=="KL":
        requested_surface=spec.get("surface",spec.get("face"))
        if requested_surface not in (None,""):
            requested_surface=str(requested_surface).strip().upper()
            if requested_surface!="MIDDLE":
                raise LoadDefinitionError(
                    "KL uses a midsurface load only; OUTER/INNER face semantics are not available. "
                    "Уберите поле face/surface из CASE или задайте MIDDLE."
                )
        surface_name="MIDDLE"
    else:
        surface_name=spec.get("surface",spec.get("face",""))
    surface,zq,Hq=_face_geometry(surface_name,R,h)

    # Pressure-controlled loads are specified per unit area of the selected
    # physical face.  Point force and line intensity are resultant-controlled:
    # their requested force must be independent of the selected face radius.
    resultant_controlled=typ in ("POINT_RADIAL_FORCE","LINE_RADIAL_GENERATOR")
    if resultant_controlled:
        radial_equivalent=radial
        radial_face=radial/Hq
    else:
        radial_face=radial
        radial_equivalent=Hq*radial_face

    if theory=="CUF":
        if basis is None: raise LoadDefinitionError("CUF basis is required")
        p0=np.asarray(g7.radial_nominal_face_load(radial_face,zq=zq,R=float(R),basis=basis),float)
    elif theory=="KL":
        # Every implemented generated load is radial on a concentric surface.
        # Its line of action passes through the corresponding midsurface point,
        # hence the equivalent KL face moment is zero; only the exact face-area
        # metric Hq enters the midsurface force resultant.
        p0=np.zeros((3,grid.ns,grid.nx),float); p0[2]=radial_equivalent
    else:
        raise LoadDefinitionError("theory must be CUF or KL")
    loads=np.stack([f*p0 for f in factors],axis=0)
    meta.update({
        "type":typ,"theory":theory,"nx":grid.nx,"ns":grid.ns,
        "dx":float(grid.dx),"ds":float(grid.ds),"stencil":int(stencil),
        "surface":surface,"face":surface,
        "surface_offset_zq":float(zq),"surface_area_metric_Hq":float(Hq),
        "load_control":"RESULTANT" if resultant_controlled else "FACE_PRESSURE",
        "radial_min":float(radial_face.min()),"radial_max":float(radial_face.max()),
        "radial_face_min":float(radial_face.min()),
        "radial_face_max":float(radial_face.max()),
        "equivalent_midsurface_radial_resultant":float(_surface_integral(radial_equivalent,grid)),
        "physical_face_radial_resultant":float(Hq*_surface_integral(radial_face,grid)),
        "equivalent_face_moment_about_midsurface":0.0,
        "load_factors":factors.tolist(),
        "load_kinematics":load_kinematics,
        "load_direction_update":"FIXED_IN_REFERENCE_CONFIGURATION",
        "pressure_direction":spec.get("pressure_direction"),
        "q_magnitude":spec.get("q_magnitude"),
        "p_line_magnitude":spec.get("p_line_magnitude"),
        "P_magnitude":spec.get("P_magnitude", meta.get("P_magnitude")),
    })
    if typ=="POINT_RADIAL_FORCE":
        target=float(meta["point_target_resultant"])
        actual=float(meta["physical_face_radial_resultant"])
        meta["point_discrete_resultant"]=actual
        meta["point_discrete_resultant_rel_error"]=(
            abs(actual-target)/max(abs(target),1e-300) if target!=0.0 else 0.0
        )
    elif typ=="LINE_RADIAL_GENERATOR":
        target=float(meta["line_target_resultant"])
        actual=float(meta["physical_face_radial_resultant"])
        meta["line_discrete_resultant"]=actual
        meta["line_discrete_resultant_rel_error"]=(
            abs(actual-target)/max(abs(target),1e-300) if target!=0.0 else 0.0
        )
    elif typ=="PATCH_RADIAL_FACE":
        area_mid=float(meta["patch_nominal_area"])
        target_mid=float(meta["patch_target_resultant"])
        meta["patch_nominal_area_midsurface"]=area_mid
        meta["patch_nominal_area_face"]=float(Hq*area_mid)
        meta["patch_target_resultant_midsurface"]=target_mid
        meta["patch_target_resultant"]=float(Hq*target_mid)
        meta["patch_discrete_resultant"]=float(meta["physical_face_radial_resultant"])
        target=float(meta["patch_target_resultant"])
        meta["patch_discrete_resultant_rel_error"]=(
            abs(meta["patch_discrete_resultant"]-target)/max(abs(target),1e-300)
            if target!=0.0 else 0.0
        )
    if return_masks:
        return loads,factors,meta,radial_face,masks
    return loads,factors,meta,radial_face


def save_generated_npz(path,spec,**kwargs):
    loads,factors,meta,radial=build_generated_loads(spec,**kwargs)
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,loads=loads,load_factors=factors,radial_surface_field=radial)
    return meta
