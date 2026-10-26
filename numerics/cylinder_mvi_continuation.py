"""Rank-fixed continuation of accepted MVI X/Y profiles across load and Newmark steps.

Only the prescribed affine trace lift may change. The free separated field
is carried without QR recompression, SVD, nodal correction or rank growth.
"""
from __future__ import annotations
import numpy as np


def continue_accepted_mvi_profiles(previous, mechanical, electrical, *,
                                   new_lift_scaled, new_scales, full_space, rank):
    if previous is None or not bool(getattr(previous,"converged",False)):
        raise ValueError("MVI_CONTINUATION_REQUIRES_CONVERGED_PREVIOUS_STEP")
    certificate=getattr(previous,"mvi_certificate",None)
    if not isinstance(certificate,dict) or certificate.get("accepted") is not True:
        raise ValueError("MVI_CONTINUATION_REQUIRES_ACCEPTED_CERTIFICATE")
    q=np.asarray(mechanical,float)
    p=None if electrical is None else np.asarray(electrical,float)
    if q.ndim!=3 or not np.isfinite(q).all() or (
            p is not None and (p.ndim!=3 or p.shape[1:]!=q.shape[1:]
                               or not np.isfinite(p).all())):
        raise ValueError("MVI_CONTINUATION_INVALID_PHYSICAL_INPUT")
    shape=(q.shape[0]+(0 if p is None else p.shape[0]),*q.shape[1:])
    original=q if p is None else np.concatenate((q,p))
    x=np.asarray(previous.x_profiles,float)
    y=np.asarray(previous.y_profiles,float)
    old_lift=np.asarray(previous.lift_scaled,float)
    old_scale=np.asarray(previous.scales,float)
    new_scale=np.asarray(new_scales,float).reshape(shape)
    new_lift=np.asarray(new_lift_scaled,float).reshape(shape)
    if (isinstance(rank,(bool,np.bool_)) or not isinstance(rank,(int,np.integer))
            or not 1<=rank<=8 or x.shape!=(shape[0],rank,shape[1])
            or y.shape!=(shape[0],rank,shape[2])
            or old_lift.shape!=shape or old_scale.shape!=shape
            or not all(np.isfinite(a).all()
                       for a in (original,x,y,old_lift,old_scale,new_scale,new_lift))
            or np.any(old_scale<=0) or np.any(new_scale<=0)):
        raise ValueError("MVI_CONTINUATION_INCOMPATIBLE_FIXED_RANK_OR_SCALE")
    old_field=old_scale*(old_lift+np.einsum("crx,crs->cxs",x,y))
    error=float(np.max(np.abs(old_field-original),initial=0))
    scale=max(1.,float(np.max(np.abs(original),initial=0)),
              float(np.max(np.abs(old_field),initial=0)))
    if error>1.e-10*scale:
        raise ValueError("MVI_CONTINUATION_PREVIOUS_FIELD_MISMATCH")
    ratio=old_scale/new_scale
    component_ratio=ratio[:,0,0]
    if np.max(np.abs(ratio-component_ratio[:,None,None]),initial=0)>(
            1.e-12*max(1.,float(np.max(np.abs(ratio),initial=0)))):
        raise ValueError("MVI_CONTINUATION_NONSEPARABLE_SCALE_CHANGE")
    new_x=x*component_ratio[:,None,None]
    new_y=y.copy()
    transported=(new_lift+np.einsum("crx,crs->cxs",new_x,new_y)).ravel()
    if not np.isfinite(transported).all():
        raise ValueError("MVI_CONTINUATION_NONFINITE_TRANSPORTED_FIELD")
    defect=np.asarray(full_space.defect(transported),float)
    if not np.isfinite(defect).all() or np.max(np.abs(defect),initial=0)>1.e-9:
        raise ValueError("MVI_CONTINUATION_VARIATION_VIOLATES_NEW_TRACE")
    return new_x.copy(),new_y,{
        "initialization":"CONTINUATION",
        "rank":int(rank),
        "changed_boundary_lift_scaled":float(np.max(
            np.abs(new_lift-ratio*old_lift),initial=0)),
        "old_state_match_error_physical":error,
        "new_trace_defect_scaled":float(np.max(np.abs(defect),initial=0)),
    }


__all__=("continue_accepted_mvi_profiles",)
