"""Carry classical fixed-rank X/Y products without an additive field."""
import numpy as np
def continue_accepted_mvi_profiles(previous,mechanical,electrical,*,new_scales,full_space,rank):
    if previous is None or not previous.converged:
        raise ValueError("MVI_CONTINUATION_REQUIRES_CONVERGED_PREVIOUS_STEP")
    cert=previous.mvi_certificate
    if not isinstance(cert,dict) or not cert.get("accepted",False):
        raise ValueError("MVI_CONTINUATION_REQUIRES_ACCEPTED_CERTIFICATE")
    q=np.asarray(mechanical,float)
    p=None if electrical is None else np.asarray(electrical,float)
    actual=q if p is None else np.concatenate((q,p))
    x=np.asarray(previous.x_profiles,float);y=np.asarray(previous.y_profiles,float)
    old=np.asarray(previous.scales,float)
    if (not 1<=rank<=8 or x.shape!=(actual.shape[0],rank,actual.shape[1])
        or y.shape!=(actual.shape[0],rank,actual.shape[2]) or old.shape!=actual.shape):
        raise ValueError("MVI_CONTINUATION_INCOMPATIBLE_FIXED_RANK_OR_SCALE")
    new=np.asarray(new_scales,float).reshape(actual.shape)
    if np.any(new<=0) or not all(np.isfinite(v).all() for v in (x,y,old,new,actual)):
        raise ValueError("MVI_CONTINUATION_INVALID_SCALES")
    prior=old*np.einsum("crx,crs->cxs",x,y)
    mismatch=float(np.max(np.abs(prior-actual),initial=0))
    if mismatch>1e-10*max(1.,float(np.max(np.abs(actual),initial=0))):
        raise ValueError("MVI_CONTINUATION_PREVIOUS_FIELD_MISMATCH")
    ratio=old/new
    if not np.allclose(ratio,ratio[:,0:1,0:1],rtol=1e-12,atol=1e-12):
        raise ValueError("MVI_CONTINUATION_NONSEPARABLE_SCALE_CHANGE")
    xx=x*ratio[:,0,0,None,None]
    yy=y.copy()
    defect=np.asarray(full_space.defect(np.einsum("crx,crs->cxs",xx,yy).ravel()))
    return xx,yy,{"initialization":"CONTINUATION","rank":rank,
       "old_state_match_error_physical":mismatch,
       "new_trace_defect_before_halfstep_scaled":float(np.max(np.abs(defect),initial=0))}
