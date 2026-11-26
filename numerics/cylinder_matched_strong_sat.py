"""Explicit natural-trace SAT correction for a matched strong FDM bulk.

The bulk is already formed by direct coefficient differentiation. This
module only adds boundary work lifted by the positive axial norm. It does
not construct a weak residual, solve a system, or import MVI/energy FDM.
"""
import numpy as np


def add_natural_trace_sat(bulk, weights, traces, actions, targets):
    """Return bulk + H^-1 sum T_k^T (a_k - g_k).

    bulk has shape (field,nx,ns), weights (nx,), each outward trace
    matrix (2,nx), each action and target (2,field,ns). The two sides
    are left/right. All arrays are physical, not dimensionless.
    Essential equations are imposed separately AFTER lifting natural
    work, so no interior bulk row is deleted for a natural condition.
    """
    r=np.asarray(bulk,float)
    w=np.asarray(weights,float)
    if (r.ndim!=3 or min(r.shape)<1 or w.shape!=(r.shape[1],)
            or not np.isfinite(r).all() or not np.isfinite(w).all()
            or np.any(w<=0.)):
        raise ValueError("MATCHED_SAT_INVALID_BULK_OR_WEIGHTS")
    if not (len(traces)==len(actions)==len(targets)):
        raise ValueError("MATCHED_SAT_TRACE_ACTION_COUNT_MISMATCH")
    out=r.copy()
    shape=(2,r.shape[0],r.shape[2])
    for trace,action,target in zip(traces,actions,targets):
        T=np.asarray(trace,float)
        a=np.asarray(action,float);g=np.asarray(target,float)
        if (T.shape!=(2,r.shape[1]) or a.shape!=shape or g.shape!=shape
                or not all(np.isfinite(x).all() for x in (T,a,g))):
            raise ValueError("MATCHED_SAT_INVALID_TRACE_OR_ACTION")
        out+=np.einsum("si,scj->cij",T/w[None,:],a-g,optimize=True)
    if not np.isfinite(out).all():
        raise ValueError("MATCHED_SAT_NONFINITE_RESIDUAL")
    return out
