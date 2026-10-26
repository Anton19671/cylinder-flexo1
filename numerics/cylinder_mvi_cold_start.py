"""Residual-informed cold-start of inactive MVI partner profiles.

Uses a pivoted QR, never SVD. This is only a coordinate initialization of
EXACTLY ZERO separated fields. It does not modify displacement or potential,
raise rank, regularize the tangent or perform a full-nodal correction.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def seed_inactive_mvi_partners(x, y, admissible_gradient):
    """Align rank-fixed circumferential partners with admissible bulk residual.

    x has (components,rank,nx), y has (components,rank,ns), and the
    dimensionless admissible residual has (components,nx,ns). This applies
    identically to static equilibrium and the effective Newmark residual.
    """
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    g=np.asarray(admissible_gradient,dtype=float)
    if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=y.shape[:2]
            or g.shape!=(x.shape[0],x.shape[2],y.shape[2])
            or not 1<=x.shape[1]<=8 or x.shape[1]>min(x.shape[2],y.shape[2])
            or not all(np.isfinite(a).all() for a in (x,y,g))):
        raise ValueError(
            "MVI cold-start inputs require finite compatible fixed-rank shapes")
    seeded_y=y.copy()
    for component in range(x.shape[0]):
        # x==0 means changing its partner cannot alter the represented field.
        if np.any(x[component]) or not np.any(g[component]):
            continue
        q,_,_=qr(g[component].T,mode="economic",pivoting=True,
                 check_finite=False)
        seeded_y[component]=q[:,:x.shape[1]].T
    return x.copy(),seeded_y


__all__=("seed_inactive_mvi_partners",)
