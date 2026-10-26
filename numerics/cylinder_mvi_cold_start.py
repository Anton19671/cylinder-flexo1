"""Residual-informed cold-start of inactive MVI partner profiles.

Uses a pivoted QR, never SVD. This is only a coordinate initialization of
EXACTLY ZERO separated fields. It does not modify displacement or potential,
raise rank, regularize the tangent or perform a full-nodal correction.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def seed_inactive_mvi_partners(x, y, admissible_gradient, *, first_axis="X"):
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
    if first_axis not in ("X", "Y"):
        raise ValueError("MVI cold-start first axis must be X or Y")
    seeded_x=x.copy()
    seeded_y=y.copy()
    for component in range(x.shape[0]):
        if not np.any(g[component]):
            continue
        if first_axis=="X":
            # Frozen Y activates the first X half-step without changing
            # the identically zero physical field.
            if np.any(x[component]):
                continue
            q,_,_=qr(g[component].T,mode="economic",pivoting=True,
                     check_finite=False)
            seeded_y[component]=q[:,:x.shape[1]].T
        else:
            # Classical sandwich ordering: frozen X, then optimize Y.
            # Only exactly zero Y components can be reseeded.
            if np.any(y[component]):
                continue
            q,_,_=qr(g[component],mode="economic",pivoting=True,
                     check_finite=False)
            seeded_x[component]=q[:,:x.shape[1]].T
    return seeded_x,seeded_y


__all__=("seed_inactive_mvi_partners",)
