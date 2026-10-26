"""Basis-independent full admissible nodal residual metrics for MVI and FDM.

The dimensionless free coordinate residual is g=B.T (scales*r)/Escale.
Taking infinity norm of g depends on orthogonal rotation of B; use
nodal projection B@g for convergence and for separate mechanical/electric
checks, excluding constrained-DOF reactions.
"""
from __future__ import annotations
import numpy as np


def admissible_nodal_metrics(basis, reduced_residual, mechanical_dofs):
    b=np.asarray(basis,float)
    g=np.asarray(reduced_residual,float)
    if (b.ndim!=2 or g.ndim!=1 or b.shape[1]!=g.size
            or isinstance(mechanical_dofs,(bool,np.bool_))
            or not isinstance(mechanical_dofs,(int,np.integer))
            or not 0<=mechanical_dofs<=b.shape[0]
            or not np.isfinite(b).all() or not np.isfinite(g).all()):
        raise ValueError("invalid admissible nodal residual basis, coordinates or block sizes")
    nodal=b@g
    mechanical=float(np.max(np.abs(nodal[:mechanical_dofs]),initial=0))
    electrical=float(np.max(np.abs(nodal[mechanical_dofs:]),initial=0))
    return dict(full=max(mechanical,electrical),mechanical=mechanical,
                electrical=electrical,
                basis_coordinates=float(np.max(np.abs(g),initial=0)))


def admissible_nodal_inf_norm(basis, reduced_residual):
    b=np.asarray(basis,float)
    g=np.asarray(reduced_residual,float)
    if (b.ndim!=2 or g.ndim!=1 or b.shape[1]!=g.size
            or not np.isfinite(b).all() or not np.isfinite(g).all()):
        raise ValueError("invalid admissible basis or reduced residual")
    return float(np.max(np.abs(b@g),initial=0))


__all__=("admissible_nodal_metrics","admissible_nodal_inf_norm")
