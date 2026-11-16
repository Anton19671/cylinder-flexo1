"""Matrix-level axial Green diagnostic for independent native strong FDM.

This is a small, algebra-only audit, not a shell calculation or a substitute
for the strong solver.  It checks whether the two ACTUAL first-derivative
operators are discrete adjoints under the trapezoidal axial quadrature.

For q (kinematic field), A (physical conjugate axial flux), W (axial
quadrature), and B=diag(-1,0,...,+1), the continuum Green identity requires

    <A,D_kin q>_W + <q,D_flux A>_W = q(L) A(L) - q(0) A(0).

The matrix defect is D_kin.T W + W D_flux - B. A small defect is only an
axial first-order identity: it does not certify higher-order boundary
conditions, nonlinearity, electromagnetism, or the physical shell.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cylinder_direct_strong_fdm import (
    DirectStrongGrid, direct_coefficient_derivative,
)
from cylinder_mkr_field_jets import MKRFieldJets


@dataclass(frozen=True)
class NativeAxialGreenAudit:
    max_absolute_matrix_defect: float
    relative_matrix_defect: float
    defect_matrix: np.ndarray
    status: str


def audit_native_axial_green(grid: DirectStrongGrid) -> NativeAxialGreenAudit:
    """Inspect the actual axial derivative pair; no physical case is solved."""
    if not isinstance(grid, DirectStrongGrid):
        raise TypeError("DirectStrongGrid required")
    if grid.nx > 64 or grid.nx*grid.nx*grid.ns > 1_000_000:
        raise ValueError("AXIAL_GREEN_DIAGNOSTIC_SIZE_LIMIT")
    n=grid.nx
    # Each component is an axial Euclidean basis vector. Using the real
    # code paths, rather than recoding the stencils, detects future drift.
    basis=np.broadcast_to(np.eye(n)[:,:,None],(n,n,grid.ns)).copy()
    dkin=MKRFieldJets(grid).derivative(basis,1,0)[:,:,0].T
    dflux=direct_coefficient_derivative(basis,1,0,grid)[:,:,0].T
    w=np.full(n,grid.dx,float)
    w[[0,-1]]*=.5
    boundary=np.zeros((n,n),float)
    boundary[0,0]=-1.
    boundary[-1,-1]=1.
    kin_work=dkin.T*w[None,:]
    flux_work=w[:,None]*dflux
    defect=kin_work+flux_work-boundary
    if not np.isfinite(defect).all():
        raise ValueError("AXIAL_GREEN_NONFINITE_DEFECT")
    absolute=float(np.max(np.abs(defect)))
    reference=max(1.,float(np.max(np.abs(kin_work))),
                  float(np.max(np.abs(flux_work))))
    relative=absolute/reference
    defect.setflags(write=False)
    return NativeAxialGreenAudit(
        absolute,relative,defect,
        ("AXIAL_FIRST_ORDER_GREEN_PAIR_MATCHED_ONLY"
         if relative<=1.e-11 else "AXIAL_FIRST_ORDER_GREEN_PAIR_NOT_ADJOINT"))


__all__=("NativeAxialGreenAudit","audit_native_axial_green")
