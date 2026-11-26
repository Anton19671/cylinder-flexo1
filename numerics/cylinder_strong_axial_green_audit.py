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
    DirectStrongGrid, direct_coefficient_derivative, direct_axial_end_derivative,
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
    w=grid.axial_weights
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



@dataclass(frozen=True)
class NativeHigherAxialGreenAudit:
    """Per-jet mismatch of native weak work vs sampled strong bulk/end work.

    The fields are ordered by axial derivative 1, 2, 3. Each defect
    matrix is read-only and maps a conjugate coefficient to the mismatch
    in the unweighted virtual nodal gradient.
    """
    orders: tuple[int, ...]
    max_absolute_defect: tuple[float, ...]
    relative_defect: tuple[float, ...]
    defect_matrices: tuple[np.ndarray, ...]
    status: str


def audit_native_higher_axial_green(
        grid: DirectStrongGrid, *, highest_order: int = 3
        ) -> NativeHigherAxialGreenAudit:
    """Audit EVERY axial order with the current independent native stencils.

    For the single coefficient A_k conjugate to D_kin,k q the discrete
    virtual work is q.T D_kin,k.T W A_k. The continuum-style strong form
    and sampled physical end actions instead produce

      q.T [(-1)^k W D_flux,k
          + sum(j=0..k-1) (-1)^(k-j-1)
            D_kin,j.T B D_flux,k-j-1] A_k.

    B has the outward -/+ endpoint signs. D_kin,0=D_flux,0=I.
    The j=1,2 terms are the *actual* essential normal-slope/curvature
    traces, not arbitrary seven-node traces. The residual matrix
    measures precisely the missing discrete Green compatibility.

    Diagnostic only: its strong bulk is not silently replaced by the
    variational adjoint, and its continuum face tractions are not
    relabelled as certified discrete reactions. No shell solve is run.
    """
    if not isinstance(grid, DirectStrongGrid):
        raise TypeError("DirectStrongGrid required")
    if type(highest_order) is not int or highest_order not in (1, 2, 3):
        raise ValueError("HIGHER_AXIAL_GREEN_ORDER_MUST_BE_1_TO_3")
    if grid.nx>64 or grid.nx*grid.nx*grid.ns>1_000_000:
        raise ValueError("AXIAL_GREEN_DIAGNOSTIC_SIZE_LIMIT")

    n=grid.nx
    basis=np.broadcast_to(np.eye(n)[:,:,None],(n,n,grid.ns)).copy()
    jets=MKRFieldJets(grid)
    identity=np.eye(n,dtype=float)
    kinematic=[identity]
    coefficient=[identity]
    for order in range(1,highest_order+1):
        kinematic.append(jets.derivative(basis,order,0)[:,:,0].T)
        coefficient.append(
            direct_coefficient_derivative(basis,order,0,grid)[:,:,0].T)

    weights=grid.axial_weights
    W=np.diag(weights)
    boundary=np.zeros((n,n),dtype=float)
    boundary[0,0]=-1.
    boundary[-1,-1]=1.

    defects=[]
    absolute=[]
    relative=[]
    for order in range(1,highest_order+1):
        weak=kinematic[order].T@W
        strong=((-1.)**order)*(W@coefficient[order])
        for trace in range(order):
            strong+=((-1.)**(order-trace-1))*(
                kinematic[trace].T@boundary@
                direct_axial_end_derivative(basis,order,trace,grid)[:,:,0].T)
        defect=weak-strong
        if not np.isfinite(defect).all():
            raise ValueError("HIGHER_AXIAL_GREEN_NONFINITE_DEFECT")
        norm=max(1.,float(np.max(np.abs(weak))),
                 float(np.max(np.abs(strong))))
        absolute.append(float(np.max(np.abs(defect))))
        relative.append(absolute[-1]/norm)
        defect.setflags(write=False)
        defects.append(defect)
    return NativeHigherAxialGreenAudit(
        tuple(range(1,highest_order+1)),
        tuple(absolute),tuple(relative),tuple(defects),
        ("HIGHER_AXIAL_GREEN_PAIR_MATCHED_ALGEBRA_ONLY"
         if max(relative)<=1.e-11 else
         "HIGHER_AXIAL_GREEN_MISMATCH_NOT_BOUNDARY_CERTIFIED"))


__all__=("NativeAxialGreenAudit","audit_native_axial_green",
         "NativeHigherAxialGreenAudit","audit_native_higher_axial_green")
