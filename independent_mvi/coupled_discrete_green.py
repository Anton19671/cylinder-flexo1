"""Exact DISCRETE Green identity for the independent coupled MVI surface.

No continuum boundary approximation is inserted here. The axial operator
Dx and its diagonal SBP norm Hx satisfy Hx Dx + Dx.T Hx = E_right-E_left.
The periodic odd-Fourier Ds is skew-adjoint in its constant norm Hs.

For any finite list of independent surface jets
  A[c,a,b] * delta((Dx**a) q (Ds**b).T),
the MVI *discrete* first variation splits exactly into
  volume = Hx Hs sum_a (-Dx)**a [sum_b (-Ds)**b A_ab]
  boundary = sum_k (Dx**k).T [E_right-E_left] Hs P_k,
  P_k = sum_(a>k) (-Dx)**(a-k-1) sum_b (-Ds)**b A_ab.
No product-rule approximation, unknown continuum material law, rank
change, MKR spatial stencil, SVD, or strong boundary enforcement.

An action conjugate to the physical outward-normal trace d_n**k q
is n**(k+1)*P_k at either end. The electric scalar potential has one
normal flux action and no electric inertial action.

This identity certifies algebraic boundary/volume decomposition for the
CURRENT chosen MVI SBP/Fourier semi-discretization. It does NOT establish
physical source-specific microfree 3D face conditions, the full correct
principal differential order, continuum convergence, or an MKR closure.
No end actions are silently applied to the nonlinear solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from independent_mvi.coupled_end_actions import formal_mvi_axial_end_actions
from independent_mvi.coupled_surface import (
    CoupledMVIWork, NativeCoupledMVISurface,
)


@dataclass(frozen=True)
class MVIDiscreteGreenIdentity:
    """Weighted nodal work and explicitly orientated end actions.

    Boundary actions are tuples per component / per trace order.
    coordinate_left/right pair with (delta q, delta q_x, delta q_xx).
    normal_left/right pair with (delta q, delta q_n, delta q_nn).
    The boundary_gradient is the exact adjoint trace lifting, NOT
    an independently imposed strong boundary-condition row.
    """
    volume_gradient: np.ndarray
    boundary_gradient: np.ndarray
    full_gradient: np.ndarray
    coordinate_left: tuple
    coordinate_right: tuple
    normal_left: tuple
    normal_right: tuple
    identity_max_abs_error: float
    identity_relative_error: float
    status: str="MVI_EXACT_DISCRETE_SBP_FOURIER_GREEN_IDENTITY"


def _max_abs(value):
    return float(np.max(np.abs(value),initial=0.))


def _sbp_and_fourier_guard(surface,relative_tolerance):
    """Reject inconsistent derivative/norm pairs before claiming identity."""
    grid=surface.grid
    hx=grid.wx
    hs=grid.ws
    dx=grid.dx
    ds=grid.ds
    B=np.zeros((grid.nx,grid.nx))
    B[0,0]=-1.
    B[-1,-1]=1.
    error_x=hx[:,None]*dx+dx.T*hx[None,:]-B
    error_s=hs[:,None]*ds+ds.T*hs[None,:]
    size_x=max(1.,_max_abs(hx[:,None]*dx),_max_abs(dx.T*hx[None,:]))
    size_s=max(1.,_max_abs(hs[:,None]*ds),_max_abs(ds.T*hs[None,:]))
    if (_max_abs(error_x)>relative_tolerance*size_x or
            _max_abs(error_s)>relative_tolerance*size_s):
        raise ValueError("MVI_DISCRETE_GREEN_REQUIRES_EXACT_SBP_FOURIER_PAIR")


def coupled_mvi_discrete_green_identity(
        surface: NativeCoupledMVISurface,
        work: CoupledMVIWork, *, relative_tolerance=2.e-9):
    """Verify exact weighted discrete Green identity for SAME weak work.

    Fail closed on absent/mismatched local coefficients, nodal gradient,
    non-SBP differentiation, or any mismatch. No physical shell solve.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("independent coupled MVI surface required")
    if isinstance(relative_tolerance,(bool,np.bool_)):
        raise ValueError("MVI_GREEN_TOLERANCE_INVALID")
    try:
        tolerance=float(relative_tolerance)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("MVI_GREEN_TOLERANCE_INVALID") from exc
    if not np.isfinite(tolerance) or not 0<tolerance<1.e-3:
        raise ValueError("MVI_GREEN_TOLERANCE_INVALID")
    formal=formal_mvi_axial_end_actions(surface,work)
    nodal=work.nodal_gradient
    if nodal is None or np.shape(nodal)!=surface.shape:
        raise ValueError("MVI_GREEN_REQUIRES_ACTUAL_2D_NODAL_FIRST_VARIATION")
    nodal=np.asarray(nodal,dtype=float)
    if not np.isfinite(nodal).all():
        raise ValueError("MVI_GREEN_NONFINITE_NODAL_FIRST_VARIATION")
    # The Green identity concerns volume weak JETS. A prescribed external
    # generalized end action is already a pure boundary variation and
    # must not be counted as a constitutive/hyperstress jet again.
    if work.external_end_gradient is not None:
        external=np.asarray(work.external_end_gradient,dtype=float)
        if external.shape!=surface.shape or not np.isfinite(external).all():
            raise ValueError("MVI_GREEN_EXTERNAL_END_WORK_SHAPE_INVALID")
        nodal=nodal-external
    _sbp_and_fourier_guard(surface,tolerance)

    dx=surface._dx_power
    weights=surface.grid.weights
    ws=surface.grid.ws
    volume=np.zeros(surface.shape,dtype=float)
    boundary=np.zeros(surface.shape,dtype=float)
    coord_left=[]
    coord_right=[]
    normal_left=[]
    normal_right=[]
    for c,reduced in enumerate(formal.circumferentially_reduced):
        max_ax=len(reduced)-1
        for a,Ca in enumerate(reduced):
            volume[c]+=(-1.)**a*(dx[a]@Ca)*weights
        field_left=[]
        field_right=[]
        field_nleft=[]
        field_nright=[]
        for k in range(max_ax):
            Pk=np.zeros((surface.grid.nx,surface.grid.ns))
            for a in range(k+1,max_ax+1):
                Pk+=(-1.)**(a-k-1)*(dx[a-k-1]@reduced[a])
            # Axial-coordinate derivative of the variation.
            L=-Pk[0].copy()
            R=Pk[-1].copy()
            field_left.append(L)
            field_right.append(R)
            # The k-th OUTWARD normal derivative is n**k * Dx**k.
            # Its conjugate action is n**(k+1)*Pk, not simply n*Pk
            # for all orders (left double traction has no sign).
            field_nleft.append((-1.)**k*L)
            field_nright.append(R.copy())
            end_node=np.zeros((surface.grid.nx,surface.grid.ns))
            end_node[0]=L*ws
            end_node[-1]=R*ws
            boundary[c]+=dx[k].T@end_node
        coord_left.append(tuple(field_left))
        coord_right.append(tuple(field_right))
        normal_left.append(tuple(field_nleft))
        normal_right.append(tuple(field_nright))

    total=volume+boundary
    defect=_max_abs(total-nodal)
    scale=max(1.,_max_abs(total),_max_abs(nodal),
              _max_abs(volume),_max_abs(boundary))
    relative=defect/scale
    if not np.isfinite(relative) or relative>tolerance:
        raise ValueError("MVI_DISCRETE_GREEN_FIRST_VARIATION_IDENTITY_FAILED")
    # A local read-only snapshot guards the result against accidental
    # modification by a caller and prevents mutation of source work.
    def locked(v):
        a=np.array(v,copy=True)
        a.setflags(write=False)
        return a
    def tuples_locked(x):
        return tuple(tuple(locked(v) for v in group) for group in x)
    return MVIDiscreteGreenIdentity(
        locked(volume),locked(boundary),locked(nodal),
        tuples_locked(coord_left),tuples_locked(coord_right),
        tuples_locked(normal_left),tuples_locked(normal_right),
        defect,relative,
    )


__all__=("MVIDiscreteGreenIdentity","coupled_mvi_discrete_green_identity")
