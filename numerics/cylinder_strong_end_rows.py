"""Physical-node axial end closure for the separate direct strong FDM route.

Krysko--Zhong section 41, equations (222)--(226). This module only places
already assembled physical PDE and constitutive flux values into rows. It is
not used by the energy-consistent variational FDM and does not derive the
coupled higher traction from displacement differences.

The full-gradient branch is valid only when every mechanical subsystem has
the nondegenerate fourth-order axial operator. A mixed-order material needs
its own operator-rank analysis and is deliberately rejected here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

CLASSICAL_SECOND_ORDER="CLASSICAL_SECOND_ORDER"
FULL_GRADIENT_FOURTH_ORDER="FULL_GRADIENT_FOURTH_ORDER"


@dataclass(frozen=True)
class AxialRowLayout:
    left_essential: int
    left_higher: int | None
    interior: tuple[int,...]
    right_higher: int | None
    right_essential: int


def axial_row_layout(nx, *, branch):
    """Give exactly one equation row per physical axial node and component."""
    if isinstance(nx,(bool,np.bool_)) or not isinstance(nx,(int,np.integer)):
        raise ValueError("nx must be an integer physical-node count")
    if branch==CLASSICAL_SECOND_ORDER and nx>=3:
        return AxialRowLayout(0,None,tuple(range(1,nx-1)),None,nx-1)
    if branch==FULL_GRADIENT_FOURTH_ORDER and nx>=5:
        return AxialRowLayout(0,1,tuple(range(2,nx-2)),nx-2,nx-1)
    raise ValueError("unsupported strong axial branch or insufficient physical nodes")


def axial_end_derivatives(field, dx_m):
    """Return (left/right first, left/right second) from (223)--(226).

    The axial node axis is -2, so input may be (nx,ns) or (nfield,nx,ns).
    Mixed end derivatives are obtained by applying the periodic tangential
    derivative to these results at the corresponding end.
    """
    f=np.asarray(field,float)
    if f.ndim not in (2,3) or f.shape[-2]<4 or f.shape[-1]<1 or not np.isfinite(f).all():
        raise ValueError("finite surface field (nx>=4,ns>=1) required")
    if isinstance(dx_m,(bool,np.bool_)):
        raise ValueError("positive finite axial spacing required")
    try:dx=float(dx_m)
    except (TypeError,ValueError) as exc:
        raise ValueError("positive finite axial spacing required") from exc
    if not math.isfinite(dx) or dx<=0:
        raise ValueError("positive finite axial spacing required")
    first_left=(-3*f[...,0,:]+4*f[...,1,:]-f[...,2,:])/(2*dx)
    second_left=(2*f[...,0,:]-5*f[...,1,:]+4*f[...,2,:]-f[...,3,:])/dx**2
    first_right=(3*f[...,-1,:]-4*f[...,-2,:]+f[...,-3,:])/(2*dx)
    second_right=(2*f[...,-1,:]-5*f[...,-2,:]
                  +4*f[...,-3,:]-f[...,-4,:])/dx**2
    return first_left,first_right,second_left,second_right


def replace_strong_end_rows(pde, displacement, *, branch,
                            prescribed_left=None, prescribed_right=None,
                            higher_left=None, higher_right=None):
    """Install essential and constitutive end rows, leaving interior PDE.

    All mechanical inputs have shape (nfield,nx,ns). Higher values are the
    full physical P_xx(q,psi) at each end, shape (nfield,ns); a zero-valued
    placeholder is never inferred from a classical limit.
    """
    r=np.asarray(pde,float)
    q=np.asarray(displacement,float)
    if (r.ndim!=3 or q.shape!=r.shape or not np.isfinite(r).all()
            or not np.isfinite(q).all()):
        raise ValueError("finite matching mechanical PDE and displacement fields required")
    nfield,nx,ns=r.shape
    if nfield<1 or ns<1:
        raise ValueError("nonempty physical fields required")
    layout=axial_row_layout(nx,branch=branch)
    shape=(nfield,ns)

    def end_array(value,label,*,required=False):
        if value is None:
            if required:raise ValueError(label+" requires a full constitutive end flux")
            return np.zeros(shape,float)
        a=np.asarray(value,float)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError(label+" must be finite (nfield,ns)")
        return a

    left=end_array(prescribed_left,"left prescribed displacement")
    right=end_array(prescribed_right,"right prescribed displacement")
    rows=r.copy()
    rows[:,layout.left_essential,:]=q[:,0,:]-left
    rows[:,layout.right_essential,:]=q[:,-1,:]-right
    if branch==FULL_GRADIENT_FOURTH_ORDER:
        rows[:,layout.left_higher,:]=end_array(higher_left,"left higher traction",required=True)
        rows[:,layout.right_higher,:]=end_array(higher_right,"right higher traction",required=True)
    elif higher_left is not None or higher_right is not None:
        raise ValueError("classical second-order branch has no higher traction rows")
    return rows


__all__=("CLASSICAL_SECOND_ORDER","FULL_GRADIENT_FOURTH_ORDER",
         "AxialRowLayout","axial_row_layout","axial_end_derivatives",
         "replace_strong_end_rows")
