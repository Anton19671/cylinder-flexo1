"""Physical thickness displacement recovery for coupled CUF TE1..TE4 MVI.

Active weak-form basis is F_k(z)=(2*z/h)**k. This is deliberately NOT
the distinct legacy final GUI provider using dimensional z**k.
These are genuine 3D displacement values recovered from 2D MVI factors;
no new mesh, interpolation in x/s, SVD, or spatial solver is involved.

Array contract: displacement_stations_m[station,u-v-w,x,s], where stations
are [inner (-h/2), middle (0), outer (+h/2)]. Non-CUF director recovery
is NOT silently approximated with a CUF Taylor basis.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cuf_g3d_weak import te_basis


@dataclass(frozen=True)
class CUFPhysicalDisplacements:
    z_stations_m: np.ndarray
    displacement_stations_m: np.ndarray

    def __post_init__(self):
        if self.z_stations_m.shape!=(3,) or (
                self.displacement_stations_m.ndim!=4 or
                self.displacement_stations_m.shape[:2]!=(3,3) or
                not np.isfinite(self.z_stations_m).all() or
                not np.isfinite(self.displacement_stations_m).all()):
            raise ValueError("MVI_CUF_INVALID_PHYSICAL_DISPLACEMENT_STATIONS")


def recover_cuf_displacements(surface,mechanical):
    """Return three surfaces for CUF; None for KL/FSDT/Reddy.

    Restrict this recovery to the active normalized CUF TE weak basis.
    Never mislabel director or KL surface coefficients as TE polynomials.
    """
    if surface.theory not in (
            "CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
        return None
    order=int(surface.theory[-1])
    m=order+1
    values=np.asarray(mechanical,dtype=float)
    expected=(3*m,surface.grid.nx,surface.grid.ns)
    if values.shape!=expected or not np.isfinite(values).all():
        raise ValueError("MVI_CUF_PHYSICAL_RECOVERY_SHAPE_INVALID")
    coefficient=values.reshape(3,m,surface.grid.nx,surface.grid.ns)
    h=surface.thickness_m
    stations=np.array((-.5*h,0.,.5*h))
    output=np.stack([
        np.einsum("ckij,k->cij",coefficient,
                  te_basis(float(z),h,order)[0],optimize=True)
        for z in stations],axis=0)
    if not np.isfinite(output).all():
        raise ValueError("MVI_CUF_PHYSICAL_RECOVERY_NONFINITE")
    stations.setflags(write=False)
    output.setflags(write=False)
    return CUFPhysicalDisplacements(stations,output)


def radial_history_row(surface,mechanical,*,step,time_s):
    """Scalar history measured at axial midpoint, angular index zero.

    Every value carries the same explicit physical observation point.
    For CUF TE, inner/middle/outer W are fully thickness-recovered.
    For KL/FSDT/Reddy, only W of the reference surface is reported.
    """
    fields=np.asarray(mechanical,dtype=float)
    if (fields.shape!=(surface.n_mechanical,surface.grid.nx,surface.grid.ns)
            or not np.isfinite(fields).all()):
        raise ValueError("MVI_RADIAL_HISTORY_FIELD_SHAPE_INVALID")
    i=surface.grid.nx//2
    j=0
    recovered=recover_cuf_displacements(surface,fields)
    if recovered is None:
        middle=fields[2]
        inner=outer=None
    else:
        inner=recovered.displacement_stations_m[0,2]
        middle=recovered.displacement_stations_m[1,2]
        outer=recovered.displacement_stations_m[2,2]
    return dict(step=int(step),time_s=float(time_s),
                axial_node=i,circumferential_node=j,
                middle_w_m=float(middle[i,j]),
                inner_w_m=("" if inner is None else float(inner[i,j])),
                outer_w_m=("" if outer is None else float(outer[i,j])),
                maximum_abs_middle_w_m=float(np.max(np.abs(middle))),
                maximum_abs_inner_w_m=("" if inner is None
                                       else float(np.max(np.abs(inner)))),
                maximum_abs_outer_w_m=("" if outer is None
                                       else float(np.max(np.abs(outer)))))


__all__=("CUFPhysicalDisplacements","recover_cuf_displacements",
         "radial_history_row")
