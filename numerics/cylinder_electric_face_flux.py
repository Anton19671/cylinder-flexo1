"""Independent dielectric-only face-flux finite differences on a cylinder.

This is a validation subproblem for the electric part of section 40.1,
not a completed strong coupled shell solver. No constitutive SVD or
pseudo-inverse is used. Field axis convention is (nx,ns).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from cylinder_staggered_surface import StaggeredCylinderSurface


@dataclass(frozen=True)
class DielectricFaceFluxResult:
    residual_c: np.ndarray
    axial_face_flux_c_per_m: np.ndarray
    circumferential_face_flux_c_per_m: np.ndarray
    potential_energy_j: float
    net_residual_c: float
    circumferential_checkerboard_detected: bool


def dielectric_face_flux_fdm(potential_v, *, length_m, radius_m, thickness_m,
                             kappa_f_per_m, free_charge_c_per_m2=None):
    """Compute weak nodal charge from independent staggered face fluxes.

    Scalar potential is independent of thickness (TE0 electric coefficient).
    Axial effective thickness is h; circumferential is
    R*log((R+h/2)/(R-h/2)). These are exact curved-shell integrals.
    End faces are natural/insulated. Fluxes include the thickness integral.
    """
    phi=np.asarray(potential_v,float)
    if phi.ndim!=2 or min(phi.shape)<4 or not np.isfinite(phi).all():
        raise ValueError("potential_v must be finite (nx>=4,ns>=4)")
    nx,ns=phi.shape
    values=(length_m,radius_m,thickness_m,kappa_f_per_m)
    if any(isinstance(v,(bool,np.bool_)) for v in values):
        raise ValueError("geometry/permittivity must be finite SI scalars")
    try: L,R,h,kappa=map(float,values)
    except (TypeError,ValueError) as exc:
        raise ValueError("geometry/permittivity must be finite SI scalars") from exc
    if (not all(map(math.isfinite,(L,R,h,kappa))) or
            L<=0 or h<=0 or R<=h/2 or kappa<=0):
        raise ValueError("require L>0, R>h/2>0, kappa>0")
    surface=StaggeredCylinderSurface(nx,ns,L,R)
    dx,ds,wx=surface.dx,surface.ds,surface.wx
    # Stable integration: int[-h/2,h/2] dz/(1+z/R).
    is_thickness=R*math.log1p(h/(R-h/2))
    gx,gs=surface.gradients(phi)
    # D = kappa*E, E=-grad(phi). These are thickness integrated C/m.
    fx=-kappa*h*gx
    fs=-kappa*is_thickness*gs
    residual=surface.weak_face_action(fx,fs)
    energy=-.5*kappa*(h*dx*ds*float(np.sum(gx*gx))+
                       is_thickness*ds*float(np.sum(wx[:,None]*gs*gs)))
    if free_charge_c_per_m2 is not None:
        charge=np.asarray(free_charge_c_per_m2,float)
        if charge.shape!=phi.shape or not np.isfinite(charge).all():
            raise ValueError("free_charge shape/nonfinite")
        supplied=wx[:,None]*ds*charge
        total=float(np.sum(supplied))
        if abs(total)>1.e-12*max(1.,float(np.sum(np.abs(supplied)))):
            raise ValueError("insulated dielectric requires zero total charge")
        residual+=supplied
        energy+=float(np.sum(supplied*phi))
    checkerboard=False
    if ns%2==0:
        checker=(-1.)**np.arange(ns)
        checkerboard=bool(np.linalg.norm(np.roll(checker,-1)-checker)>0)
    return DielectricFaceFluxResult(
        residual_c=residual,axial_face_flux_c_per_m=fx,
        circumferential_face_flux_c_per_m=fs,
        potential_energy_j=float(energy),
        net_residual_c=float(np.sum(residual)),
        circumferential_checkerboard_detected=checkerboard)


__all__=("DielectricFaceFluxResult","dielectric_face_flux_fdm")
