"""Independent MVI electric inner/outer cylindrical FACE work and diagnostics.

Native fixed-rank MVI only. The physical thickness basis is G_k(z) =
(2z/h)**k. A prescribed free electrode charge sigma [C/m^2 physical
face area] contributes +int_face sigma*phi dA_face to the electric
enthalpy (same sign as existing generalized free charge). Therefore
its generalized midsurface charge is H_face*sigma*G_k(z_face), where
H_face=1+z_face/R, and NO extra sign is put on sigma.

The physical induction D is recovered at the REAL inner/outer
reference faces using the SAME local Krysko-Zhong enthalpy_derivatives,
without a reference solver, FDM stencil, or numerical interpolation.
The diagnostic natural flux condition is n_z*D_z+sigma=0, n_z=-1/+1.
This is NOT enforced pointwise in the thickness-reduced polynomial
electric approximation; it is NOT an acceptance/certification gate.

Face POTENTIAL (essential Dirichlet) and floating insulation gauge are
deliberately rejected until their coupled constraints are implemented.
No false claim of physical microfree/strong face closure is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from flexo_gradient_g3d import enthalpy_derivatives
from cuf_g3d_weak import te_basis
from independent_mvi.coupled_surface import (
    NativeCoupledMVISurface,CoupledMVIFactors,
)


@dataclass(frozen=True)
class MVIPhysicalElectricFace:
    side: str
    condition: str
    z_m: float
    outward_normal: int
    induction3_c_per_m2: np.ndarray
    outward_induction_c_per_m2: np.ndarray
    prescribed_free_charge_c_per_m2: np.ndarray
    natural_flux_difference_c_per_m2: np.ndarray
    physical_area_m2: float
    flux_difference_rms_c_per_m2: float
    flux_difference_abs_max_c_per_m2: float
    status: str="DIAGNOSTIC_ONLY_POINTWISE_FACE_FLUX_NOT_ENFORCED"


def _face_spec(surface, side, spec):
    if side not in ("inner","outer"):
        raise ValueError("MVI_ELECTRIC_FACE_SIDE_INVALID")
    if not surface.electrical_on:
        if spec is not None:
            raise ValueError("MVI_ELECTRIC_FACE_WITH_ELECTRICITY_OFF")
        return "OFF",None
    if spec is None:
        return "INSULATED",np.zeros((surface.grid.nx,surface.grid.ns))
    if not isinstance(spec,dict) or set(spec)!=set(("mode","charge_c_per_m2")):
        raise ValueError("MVI_ELECTRIC_FACE_SCHEMA_INVALID")
    mode=spec["mode"]
    if mode=="POTENTIAL":
        raise ValueError("MVI_FACE_DIRICHLET_POTENTIAL_NOT_IMPLEMENTED")
    if mode not in ("INSULATED","CHARGE"):
        raise ValueError("MVI_ELECTRIC_FACE_MODE_UNSUPPORTED")
    raw=spec["charge_c_per_m2"]
    if mode=="INSULATED":
        if raw is not None:
            raise ValueError("MVI_INSULATED_FACE_REQUIRES_NULL_CHARGE")
        return mode,np.zeros((surface.grid.nx,surface.grid.ns))
    if raw is None or isinstance(raw,(bool,np.bool_)):
        raise ValueError("MVI_FACE_CHARGE_FINITE_SCALAR_OR_FIELD_REQUIRED")
    try:
        charge=np.asarray(raw,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("MVI_FACE_CHARGE_FINITE_SCALAR_OR_FIELD_REQUIRED") from exc
    if charge.ndim==0:
        charge=np.full((surface.grid.nx,surface.grid.ns),float(charge))
    if (charge.shape!=(surface.grid.nx,surface.grid.ns)
            or not np.isfinite(charge).all()):
        raise ValueError("MVI_FACE_CHARGE_FIELD_SHAPE_OR_FINITE_INVALID")
    return mode,charge.copy()


def face_generalized_free_charge(surface:NativeCoupledMVISurface, *,
                                 inner=None,outer=None):
    """Return (n_electric,nx,ns) exact generalized face-charge work.

    INSULATED contributes zero. CHARGE contributes +H_face sigma G_k(z).
    POTENTIAL and electricity-OFF-with-face-spec are rejected.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("NativeCoupledMVISurface required")
    _,charge_inner=_face_spec(surface,"inner",inner)
    _,charge_outer=_face_spec(surface,"outer",outer)
    if not surface.electrical_on:
        return None
    out=np.zeros((surface.n_electric,surface.grid.nx,surface.grid.ns))
    for normal,charge in ((-1,charge_inner),(1,charge_outer)):
        z=normal*surface.thickness_m/2.
        H=1.+z/surface.grid.radius_m
        basis=te_basis(z,surface.thickness_m,surface.potential_order)[0]
        out+=basis[:,None,None]*H*charge[None,:,:]
    if not np.isfinite(out).all():
        raise ValueError("MVI_ELECTRIC_FACE_GENERALIZED_CHARGE_NONFINITE")
    return out


def recover_mvi_electric_faces(surface:NativeCoupledMVISurface,
                               factors:CoupledMVIFactors, *,
                               inner=None,outer=None):
    """Recover physical D and sign-correct natural face-flux diagnostics.

    The radial face area is H_face times the REFERENCE midsurface
    quadrature. Face flux mismatch is a DIAGNOSTIC, not an automatic
    rejection of an otherwise accepted polynomial electric field.
    """
    if not isinstance(surface,NativeCoupledMVISurface):
        raise TypeError("NativeCoupledMVISurface required")
    if not surface.electrical_on:
        raise ValueError("MVI_FACE_RECOVERY_REQUIRES_ELECTRICAL_MODEL")
    surface._check_factors(factors)
    modes=(_face_spec(surface,"inner",inner),
           _face_spec(surface,"outer",outer))
    from cuf_g3d_weak import point_state as cuf_point
    from cylinder_director_g3d import director_point
    from cylinder_kl_g3d_weak import kl_point
    _,_,fields=surface._surface_derivatives(factors)
    result={}
    for side,normal,mode in (("inner",-1,modes[0]),
                             ("outer",1,modes[1])):
        condition,charge=mode
        z=normal*surface.thickness_m/2.
        D=np.zeros((surface.grid.nx,surface.grid.ns,3))
        for i in range(surface.grid.nx):
            for j in range(surface.grid.ns):
                jets=surface._point(fields,i,j)
                kwargs=dict(z_m=z,radius_m=surface.grid.radius_m,
                            thickness_m=surface.thickness_m,
                            geometry_model=surface.geometry_model)
                if surface.theory.startswith("CUF_TE"):
                    point=cuf_point(jets,**kwargs)
                elif surface.theory in ("FSDT","REDDY"):
                    point=director_point(jets,theory=surface.theory,**kwargs)
                else:
                    point=kl_point(jets,**kwargs)
                D[i,j]=enthalpy_derivatives(
                    point.strain6,point.gradient6x3,
                    point.electric3,surface.material)[2]
        outward=normal*D[:,:,2]
        mismatch=outward+charge
        area_weights=surface.grid.weights*(
            1.+z/surface.grid.radius_m)
        physical_area=float(np.sum(area_weights))
        rms=math.sqrt(float(np.sum(
            area_weights*mismatch*mismatch)/physical_area))
        if not (np.isfinite(D).all() and np.isfinite(mismatch).all()
                and math.isfinite(rms)):
            raise ValueError("MVI_FACE_RECOVERY_NONFINITE_INDUCTION")
        def locked(raw):
            a=np.asarray(raw,dtype=float).copy()
            a.setflags(write=False)
            return a
        result[side]=MVIPhysicalElectricFace(
            side,condition,z,normal,locked(D),
            locked(outward),locked(charge),locked(mismatch),
            physical_area,rms,
            float(np.max(np.abs(mismatch),initial=0.)))
    return result


def summarize_mvi_electric_faces(recovered):
    """JSON-safe research-only diagnostics; never label as boundary PASS."""
    if not isinstance(recovered,dict) or set(recovered)!=set(("inner","outer")):
        raise ValueError("MVI_FACE_RECOVERY_MISSING_BOTH_SURFACES")
    return {
        "physical_face_flux_certified":False,
        "pointwise_condition_enforced":False,
        "faces":{
            side:{
                "mode":recovered[side].condition,
                "outward_normal":recovered[side].outward_normal,
                "z_m":recovered[side].z_m,
                "physical_area_m2":recovered[side].physical_area_m2,
                "natural_flux_difference_rms_c_per_m2":
                    recovered[side].flux_difference_rms_c_per_m2,
                "natural_flux_difference_abs_max_c_per_m2":
                    recovered[side].flux_difference_abs_max_c_per_m2,
                "status":recovered[side].status,
            } for side in ("inner","outer")
        },
    }


__all__=("MVIPhysicalElectricFace","face_generalized_free_charge",
         "recover_mvi_electric_faces","summarize_mvi_electric_faces")
