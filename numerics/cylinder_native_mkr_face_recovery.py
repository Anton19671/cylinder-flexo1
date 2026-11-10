"""Physical electric induction on real cylinder faces for native strong MKR.

Only local constitutive laws are shared with the separate MVI method;
field differentiation is exclusively MKRFieldJets, never SBP/Fourier
MVI operators or variational-solver projections.

The face induction and outward normal charge balance are diagnostics.
The thickness-polynomial G3D/Gz shell does not automatically meet
every pointwise radial electric face condition. No extra strong face
equation is inserted without deriving the compatible source model.
"""
from __future__ import annotations
import math
import numpy as np

from cylinder_direct_strong_model import DirectStrongShell
from cylinder_mkr_field_jets import ELECTRIC_JETS
from flexo_gradient_g3d import enthalpy_derivatives


def _charge_field(item,shape):
    if item is None:
        return "INSULATED",np.zeros(shape)
    if not isinstance(item,dict) or set(item)!={"mode","charge_c_per_m2"}:
        raise ValueError("MKR_FACE_ENTRY_KEYS_INVALID")
    mode=item["mode"]
    if mode=="INSULATED":
        if item["charge_c_per_m2"] is not None:
            raise ValueError("MKR_FACE_INSULATED_REQUIRES_NULL")
        return mode,np.zeros(shape)
    if mode!="CHARGE":
        raise ValueError("MKR_FACE_POTENTIAL_NOT_IMPLEMENTED")
    raw=item["charge_c_per_m2"]
    if raw is None or isinstance(raw,(bool,np.bool_)):
        raise ValueError("MKR_FACE_CHARGE_INVALID")
    try:
        charge=np.asarray(raw,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("MKR_FACE_CHARGE_INVALID") from exc
    if charge.ndim==0:
        charge=np.full(shape,float(charge))
    if charge.shape!=shape or not np.isfinite(charge).all():
        raise ValueError("MKR_FACE_CHARGE_SHAPE_INVALID")
    return mode,charge


def recover_native_mkr_face_induction(model:DirectStrongShell,mechanical,
                                      electrical,*,face_specification=None):
    """Return JSON-safe physical surface induction diagnostic.

    Each face: D_z, normal D_n=n D_z, generalized prescribed charge
    sigma in C/m^2 and weighted RMS difference of D_n+sigma.
    No physical shell solve, boundary row or virtual-work assembly.
    """
    if not isinstance(model,DirectStrongShell) or not model.electrical_on:
        raise ValueError("MKR_FACE_RECOVERY_NEEDS_ELECTRICAL_MODEL")
    q=np.asarray(mechanical,dtype=float)
    p=np.asarray(electrical,dtype=float)
    if (q.shape!=model.shape or p.ndim!=3 or
            p.shape[1:]!=model.shape[1:] or
            not np.isfinite(q).all() or not np.isfinite(p).all()):
        raise ValueError("MKR_FACE_RECOVERY_STATE_SHAPE_INVALID")
    spec=({"inner":None,"outer":None} if face_specification is None
          else face_specification)
    if not isinstance(spec,dict) or set(spec)!={"inner","outer"}:
        raise ValueError("MKR_FACE_REQUIRES_INNER_OUTER")
    from cuf_g3d_weak import point_state as cuf_point
    from cylinder_director_g3d import director_point
    from cylinder_kl_g3d_weak import kl_point
    mechanical_jets=model.field_jets.prepare(q,model.derivative_orders)
    electric_jets=model.field_jets.prepare(p,ELECTRIC_JETS)
    shape=(model.grid.nx,model.grid.ns)
    dx=model.grid.dx
    ds=model.grid.ds
    weights=model.area_weights
    result={}
    for side,n in (("inner",-1),("outer",1)):
        mode,charge=_charge_field(spec[side],shape)
        z=n*model.thickness_m/2.
        D=np.zeros((model.grid.nx,model.grid.ns,3))
        for i in range(model.grid.nx):
            for j in range(model.grid.ns):
                jets=model._point_jets(mechanical_jets,electric_jets,i,j)
                common=dict(z_m=z,radius_m=model.grid.radius_m,
                            thickness_m=model.thickness_m,
                            geometry_model=model.geometry_model)
                if model.theory.startswith("CUF_TE"):
                    point=cuf_point(jets,**common)
                elif model.theory in ("FSDT","REDDY"):
                    point=director_point(jets,theory=model.theory,**common)
                else:
                    point=kl_point(jets,**common)
                D[i,j]=enthalpy_derivatives(
                    point.strain6,point.gradient6x3,
                    point.electric3,model.material)[2]
        outward=n*D[:,:,2]
        difference=outward+charge
        area=weights*(1.+z/model.grid.radius_m)
        physical_area=float(np.sum(area))
        mean=math.sqrt(float(np.sum(area*difference*difference)/
                             physical_area))
        if (not np.isfinite(D).all() or not np.isfinite(difference).all()
                or not math.isfinite(mean)):
            raise ValueError("MKR_FACE_RECOVERY_NONFINITE")
        result[side]=dict(
            mode=mode,normal=n,z_m=z,physical_area_m2=physical_area,
            natural_flux_difference_rms_c_per_m2=mean,
            natural_flux_difference_abs_max_c_per_m2=float(
                np.max(np.abs(difference),initial=0.)),
            induction_normal_mean_c_per_m2=float(
                np.sum(outward*area)/physical_area),
            diagnostic_only=True,pointwise_condition_enforced=False)
    return dict(physical_face_flux_certified=False,
                model=model.theory,gradient_model=model.material.gradient_model,
                faces=result)


__all__=("recover_native_mkr_face_induction",)
