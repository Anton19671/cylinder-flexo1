"""Independent pointwise electric-face residuals on the actual shell faces.

This is a diagnostic derived from the constitutive induction in the SAME
G3D/Gz mechanical/electric state solved by MVI or variational FDM. It is
not a substitute for the thickness-integrated electrical equilibrium.

For CHARGE the input is the physical free charge of the exterior electrode.
The variational work adds +integral(sigma * potential dA_face), and the
natural condition is D dot n_out + sigma = 0.
The inner face uses n_z=-1; the outer face uses n_z=+1.
"""
from __future__ import annotations

import math
import numpy as np
from cylinder_variational_boundary import _electric_mode, _time_factor, _number
from cylinder_generalized_face_traction import generalized_face_tractions


def face_conditions_from_recovery(functional, recovered, boundary_spec, *,
                                  calculation_kind="STATIC", time_s=None,
                                  electric_induction_scale_c_per_m2=None):
    if not isinstance(boundary_spec, dict):
        raise ValueError("boundary_spec must be a dictionary")
    electr=boundary_spec.get("electrical", {})
    if not isinstance(electr, dict):
        raise ValueError("electrical BC must be a dictionary")
    kind=str(calculation_kind).upper()
    if kind not in ("STATIC", "DYNAMIC"):
        raise ValueError("calculation_kind must be STATIC or DYNAMIC")
    if kind=="DYNAMIC" and (
        time_s is None or not math.isfinite(float(time_s))):
        raise ValueError("dynamic faces require a finite time_s")
    scale=None
    if electric_induction_scale_c_per_m2 is not None:
        scale=_number(electric_induction_scale_c_per_m2, "electric induction scale")
        if scale<=0:
            raise ValueError("electric induction scale must be positive")
    report={
        "electrical_enabled":bool(functional.electrical_on),
        "electric_induction_scale_c_per_m2":scale,
        "electric_faces":{},
        "mechanical_face_strong_check":{
            "status":"NOT_IMPLEMENTED_FULL_SECOND_GRADIENT_FACE_OPERATOR",
            "message":"Classical stress alone omits higher tractions, curvature and edge terms"
        }
    }
    # Only the normal-normal double traction is locally available.
    # The COMPLETE generalized mechanical traction still requires the
    # surface divergence of higher-order stress and curvature/edge terms.
    # Do not classify this partial diagnostic as a strong-form PASS.
    for side in ("inner","outer"):
        name=f"face_{side}_normal_double_traction3_n_per_m"
        if name not in recovered:
            raise ValueError(f"missing mechanical double-traction recovery: {name}")
        g=np.asarray(recovered[name],float)
        if g.shape!=(*functional.ops.shape,3) or not np.isfinite(g).all():
            raise ValueError("invalid mechanical face double-traction shape")
        report["mechanical_face_strong_check"][f"{side}_double_traction_abs_max_n_per_m"]=(
            float(np.max(np.abs(g),initial=0.)))
    report["mechanical_face_strong_check"]["available_component"]=(
        "NORMAL_DOUBLE_AND_GENERALIZED_TRACTION_CANDIDATES")
    # A second-gradient strong face candidate is recovered from physical
    # tensors; it is NOT the variational weak residual or an acceptance gate.
    output=generalized_face_tractions(
        recovered,functional.ops,radius_m=functional.R)
    surface_weights=np.asarray(functional.W,float)
    for side,zsign in (("inner",-1.),("outer",1.)):
        H=1.+zsign*functional.h/(2.*functional.R)
        area_weights=surface_weights*H
        area=float(np.sum(area_weights))
        for field,unit in (
            ("generalized_traction3_pa_candidate","pa"),
            ("normal_double_traction3_n_per_m_candidate","n_per_m")):
            vector=np.asarray(output[f"face_{side}_{field}"],float)
            squared=np.sum(vector*vector,axis=-1)
            rms=float(np.sqrt(np.sum(area_weights*squared)/area))
            report["mechanical_face_strong_check"][
                f"{side}_{field}_rms_{unit}"]=rms
            report["mechanical_face_strong_check"][
                f"{side}_{field}_abs_max_{unit}"]=float(
                    np.max(np.abs(vector),initial=0))
    report["mechanical_face_strong_check"]["candidate_status"]=(
        "COVARIANT_RECOVERY_APPROXIMATE_NOT_INDEPENDENTLY_VERIFIED")
    report["mechanical_face_strong_check"]["strong_check_performed"]=False
    if not functional.electrical_on:
        report["electric_status"]="DISABLED"
        return output,report
    d=np.asarray(recovered["induction3_c_per_m2"],float)
    nx,ns=functional.ops.shape
    if (d.ndim!=4 or d.shape[0]<2 or d.shape[1:]!=(nx,ns,3)
            or not np.isfinite(d).all()):
        raise ValueError("invalid recovered physical induction shape or value")
    w=np.asarray(functional.W,float)
    if w.shape!=(nx,ns) or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError("invalid physical surface quadrature")
    for side,index,normal in (("inner",0,-1.),("outer",-1,1.)):
        site=electr.get(side,{})
        if not isinstance(site,dict):
            raise ValueError(side+" electrical condition must be an object")
        mode=_electric_mode(site)
        outward=normal*d[index,:,:,2]
        output["face_"+side+"_electric_outward_induction_c_per_m2"]=outward.copy()
        H=1.+normal*functional.h/(2.*functional.R)
        face_weight=w*H
        area=float(np.sum(face_weight))
        values={
            "mode":mode,
            "physical_face_area_m2":area,
            "normal_sign_reference_z":normal,
            "natural_flux_check_applicable":mode!="POTENTIAL",
        }
        if mode=="POTENTIAL":
            values.update(
                status="DIRICHLET_POTENTIAL_FACE_FLUX_IS_REACTION",
                outward_flux_abs_max_c_per_m2=float(np.max(np.abs(outward),initial=0)))
        else:
            prescribed=(0. if mode=="INSULATED"
                        else _number(site.get("value"),side+" prescribed charge")
                        *_time_factor(site,kind,time_s))
            residual=outward+prescribed
            output["face_"+side+"_electric_outward_residual_c_per_m2"]=residual.copy()
            rms=float(np.sqrt(np.sum(face_weight*residual*residual)/area))
            values.update(
                status="DIAGNOSTIC_ONLY_NOT_EQUILIBRIUM_ACCEPTANCE_GATE",
                prescribed_free_charge_c_per_m2=float(prescribed),
                prescribed_outward_induction_c_per_m2=float(-prescribed),
                normal_residual_rms_c_per_m2=rms,
                normal_residual_abs_max_c_per_m2=float(
                    np.max(np.abs(residual),initial=0)),
                normal_residual_relative_rms=(None if scale is None else rms/scale))
        report["electric_faces"][side]=values
    report["electric_status"]="DIAGNOSTIC_ONLY"
    return output,report


__all__=("face_conditions_from_recovery",)
