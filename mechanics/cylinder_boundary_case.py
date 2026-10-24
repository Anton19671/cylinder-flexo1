#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference BC CASE validation and strict production compatibility gate."""
from __future__ import annotations
import math
from cylinder_variational_boundary import ELECTRIC_SITES, ENDS, validate_boundary_passport

CLASSICAL_STATUS="CLASSICAL_PRESET_REUSED"
REFERENCE_STATUS="REFERENCE_ONLY"
VECTOR_KEYS=("prescribed","traction","double_traction","triple_traction","spring","spring_reference")
PRESETS={"FREE","CLAMPED","SIMPLY_SUPPORTED","GUIDED","SYMMETRY"}


def inspect_variational_case(case):
    bc=case.get("boundary_conditions",{})
    if not isinstance(bc,dict):raise ValueError("boundary_conditions must be an object")
    spec=bc.get("variational")
    if spec is None:return None
    if not isinstance(spec,dict):raise ValueError("variational BC must be an object")
    theory=str(case.get("theory","")).upper()
    if theory=="CUF":
        theory="CUF_"+str(case.get("cuf_model","TE1")).upper()
    if theory=="CUF_TE1" or theory in ("CUF_TE2","CUF_TE3","CUF_TE4"):
        n=3*(int(theory[-1])+1)
    elif theory=="KL":n=3
    elif theory in ("FSDT","REDDY"):n=5
    else:raise ValueError("unknown variational theory")
    kind=str(case.get("calculation_kind","STATIC")).upper()
    scale=str(case.get("scale_model","MACRO")).upper()
    if scale not in ("MACRO","MICRO"):raise ValueError("unknown scale_model")
    micro=case.get("micro",{}) if scale=="MICRO" else {}
    if not isinstance(micro,dict):raise ValueError("micro must be an object")
    flag=micro.get("enable_flexoelectric",False)
    if type(flag)!=bool:raise ValueError("enable_flexoelectric must be boolean")
    electrical=scale=="MICRO" and flag
    validate_boundary_passport(spec,electrical_enabled=electrical,
                               calculation_kind=kind,theory=theory)
    mechanics=spec.get("mechanical",{})
    electric=spec.get("electrical",{})
    reasons=[]
    for side in ENDS:
        end=mechanics.get(side,{})
        support=str(end.get("support","FREE")).upper()
        higher=str(end.get("higher","NATURAL")).upper()
        if higher!="NATURAL" and scale!="MICRO":
            raise ValueError(side+": higher traces require MICRO G3D")
        if higher!="NATURAL":reasons.append(side+": G3D higher-order trace")
        if support not in PRESETS:
            reasons.append(side+": CUSTOM reference constraints")
            fixed=end.get("fixed_components",[])
            if not isinstance(fixed,list) or any(type(i)!=int or i<0 or i>=n for i in fixed) or len(set(fixed))!=len(fixed):
                raise ValueError(side+": invalid CUSTOM fixed_components")
        elif "fixed_components" in end:
            raise ValueError(side+": fixed_components requires CUSTOM")
        for key in VECTOR_KEYS:
            if key in end:
                vals=end[key]
                if not isinstance(vals,(list,tuple)) or len(vals)!=n:
                    raise ValueError(side+"."+key+": expected exactly "+str(n)+" entries")
                for x in vals:
                    if isinstance(x,bool):raise ValueError(side+"."+key+": nonfinite")
                    try:v=float(x)
                    except (ValueError,TypeError) as exc:raise ValueError(side+"."+key+": nonnumeric") from exc
                    if not math.isfinite(v):raise ValueError(side+"."+key+": nonfinite")
                    if key=="spring" and v<0:raise ValueError(side+".spring must be nonnegative")
                    if v!=0:reasons.append(side+": nonzero "+key)
        if "time_law" in end:reasons.append(side+": explicit time law")
        preset=str(bc.get(side,"")).upper()
        if (str(bc.get("registry_version","BC_REGISTRY_V1")).upper()=="BC_REGISTRY_V1"
            and theory.startswith("CUF_") and preset=="SIMPLY_SUPPORTED"):
            preset="PINNED_S4"
        if support!=preset:reasons.append(side+": variational support differs from active classical preset")
        # CUF S3 applies to tau=0 only, whereas the reference assembler
        # fixes the three zero-order fields at every tau. Never equate them.
        if theory.startswith("CUF_") and support=="SIMPLY_SUPPORTED":
            reasons.append(side+": CUF S3 thickness-mode constraints differ")
    for site in ELECTRIC_SITES:
        obj=electric.get(site,{})
        mode=str(obj.get("mode","INSULATED")).upper()
        if mode=="INSULATED" and ("value" in obj or "time_law" in obj):
            raise ValueError(site+": INSULATED must not specify charge/voltage/time law")
        if mode!="INSULATED":reasons.append(site+": electrical "+mode)
    if bc.get("advanced") is not None:reasons.append("classical INDIVIDUAL boundary")
    if kind=="DYNAMIC":reasons.append("DYNAMIC global equations pending")
    if scale=="MICRO":reasons.append("MICRO G3D global equations pending")
    reasons=list(dict.fromkeys(reasons))
    return {"status":REFERENCE_STATUS if reasons else CLASSICAL_STATUS,
            "theory":theory,"calculation_kind":kind,
            "electrical_enabled":electrical,"gradient_enabled":scale=="MICRO",
            "reasons":reasons}
