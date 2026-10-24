#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 regression: unit system is operational, not a report-only label.

The mechanics remains coherent-unit mathematics.  V5 acts at the CASE/GUI
boundary:
- units.system is explicit and mandatory;
- exact conversions between supported systems are defined centrally;
- unit-bearing CASE scalars are interpreted in the declared system and exposed
  as SI diagnostics;
- conspicuous Pa/MPa or m/mm slips produce explicit non-blocking warnings,
  without forbidding unusual research materials/geometries.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

import final_units as fu
import cylinder_final_stage11a_dispatcher as disp

BASE=Path(__file__).resolve().parent


def _base_case(system: str):
    if system == "N_MM_MPA":
        L,R,h,E,q=100.0,50.0,1.0,70000.0,1.0
    elif system == "SI_N_M_PA":
        L,R,h,E,q=0.1,0.05,0.001,70.0e9,1.0e6
    else:
        raise ValueError(system)
    return {
        "theory":"CUF","cuf_model":"TE1","method":"FDM","mode":"LINEAR_ELASTIC",
        "units":{"system":system},
        "geometry":{"L":L,"R":R,"h":h},
        "material":{"E0":E,"nu0":0.3,"source":"USER","name":None},
        "fdm_grid":{"nx":7,"ns":7,"stencil":7},
        "mvi_grid":{"nx":7,"ns":7,"stencil":7},
        "quadrature":{"type":"GAUSS","n_gauss":3},
        "boundary_conditions":{"left":"CLAMPED","right":"CLAMPED","advanced":None},
        "load_definition":{
            "type":"PATCH_RADIAL_FACE","q":q,"face":"OUTER",
            "x1":0.2*L,"x2":0.8*L,
            "circumferential_coordinate":"S",
            "c1":0.1*R,"c2":0.7*R,
            "full_circumference":False,
            "load_factors":[1.0],
        },
        "linear":{"residual_tolerance_relative":1.0e-10},
        "output_dir":"OUT",
    }


def _validated(cfg):
    with tempfile.TemporaryDirectory(prefix="v5_case_") as td:
        p=Path(td)/"case.json"
        p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        return disp.validate_final_case(p)


def _checked_si(rep,key):
    return float(rep["unit_validation"]["checked"][key]["si"])


def run_checks(verbose=True):
    # Exact base conversions.
    conv={
        "1000mm_to_m":fu.convert_value(1000.0,"length","N_MM_MPA","SI_N_M_PA"),
        "210000MPa_to_Pa":fu.convert_value(210000.0,"stress","N_MM_MPA","SI_N_M_PA"),
        "1MPa_to_Pa":fu.convert_value(1.0,"pressure","N_MM_MPA","SI_N_M_PA"),
        "1N_per_mm_to_N_per_m":fu.convert_value(1.0,"line_load","N_MM_MPA","SI_N_M_PA"),
    }
    expected={
        "1000mm_to_m":1.0,
        "210000MPa_to_Pa":210000.0e6,
        "1MPa_to_Pa":1.0e6,
        "1N_per_mm_to_N_per_m":1000.0,
    }
    for k,v in conv.items():
        if not np.isclose(v,expected[k],rtol=0.0,atol=max(1e-15,abs(expected[k])*1e-15)):
            raise AssertionError(f"V5 conversion {k}: {v} != {expected[k]}")
    # Round trip must be exact to floating precision.
    for kind,val in (("length",123.456),("stress",98765.0),("pressure",2.5),("line_load",17.0)):
        si=fu.convert_value(val,kind,"N_MM_MPA","SI_N_M_PA")
        back=fu.convert_value(si,kind,"SI_N_M_PA","N_MM_MPA")
        if not np.isclose(back,val,rtol=2e-15,atol=0.0):
            raise AssertionError(f"V5 round trip failed for {kind}: {back} vs {val}")

    mm=_validated(_base_case("N_MM_MPA"))
    si=_validated(_base_case("SI_N_M_PA"))
    if not mm.get("valid") or not si.get("valid"):
        raise AssertionError("equivalent V5 cases did not validate")
    if mm["units"]["system"]!="N_MM_MPA" or si["units"]["system"]!="SI_N_M_PA":
        raise AssertionError("declared unit systems were not propagated")

    # Equivalent physical cases must produce the same SI diagnostics.
    keys=(
        "geometry.L","geometry.R","geometry.h","material.E0",
        "load_definition.q","load_definition.x1","load_definition.x2",
        "load_definition.c1","load_definition.c2",
    )
    equivalence={}
    for key in keys:
        a=_checked_si(mm,key); b=_checked_si(si,key)
        rel=abs(a-b)/max(abs(a),abs(b),1e-300)
        if rel>2e-14:
            raise AssertionError(f"V5 SI interpretation mismatch {key}: {a} vs {b}, rel={rel:.3e}")
        equivalence[key]={"mm_case_si":a,"si_case_si":b,"rel_difference":rel}

    # Exact audit failure mode: Pa-scale E accidentally entered while MPa is declared.
    # Per the audit this is a WARNING, not a hard error: unusual materials remain legal.
    bad=copy.deepcopy(_base_case("N_MM_MPA"))
    bad["material"]["E0"]=2.1e11
    bad_rep=_validated(bad)
    bad_warn=list(bad_rep.get("unit_validation",{}).get("warnings",[]))
    if not bad_warn or not any("E0=" in x and "нетипично" in x for x in bad_warn):
        raise AssertionError(f"V5 failed to warn on E=2.1e11 declared as MPa: {bad_warn}")

    # No implicit/default unit guessing is allowed in final CASEs.
    missing=copy.deepcopy(_base_case("N_MM_MPA")); missing.pop("units",None)
    try:
        _validated(missing)
    except disp.FinalCaseError as exc:
        if "CASE.units" not in str(exc):
            raise AssertionError(f"missing units rejected unclearly: {exc}") from exc
        missing_rejected=str(exc)
    else:
        raise AssertionError("V5 accepted CASE without explicit units.system")

    rep={
        "status":"PASS",
        "supported_systems":list(fu.SYSTEMS),
        "exact_conversions":conv,
        "equivalent_case_si_diagnostics":equivalence,
        "audit_bad_example_warning":bad_warn,
        "missing_units_rejected":missing_rejected,
        "policy":"coherent mechanics + explicit declared-unit interpretation + non-blocking plausibility warnings; no silent guessing",
    }
    if verbose:
        print("V5_SELF_CHECK: PASS")
        print("  exact conversion: 1000 mm = 1 m; 210000 MPa = 2.1e11 Pa")
        print("  equivalent N-mm-MPa / SI CASE diagnostics: PASS")
        print("  E0=2.1e11 with N_MM_MPA: WARNING emitted, calculation remains legal")
        print("  CASE without units.system: REJECTED")
    return rep


if __name__=="__main__":
    rep=run_checks(True)
    (BASE/"V5_VERIFICATION_RESULTS.json").write_text(json.dumps(rep,indent=2,sort_keys=True),encoding="utf-8")
    print("V5_REPORT_JSON="+json.dumps(rep,sort_keys=True))
