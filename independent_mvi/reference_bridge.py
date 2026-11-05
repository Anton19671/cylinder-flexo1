"""Fail-closed adapter from a saved reference CASE to the independent MVI solver.

Physical input transfer only; no solver construction or physical calculation.
The numerical spatial operators are intentionally distinct. Unsupported
physics, boundary conditions or timing are REJECTED, not approximated.
"""
from __future__ import annotations
from copy import deepcopy
from math import isfinite

from independent_mvi.linear_models import VariationalModelCase

SCHEMA="INDEPENDENT_MVI_SEVEN_MODELS_V1"
SUPPORTED=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
SCHEMES=("BATHE","NEWMARK","GAUSS_IRK4")


def _positive(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError("MVI_BRIDGE_"+name+"_INVALID_NUMBER")
    answer=float(value)
    if not isfinite(answer) or answer<=0:
        raise ValueError("MVI_BRIDGE_"+name+"_INVALID_VALUE")
    return answer


def _finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError("MVI_BRIDGE_"+name+"_INVALID_NUMBER")
    answer=float(value)
    if not isfinite(answer):
        raise ValueError("MVI_BRIDGE_"+name+"_NOT_FINITE")
    return answer


def from_reference_case(case):
    """Accept only linear HOOKE/MACRO zero-state harmonic uniformly loaded C-C."""
    if not isinstance(case,dict):
        raise ValueError("MVI_BRIDGE_REQUIRES_SOURCE_OBJECT")
    source=deepcopy(case)
    if source.get("solver_backend")!="VARIATIONAL_REFERENCE":
        raise ValueError("MVI_BRIDGE_REFERENCE_BACKEND_REQUIRED")
    if str(source.get("method","")).upper()!="MVI":
        raise ValueError("MVI_BRIDGE_OTHER_METHOD_REJECTED")
    for key,expected in (
        ("calculation_kind","DYNAMIC"),
        ("geometry_model","LINEAR"),
        ("mode","LINEAR_ELASTIC"),
        ("material_model","HOOKE"),
        ("scale_model","MACRO"),
    ):
        if str(source.get(key,"")).upper()!=expected:
            raise ValueError("MVI_BRIDGE_UNSUPPORTED_"+key.upper())
    if (source.get("units") or {}).get("system")!="SI_N_M_PA":
        raise ValueError("MVI_BRIDGE_REQUIRES_SI_UNITS")
    if (source.get("micro") or {}).get("enable_flexoelectric"):
        raise ValueError("MVI_BRIDGE_FLEXO_NOT_SUPPORTED")

    theory=str(source.get("theory","")).upper()
    if theory=="CUF":
        theory="CUF_"+str(source.get("cuf_model","")).upper()
    if theory not in SUPPORTED:
        raise ValueError("MVI_BRIDGE_UNSUPPORTED_THEORY")

    geometry=source.get("geometry")
    material=source.get("material")
    dynamics=source.get("dynamics")
    grid=source.get("mvi_grid")
    if not all(isinstance(item,dict) for item in
               (geometry,material,dynamics,grid)):
        raise ValueError("MVI_BRIDGE_SOURCE_FIELDS_INCOMPLETE")
    if source.get("fdm_grid") is not None:
        raise ValueError("MVI_BRIDGE_CROSS_MVI_GRID_REJECTED")
    nx,ns=grid.get("nx"),grid.get("ns")
    if type(nx) is not int or type(ns) is not int:
        raise ValueError("MVI_BRIDGE_GRID_NODES_INVALID")
    steps=dynamics.get("n_steps")
    if type(steps) is not int or not 1<=steps<=120:
        raise ValueError("MVI_BRIDGE_TIME_STEPS_INVALID")
    if str(dynamics.get("damping","NONE")).upper()!="NONE":
        raise ValueError("MVI_BRIDGE_DAMPING_UNSUPPORTED")
    for k in ("start_time_s","mass_damping_rate_per_s","eta_s"):
        if _finite(dynamics.get(k,0),k)!=0:
            raise ValueError("MVI_BRIDGE_NONZERO_"+k.upper())
    initial=dynamics.get("initial_conditions") or {"type":"ZERO"}
    if not isinstance(initial,dict) or initial.get("type")!="ZERO":
        raise ValueError("MVI_BRIDGE_NONZERO_INITIAL_STATE")
    if _finite(dynamics.get("newmark_beta",.25),"beta")!=.25:
        raise ValueError("MVI_BRIDGE_NONSTANDARD_NEWMARK_BETA")
    if _finite(dynamics.get("newmark_gamma",.5),"gamma")!=.5:
        raise ValueError("MVI_BRIDGE_NONSTANDARD_NEWMARK_GAMMA")
    scheme=str(dynamics.get("integrator","NEWMARK")).upper()
    if scheme=="GAUSS2_IRK4":
        scheme="GAUSS_IRK4"
    if scheme not in SCHEMES:
        raise ValueError("MVI_BRIDGE_UNSUPPORTED_TIME_MVI")
    law=dynamics.get("time_law")
    if not isinstance(law,dict) or str(law.get("kind","")).upper()!="HARMONIC":
        raise ValueError("MVI_BRIDGE_ONLY_HARMONIC_TIME_LOAD")
    if _finite(law.get("amplitude",1),"amplitude")!=1:
        raise ValueError("MVI_BRIDGE_TIME_AMPLITUDE_UNSUPPORTED")
    if (_finite(law.get("phase_rad",0),"phase")!=0 or
        _finite(law.get("start_time_s",0),"start_time")!=0):
        raise ValueError("MVI_BRIDGE_TIME_OFFSET_UNSUPPORTED")

    load=source.get("load_definition")
    if (not isinstance(load,dict) or
        load.get("type")!="UNIFORM_RADIAL_FACE" or
        load.get("face")!="MIDDLE" or
        load.get("direction")!="RADIAL_INWARD" or
        load.get("kinematics","DEAD_REFERENCE")!="DEAD_REFERENCE" or
        set(load)-{"type","face","direction","q","kinematics"}):
        raise ValueError("MVI_BRIDGE_LOAD_NOT_EQUIVALENT")
    bc=(source.get("boundary_conditions") or {}).get("variational") or {}
    mechanical=bc.get("mechanical") or {}
    if (not isinstance(mechanical,dict) or
        set(mechanical)!={"left","right"} or
        any(not isinstance(mechanical[end],dict) or
            mechanical[end].get("support")!="CLAMPED" or
            mechanical[end].get("higher")!="NATURAL"
            for end in ("left","right"))):
        raise ValueError("MVI_BRIDGE_ONLY_BOTH_ENDS_CLAMPED")
    if bc.get("electrical"):
        raise ValueError("MVI_BRIDGE_NO_ELECTRICAL_TRACE")

    physics={
        "theory":theory,
        "length_m":_positive(geometry.get("L"),"L"),
        "radius_m":_positive(geometry.get("R"),"R"),
        "thickness_m":_positive(geometry.get("h"),"h"),
        "young_pa":_positive(material.get("E0"),"E0"),
        "poisson":_finite(material.get("nu0"),"nu0"),
        "density_kg_m3":_positive(dynamics.get("density_kg_m3"),"rho"),
        "nx":nx,"ns":ns,
        "pressure_pa":_positive(load.get("q"),"pressure"),
        "frequency_hz":_positive(law.get("frequency_hz"),"frequency"),
        "pressure_kind":"UNIFORM",
    }
    VariationalModelCase(**physics)  # input validation only
    rank=source.get("rank")
    if type(rank) is not int or not 1<=rank<=8:
        raise ValueError("MVI_BRIDGE_RANK_OUTSIDE_1_TO_8")
    if rank>min(ns,nx-(4 if theory=="KL" else 2)):
        raise ValueError("MVI_BRIDGE_RANK_EXCEEDS_CLAMP_CAPACITY")
    return {
        "schema":SCHEMA,"physics":physics,
        "time":{"integrator":scheme,
                "dt_s":_positive(dynamics.get("time_step_s"),"dt_s"),
                "steps":steps},
        "rank":rank,
        "output_dir":"output/imported_independent_mvi",
    }


__all__=("from_reference_case",)
