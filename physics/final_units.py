#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit unit policy for the final cylinder-solver interface (V5).

The mechanics kernels remain coherent-unit mathematics and may operate in either
supported coherent system.  This module makes the selected system operational at
the user/dispatcher boundary by providing exact conversions and *non-blocking*
plausibility diagnostics, as required by the audit.

Important: plausibility is a warning, not a material restriction.  Unusual
materials/geometries remain legal.  No unit system is guessed silently.
"""
from __future__ import annotations
import math

SYSTEMS=("N_MM_MPA","SI_N_M_PA")

UNIT_LABELS={
    "N_MM_MPA": {
        "length":"mm", "displacement":"mm", "stress":"MPa",
        "pressure":"N/mm²", "line_load":"N/mm", "force":"N",
    },
    "SI_N_M_PA": {
        "length":"m", "displacement":"m", "stress":"Pa",
        "pressure":"Pa", "line_load":"N/m", "force":"N",
    },
}

_TO_SI={
    "N_MM_MPA": {
        "length":1.0e-3,       # mm -> m
        "stress":1.0e6,        # MPa -> Pa
        "pressure":1.0e6,      # N/mm^2 -> Pa
        "line_load":1.0e3,     # N/mm -> N/m
        "force":1.0,
    },
    "SI_N_M_PA": {
        "length":1.0,
        "stress":1.0,
        "pressure":1.0,
        "line_load":1.0,
        "force":1.0,
    },
}

class UnitPolicyError(ValueError):
    pass


def normalize_system(system:str)->str:
    s=str(system).upper().strip()
    if s not in SYSTEMS:
        raise UnitPolicyError(f"unit system must be one of {SYSTEMS}; got {system!r}")
    return s


def unit_labels(system:str)->dict:
    """Return user-facing labels for the selected coherent unit system."""
    return dict(UNIT_LABELS[normalize_system(system)])


def _plain_load_definition(load_definition):
    """Unwrap dispatcher metadata without guessing an external load scale."""
    if not isinstance(load_definition,dict):
        return {}
    definition=load_definition.get("definition")
    return definition if isinstance(definition,dict) else load_definition


def load_axis(factors,load_definition,system:str):
    """Return dimensional load values, axis label and machine-readable key.

    For built-in pressure and line-load generators the displayed ordinate is
    ``load factor * nominal amplitude``.  An arbitrary external NPZ has no
    generally meaningful scalar amplitude, so only there lambda is retained.
    """
    labels=unit_labels(system)
    spec=_plain_load_definition(load_definition)
    typ=str(spec.get("type","")).upper()
    vals=[float(v) for v in factors]
    if typ in ("UNIFORM_RADIAL_FACE","HARMONIC_RADIAL_FACE","PATCH_RADIAL_FACE"):
        try: amplitude=float(spec["q"])
        except Exception: pass
        else:
            return [abs(amplitude)*v for v in vals],f"q [{labels['pressure']}]","q"
    if typ=="LINE_RADIAL_GENERATOR":
        try: amplitude=float(spec["p_line"])
        except Exception: pass
        else:
            return [abs(amplitude)*v for v in vals],f"p_line [{labels['line_load']}]","p_line"
    if typ=="POINT_RADIAL_FORCE":
        try: amplitude=float(spec["P"])
        except Exception: pass
        else:
            return [abs(amplitude)*v for v in vals],f"P [{labels['force']}]","P"
    return vals,"lambda [-]","load_factor"


def conversion_factor(kind:str, source:str, target:str)->float:
    """Multiply a numeric value by this factor to convert source -> target."""
    source=normalize_system(source); target=normalize_system(target)
    if kind not in _TO_SI[source]:
        raise UnitPolicyError(f"unsupported dimensional kind: {kind}")
    return _TO_SI[source][kind]/_TO_SI[target][kind]


def convert_value(value,kind:str,source:str,target:str):
    return float(value)*conversion_factor(kind,source,target)


def to_si(value,kind:str,system:str)->float:
    s=normalize_system(system)
    if kind not in _TO_SI[s]:
        raise UnitPolicyError(f"unsupported dimensional kind: {kind}")
    return float(value)*_TO_SI[s][kind]


def _finite_number(v,name):
    try: x=float(v)
    except Exception as exc: raise UnitPolicyError(f"{name}: expected a number") from exc
    if not math.isfinite(x): raise UnitPolicyError(f"{name}: expected a finite number")
    return x


def plausibility_warnings(system:str, *, L=None, R=None, h=None, E0=None, load_max=None):
    """Return non-blocking V5 warnings for conspicuous scale mismatches.

    These are deliberately broad engineering plausibility ranges copied from
    the audit recommendation.  They must never be used as constitutive/model
    admissibility conditions.
    """
    s=normalize_system(system)
    warn=[]
    if E0 is not None:
        e=_finite_number(E0,"E0")
        if s=="N_MM_MPA":
            if e < 1.0e3 or e > 1.0e7:
                warn.append(
                    f"V5: E0={e:.4g} нетипично для MPa "
                    f"(ориентиры: сталь ~2.1e5, алюминий ~7.0e4, полимеры ~1e3). "
                    "Проверьте units.system; это предупреждение, не запрет."
                )
        else:
            if e < 1.0e8 or e > 1.0e13:
                warn.append(
                    f"V5: E0={e:.4g} нетипично для Pa (сталь ~2.1e11). "
                    "Проверьте units.system; это предупреждение, не запрет."
                )
    if h is not None:
        hh=_finite_number(h,"h")
        if s=="N_MM_MPA":
            if hh < 1.0e-3 or hh > 1.0e3:
                warn.append(
                    f"V5: h={hh:.4g} нетипично для mm. Проверьте units.system; "
                    "это предупреждение, не запрет."
                )
        else:
            if hh < 1.0e-6 or hh > 1.0:
                warn.append(
                    f"V5: h={hh:.4g} нетипично для m. Проверьте units.system; "
                    "это предупреждение, не запрет."
                )
    if h is not None and R is not None:
        hh=_finite_number(h,"h"); rr=_finite_number(R,"R")
        if rr>0.0 and hh/rr>0.2:
            warn.append(
                f"V5: h/R={hh/rr:.3g}; оболочка не тонкая, TE1/KL могут быть недостаточны. "
                "Это модельное предупреждение, запуск не блокируется."
            )
    # load_max is accepted for API compatibility with the audit sketch.  No
    # universal physical range is imposed because arbitrary research loads are
    # valid and their scale depends strongly on geometry/material.
    if load_max is not None:
        _finite_number(load_max,"load_max")
    return warn


def _diag(value,kind:str,system:str,name:str):
    x=_finite_number(value,name)
    return {"declared":x,"si":to_si(x,kind,system),"kind":kind}


def validate_case_scalars(cfg:dict,system:str)->dict:
    """Interpret unit-bearing CASE scalars and return SI diagnostics + warnings.

    No plausibility warning is promoted to an exception.  Structural input
    validity (positive geometry/E, etc.) remains the responsibility of the
    normal CASE parser.
    """
    s=normalize_system(system)
    out={"system":s,"checked":{},"warnings":[]}
    g=cfg.get("geometry") or {}
    for key in ("L","R","h"):
        if key in g and g[key] is not None:
            out["checked"][f"geometry.{key}"]=_diag(g[key],"length",s,f"geometry.{key}")
    m=cfg.get("material") or {}
    if m.get("E0") is not None:
        out["checked"]["material.E0"]=_diag(m["E0"],"stress",s,"material.E0")

    ld=cfg.get("load_definition")
    if isinstance(ld,dict):
        for key in ("x1","x2","band_width"):
            if ld.get(key) is not None:
                out["checked"][f"load_definition.{key}"]=_diag(
                    ld[key],"length",s,f"load_definition.{key}")
        if str(ld.get("circumferential_coordinate","THETA")).upper()=="S":
            for key in ("c1","c2","c0"):
                if ld.get(key) is not None:
                    out["checked"][f"load_definition.{key}"]=_diag(
                        ld[key],"length",s,f"load_definition.{key}")
        if ld.get("q") is not None:
            out["checked"]["load_definition.q"]=_diag(
                ld["q"],"pressure",s,"load_definition.q")
        if ld.get("p_line") is not None:
            out["checked"]["load_definition.p_line"]=_diag(
                ld["p_line"],"line_load",s,"load_definition.p_line")

    out["warnings"]=plausibility_warnings(
        s,
        L=g.get("L"),R=g.get("R"),h=g.get("h"),E0=m.get("E0"),
        load_max=(ld.get("q") if isinstance(ld,dict) and ld.get("q") is not None else None),
    )
    return out
