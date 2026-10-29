#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-aware end-boundary-condition registry for the final cylinder solver.

The registry deliberately separates the *physical support name* selected by the
user from its mathematical realization in a specific kinematic theory.

KL generalized boundary pairs at x=const:
    (u, T_u), (v, T_v), (w, T_w), (theta_n, T_theta)

CUF TE(N) generalized boundary pairs for every tau=0..N:
    (u_tau, T_u_tau), (v_tau, T_v_tau), (w_tau, T_w_tau)

Kinds:
    GEOMETRIC : q = prescribed value
    NATURAL   : T = prescribed traction/resultant
    ELASTIC   : T + k (q-q0) = prescribed traction
"""
from __future__ import annotations

MODEL_THEORIES=("KL","CUF")
STANDARD_PRESETS=(
    "CLAMPED",
    "SIMPLY_SUPPORTED",
    "FREE",
    "SYMMETRY",
    "GUIDED",
    "ELASTIC_ENGINEERING",
)

# Historical pinned-support codes remain accepted by the dispatcher.  The KL
# GUI intentionally exposes S1/S2/S4 and identifies the canonical simple
# support as S3.  CUF keeps its canonical modern simple support and does not
# expose its legacy S3 alias because that alias is identical to CLAMPED in a
# displacement-only CUF model.
LEGACY_PRESETS=("PINNED_S1","PINNED_S2","PINNED_S3","PINNED_S4")
BC_REGISTRY_VERSION="BC_REGISTRY_V2"


def model_tag(theory,cuf_model=None):
    t=str(theory).upper().strip()
    if t=="KL":
        return "KL"
    if t=="CUF":
        m=str(cuf_model or "TE1").upper().strip()
        if m not in ("TE1","TE2","TE3","TE4"):
            raise ValueError(f"unsupported CUF model: {m}")
        return f"CUF_{m}"
    raise ValueError(f"unsupported theory: {theory}")


def cuf_m(cuf_model):
    m=str(cuf_model).upper().strip()
    if m not in ("TE1","TE2","TE3","TE4"):
        raise ValueError(f"unsupported CUF model: {m}")
    return int(m[2:])+1


def normalize_preset(code,theory):
    p=str(code).upper().strip()
    t=str(theory).upper().strip()
    if p not in STANDARD_PRESETS+LEGACY_PRESETS:
        raise ValueError(f"unsupported boundary preset: {p}")
    # SIMPLY_SUPPORTED has one physical meaning in both theories: the
    # immovable S3 hinge on the reference/midsurface.  CUF applies the
    # essential conditions only to tau=0; higher thickness coefficients are
    # traction-free and are resolved by cuf_kinds_tau().
    if p=="SIMPLY_SUPPORTED":
        return "PINNED_S3" if t=="KL" else "SIMPLY_SUPPORTED"
    return p


def kl_kinds(code):
    p=normalize_preset(code,"KL")
    patterns={
        # u, v, w, theta_n
        "CLAMPED": ("GEOMETRIC","GEOMETRIC","GEOMETRIC","GEOMETRIC"),
        "FREE": ("NATURAL","NATURAL","NATURAL","NATURAL"),
        "PINNED_S1": ("NATURAL","GEOMETRIC","GEOMETRIC","NATURAL"),
        "PINNED_S2": ("GEOMETRIC","NATURAL","GEOMETRIC","NATURAL"),
        "PINNED_S3": ("GEOMETRIC","GEOMETRIC","GEOMETRIC","NATURAL"),
        "PINNED_S4": ("NATURAL","NATURAL","GEOMETRIC","NATURAL"),
        # Symmetry plane x=const: u=0 and theta_n=0 are essential; tangential
        # and transverse generalized tractions vanish.
        "SYMMETRY": ("GEOMETRIC","NATURAL","NATURAL","GEOMETRIC"),
        # Axially sliding guided clamp: axial traction-free, while v,w and
        # normal rotation are restrained.
        "GUIDED": ("NATURAL","GEOMETRIC","GEOMETRIC","GEOMETRIC"),
        "ELASTIC_ENGINEERING": ("ELASTIC","ELASTIC","ELASTIC","ELASTIC"),
    }
    return patterns[p]


def cuf_kinds(code):
    """Return CUF boundary kinds for thickness-independent presets.

    For SIMPLY_SUPPORTED use cuf_kinds_tau(): its physical S3 conditions are
    intentionally different for tau=0 and tau>=1.
    """
    p=normalize_preset(code,"CUF")
    patterns={
        # u_tau, v_tau, w_tau
        "CLAMPED": ("GEOMETRIC","GEOMETRIC","GEOMETRIC"),
        "FREE": ("NATURAL","NATURAL","NATURAL"),
        "PINNED_S1": ("NATURAL","GEOMETRIC","GEOMETRIC"),
        "PINNED_S2": ("GEOMETRIC","NATURAL","GEOMETRIC"),
        # Legacy CUF S3 is retained for existing CASE files.
        "PINNED_S3": ("GEOMETRIC","GEOMETRIC","GEOMETRIC"),
        # Movable in-plane hinge; keep it explicit and separate from S3.
        "PINNED_S4": ("NATURAL","NATURAL","GEOMETRIC"),
        "SYMMETRY": ("GEOMETRIC","NATURAL","NATURAL"),
        "GUIDED": ("NATURAL","GEOMETRIC","GEOMETRIC"),
        "ELASTIC_ENGINEERING": ("ELASTIC","ELASTIC","ELASTIC"),
    }
    if p=="SIMPLY_SUPPORTED":
        raise ValueError(
            "CUF SIMPLY_SUPPORTED depends on thickness index tau; "
            "call cuf_kinds_tau(code, tau)"
        )
    return patterns[p]


def cuf_kinds_tau(code,tau):
    """Return CUF end-condition kinds for one thickness coefficient."""
    p=normalize_preset(code,"CUF")
    if p=="SIMPLY_SUPPORTED":
        return (
            ("GEOMETRIC","GEOMETRIC","GEOMETRIC")
            if int(tau)==0
            else ("NATURAL","NATURAL","NATURAL")
        )
    return cuf_kinds(p)




def allowed_presets(theory):
    t=str(theory).upper().strip()
    if t not in MODEL_THEORIES:
        raise ValueError(f"unsupported theory: {theory}")
    if t=="KL":
        # SIMPLY_SUPPORTED is the backward-compatible canonical alias of S3.
        # S1/S2/S4 are distinct physically meaningful sliding variants.
        return STANDARD_PRESETS+("PINNED_S1","PINNED_S2","PINNED_S4")
    return STANDARD_PRESETS+("PINNED_S4",)


def resolved_preset_code(code,theory):
    """Return the exact mathematical preset used by the active theory."""
    return normalize_preset(code,theory)


def gui_caption_map(theory,cuf_model=None):
    """Return ordered caption->code mapping for the active kinematic model."""
    t=str(theory).upper().strip()
    if t=="KL":
        return {
            "Выберите…":"",
            "Заделка · KL: u=v=w=theta_n=0":"CLAMPED",
            "Шарнир S1 · осевое скольжение · KL: Tu=0, v=w=0, Mx=0":"PINNED_S1",
            "Шарнир S2 · окружное скольжение · KL: u=0, Tv=0, w=0, Mx=0":"PINNED_S2",
            "Шарнир S3 · неподвижный · KL: u=v=w=0, Mx=0":"SIMPLY_SUPPORTED",
            "Шарнир S4 · подвижный в плоскости · KL: Tu=Tv=0, w=0, Mx=0":"PINNED_S4",
            "Свободный · KL: Tu=Tv=Tw=Ttheta=0":"FREE",
            "Симметрия · KL: u=theta_n=0, Tv=Tw=0":"SYMMETRY",
            "Упругая / инженерная · KL: T+k(q-q0)=Tbar":"ELASTIC_ENGINEERING",
        }
    m=cuf_m(cuf_model or "TE1")
    n=m-1
    return {
        "Выберите…":"",
        f"Заделка · {cuf_model}: uτ=vτ=wτ=0, τ=0…{n}":"CLAMPED",
        f"Шарнир S3 · неподвижный · {cuf_model}: u0=v0=w0=0; Tτ≥1=0":"SIMPLY_SUPPORTED",
        f"Шарнир S4 · подвижный в плоскости · {cuf_model}: Tuτ=Tvτ=0, wτ=0, τ=0…{n}":"PINNED_S4",
        f"Свободный · {cuf_model}: Tuτ=Tvτ=Twτ=0, τ=0…{n}":"FREE",
        f"Симметрия · {cuf_model}: uτ=0, Tvτ=Twτ=0, τ=0…{n}":"SYMMETRY",
        f"Упругая / инженерная · {cuf_model}: Tατ+kατ(qατ-q0)=Tbar":"ELASTIC_ENGINEERING",
    }


def formula_text(theory,cuf_model,code):
    """Compact mathematical description shown below the GUI selectors."""
    p=str(code or "").upper().strip()
    if not p:
        return "не выбрано"
    t=str(theory).upper().strip()
    if t=="KL":
        texts={
            "CLAMPED":"u=0; v=0; w=0; theta_n=0 (на x=const: w,x=0)",
            "SIMPLY_SUPPORTED":"S3 (неподвижный): u=0; v=0; w=0; Mx=0",
            "PINNED_S1":"S1 (осевое скольжение): Tu=0; v=0; w=0; Mx=0",
            "PINNED_S2":"S2 (окружное скольжение): u=0; Tv=0; w=0; Mx=0",
            "PINNED_S3":"S3 (неподвижный): u=0; v=0; w=0; Mx=0",
            "PINNED_S4":"S4 (подвижный в плоскости): Tu=0; Tv=0; w=0; Mx=0",
            "FREE":"Tu=0; Tv=0; Tw=0; Ttheta=0",
            "SYMMETRY":"u=0; theta_n=0; Tv=0; Tw=0",
            "GUIDED":"Tu=0; v=0; w=0; theta_n=0",
            "ELASTIC_ENGINEERING":"T_i + k_i(q_i-q0_i) = Tbar_i, i={u,v,w,theta}",
        }
        return texts[p]
    n=cuf_m(cuf_model)-1
    texts={
        "CLAMPED":f"uτ=vτ=wτ=0, τ=0…{n}",
        "SIMPLY_SUPPORTED":f"S3 (неподвижный): u0=v0=w0=0; Tuτ=Tvτ=Twτ=0, τ=1…{n}",
        "PINNED_S1":f"Tuτ=0; vτ=wτ=0, τ=0…{n}",
        "PINNED_S2":f"uτ=wτ=0; Tvτ=0, τ=0…{n}",
        "PINNED_S3":f"uτ=vτ=wτ=0, τ=0…{n} (legacy alias CLAMPED)",
        "PINNED_S4":f"Tuτ=Tvτ=0; wτ=0, τ=0…{n}",
        "FREE":f"Tuτ=Tvτ=Twτ=0, τ=0…{n}",
        "SYMMETRY":f"uτ=0; Tvτ=Twτ=0, τ=0…{n}",
        "GUIDED":f"Tuτ=0; vτ=wτ=0, τ=0…{n}",
        "ELASTIC_ENGINEERING":f"Tατ+kατ(qατ-q0ατ)=Tbarατ, τ=0…{n}",
    }
    return texts[p]
