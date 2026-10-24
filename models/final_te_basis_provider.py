#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named Taylor-expansion CUF thickness bases used by the final interface.

Uniform TE-N basis:
    F_tau(z)=z**tau, tau=0..N.

Only TE1..TE4 are exposed by the final GUI.  The general CUF mechanics remains
in cylinder_cuf_stage7a_general_mechanics.py and is not modified here.
"""
from __future__ import annotations
import cylinder_cuf_stage7a_general_mechanics as g7

SUPPORTED_TE_MODELS=("TE1","TE2","TE3","TE4")


def taylor_basis(order:int):
    n=int(order)
    if n < 1 or n > 4:
        raise ValueError("Final interface supports only verified TE1..TE4")
    funcs=[]; dfuncs=[]; names=[]
    for tau in range(n+1):
        funcs.append(lambda z, p=tau: float(z)**p)
        if tau == 0:
            dfuncs.append(lambda z: 0.0)
        else:
            dfuncs.append(lambda z, p=tau: p*(float(z)**(p-1)))
        names.append("F0=1" if tau==0 else ("F1=z" if tau==1 else f"F{tau}=z^{tau}"))
    return g7.CallableThicknessBasis(tuple(funcs),tuple(dfuncs),tuple(names))


def get_basis(model="TE1"):
    model=str(model).upper().strip()
    if model not in SUPPORTED_TE_MODELS:
        raise ValueError(f"model must be one of {SUPPORTED_TE_MODELS}")
    return taylor_basis(int(model[2:]))


def get_te1_basis(): return get_basis("TE1")
def get_te2_basis(): return get_basis("TE2")
def get_te3_basis(): return get_basis("TE3")
def get_te4_basis(): return get_basis("TE4")

# Backward-compatible provider name used by older child cases.
def get_basis_te1(): return get_te1_basis()
