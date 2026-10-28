"""Operator-level high-frequency diagnostics (audit V-3).

The Rayleigh ratios below are operator tests, NOT shell frequencies,
physical stress convergence, or a mesh convergence certificate.
"""
from __future__ import annotations
import numpy as np


def spatial_operator_diagnostics(operators, scheme):
    nx,ns=operators.shape
    Lx=float(np.sum(operators.wx)); Ls=float(np.sum(operators.ws))
    if nx<4 or ns<4 or Lx<=0 or Ls<=0:
        raise ValueError("invalid surface grid for spectrum diagnostics")
    ax=(-1.)**np.arange(nx)
    frequency_x=np.pi*(nx-1)/Lx
    bending_x=operators.Dx@(operators.Dx@ax)
    axial_ratio=float(np.sum(operators.wx*np.abs(bending_x)**2)/
                      (frequency_x**4*np.sum(operators.wx*ax*ax)))
    k=ns//2 if ns%2==0 else (ns-1)//2
    harmonic=np.exp(2j*np.pi*k*np.arange(ns)/ns)
    frequency_s=2*np.pi*k/Ls
    bending_s=operators.Ds@(operators.Ds@harmonic)
    periodic_ratio=float(np.sum(operators.ws*np.abs(bending_s)**2)/
                         (frequency_s**4*np.sum(operators.ws*np.abs(harmonic)**2)))
    return {
        "operator_scheme":str(scheme).upper(),
        "nx":int(nx),"ns":int(ns),
        "axial_alternating_mode_bending_indicator":axial_ratio,
        "circumferential_highest_mode_bending_indicator":periodic_ratio,
        "circumferential_highest_tested_harmonic":int(k),
        "axial_high_frequency_warning":bool(axial_ratio<.2),
        "circumferential_high_frequency_warning":bool(periodic_ratio<.2),
        "meaning":"operator-only; no shell frequency or stress convergence claim",
        "mesh_convergence_verified":False,
        "frequency_convergence_verified":False,
    }


__all__=("spatial_operator_diagnostics",)
