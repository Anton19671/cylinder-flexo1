"""Conservative applicability diagnostic for full-3D StVK shell kinematics.

This is not a constitutive correction. In the flat, linear, pure-bending
limit, an E_zz-constrained model retains lambda+2G instead of the
plane-stress modulus E/(1-nu**2). In a G3D/Gz coupled energy an effective
plane-stress reduction cannot be obtained by silently replacing lambda.
"""
from __future__ import annotations

import math

_LOW_ORDER = frozenset(("KL", "FSDT", "REDDY", "CUF_TE1"))
_ALL = _LOW_ORDER | frozenset(("CUF_TE2", "CUF_TE3", "CUF_TE4"))


def thickness_locking_diagnostic(theory, poisson):
    """Return a report-only warning; never alter physical field equations."""
    name = str(theory).upper()
    if name not in _ALL:
        raise ValueError("unsupported shell theory for thickness-locking diagnostic")
    if isinstance(poisson, bool):
        raise ValueError("Poisson ratio must be real")
    nu = float(poisson)
    if not math.isfinite(nu) or not -1. < nu < .5:
        raise ValueError("Poisson ratio must satisfy -1 < nu < 0.5")
    restricted = name in _LOW_ORDER
    ratio = ((1. - nu)**2 / (1. - 2.*nu)) if restricted else None
    return {
        "constitutive_law": "FULL_3D_GREEN_LAGRANGE_STVK",
        "plane_stress_reduction_applied": False,
        "thickness_locking": "YES" if restricted else "NOT_ESTABLISHED",
        "flat_limit_pure_bending_energy_ratio": ratio,
        "assessment": ("RESTRICTED_THICKNESS_STRAIN_POSSIBLE_POISSON_LOCKING"
                       if restricted else "NOT_CERTIFIED_BY_LOW_ORDER_DIAGNOSTIC"),
        "warning": (
            "Low-order shell kinematics restrict E_zz: the unmodified 3D "
            "constitutive law can overestimate bending stiffness. The "
            "reported flat-limit ratio is not a correction factor for "
            "coupled G3D/Gz flexoelectric or finite-strain calculations."
            if restricted else
            "A higher-order thickness expansion does not itself establish "
            "mesh, thickness-order, or coupled-physics convergence."
        ),
        "quantitative_bending_validation_complete": False,
    }


__all__ = ("thickness_locking_diagnostic",)
