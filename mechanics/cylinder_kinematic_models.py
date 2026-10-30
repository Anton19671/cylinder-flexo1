"""Common shell strain measures and their consistent first variations.

Every theory passes its own deformation-gradient jets, evaluated in the
orthonormal material frame of the cylindrical reference configuration.
Only the strain measure changes; no shell displacement ansatz is replaced.

LINEAR: symmetric part of displacement gradient.
VON_KARMAN: small strain plus quadratic *tangential* membrane terms from
radial slopes (w_,x and cylindrical w_,s-v/R, with the exact H(z) metric).
GREEN_LAGRANGE: full St Venant--Kirchhoff reference strain.

Engineering ordering: (E11,E22,E33,2E12,2E13,2E23).
"""
from __future__ import annotations

import numpy as np

KINEMATIC_MODELS = ("LINEAR", "VON_KARMAN", "GREEN_LAGRANGE")
_EYE = np.eye(3)


def normalized_model(model):
    value = str(model).upper().strip()
    if value not in KINEMATIC_MODELS:
        raise ValueError("geometry_model must be LINEAR, VON_KARMAN or GREEN_LAGRANGE")
    return value


def _sym(a):
    return .5 * (a + a.T)


def _engineering(e):
    return np.asarray((e[0, 0], e[1, 1], e[2, 2],
                       2 * e[0, 1], 2 * e[0, 2], 2 * e[1, 2]), float)


def _radial_tangential_slope(a):
    """Radial displacement gradient along x and s, excluding thickness slope."""
    return np.array((a[2, 0], a[2, 1], 0.0), float)


def strain_jets(f, fx, fs, fz, model):
    """Return engineering strain and three coordinate derivatives."""
    model = normalized_model(model)
    a = np.asarray(f, float) - _EYE
    derivatives = tuple(np.asarray(x, float) for x in (fx, fs, fz))
    if model == "GREEN_LAGRANGE":
        e = _sym(a) + .5 * (a.T @ a)
        gradient = [_sym(d) + _sym(a.T @ d) for d in derivatives]
    elif model == "LINEAR":
        e = _sym(a)
        gradient = [_sym(d) for d in derivatives]
    else:
        slope = _radial_tangential_slope(a)
        e = _sym(a) + .5 * np.outer(slope, slope)
        gradient = []
        for d in derivatives:
            dslope = _radial_tangential_slope(d)
            gradient.append(_sym(d) + _sym(np.outer(slope, dslope)))
    return _engineering(e), [_engineering(d) for d in gradient]


def virtual_strain_jets(f, fx, fs, fz, df, dfx, dfs, dfz, model):
    """Consistent Gateaux derivative of strain_jets, including gradient terms."""
    model = normalized_model(model)
    a = np.asarray(f, float) - _EYE
    da = np.asarray(df, float)
    derivatives = tuple(np.asarray(x, float) for x in (fx, fs, fz))
    variations = tuple(np.asarray(x, float) for x in (dfx, dfs, dfz))
    if model == "GREEN_LAGRANGE":
        de = _sym(da) + _sym(a.T @ da)
        gradients = [_sym(dd) + _sym(da.T @ d + a.T @ dd)
                     for d, dd in zip(derivatives, variations)]
    elif model == "LINEAR":
        de = _sym(da)
        gradients = [_sym(dd) for dd in variations]
    else:
        slope = _radial_tangential_slope(a)
        dslope = _radial_tangential_slope(da)
        de = _sym(da) + _sym(np.outer(slope, dslope))
        gradients = []
        for d, dd in zip(derivatives, variations):
            grad_slope = _radial_tangential_slope(d)
            grad_dslope = _radial_tangential_slope(dd)
            gradients.append(
                _sym(dd) +
                _sym(np.outer(dslope, grad_slope) +
                     np.outer(slope, grad_dslope)))
    return _engineering(de), [_engineering(d) for d in gradients]


__all__ = ("KINEMATIC_MODELS", "normalized_model", "strain_jets",
           "virtual_strain_jets")
