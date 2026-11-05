"""Exact independent analytic CUF TE1..4 inertia-moment benchmarks.

No physical shell simulations or MKR/MVI comparison as an oracle.
The reference is an analytic thickness integral on a cylinder:
  rho * integral_{-h/2}^{h/2} (2*z/h)^(p+q) * (1+z/R) dz
which depends on the first curvature moment for odd p+q.
Verifies each independently assembled M matrix, including all
three translation families and absence of TE derivative inertia.
"""
from __future__ import annotations

import numpy as np
import pytest

from independent_mkr.linear_models import StrongModelCase,IndependentStrongModel
from independent_mvi.linear_models import (
    VariationalModelCase,IndependentVariationalModel)


def _analytic_moment(p,q,h,R,rho):
    exponent=p+q
    def average_power(n):
        return (1./(n+1)) if n%2==0 else 0.
    return rho*h*(
        average_power(exponent)+h/(2*R)*average_power(exponent+1))


@pytest.mark.parametrize("theory",("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("method",("MKR","MVI"))
def test_cuf_all_exact_translation_mass_moments(method,theory):
    if method=="MKR":
        c=StrongModelCase(theory=theory)
        model=IndependentStrongModel(c)
    else:
        c=VariationalModelCase(theory=theory)
        model=IndependentVariationalModel(c,rank=1)
    order=int(theory[-1])+1
    mass=model.mass
    for p in range(order):
        for q in range(order):
            expected=_analytic_moment(p,q,c.thickness_m,
                                      c.radius_m,c.density_kg_m3)
            for comp in range(3):
                a=comp*order+p
                b=comp*order+q
                np.testing.assert_allclose(
                    mass[0,0,a,b],expected,rtol=2.e-13,atol=2.e-12)
                # No spurious mass mixing for distinct physical velocity
                # components in purely polynomial CUF.
                for other in range(3):
                    if other!=comp:
                        assert mass[0,0,a,other*order+q]==0.
    assert np.max(np.abs(mass[1:]))==0., (
        "CUF_VELOCITY_DERIVATIVE_SHOULD_NOT_APPEAR",theory,method)
    assert np.max(np.abs(mass[:,1:]))==0.


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_cylinder_curvature_makes_odd_cuf_thickness_moments_nonzero(method):
    c=(StrongModelCase(theory="CUF_TE4") if method=="MKR" else
       VariationalModelCase(theory="CUF_TE4"))
    solver=(IndependentStrongModel(c) if method=="MKR" else
            IndependentVariationalModel(c,rank=1))
    expected=c.density_kg_m3*c.thickness_m**2/(6*c.radius_m)
    np.testing.assert_allclose(
        solver.mass[0,0,0,1],expected,rtol=2.e-13,atol=2.e-12)
    assert expected>0.
