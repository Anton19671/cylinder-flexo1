"""MKR's own 2D conservative face-flux derivatives, algorithm section 40.

Algebraic energy/difference checks ONLY: no shell run or time step.
The MVI method is not imported and none of its spatial matrices are used.
"""
from math import pi
import numpy as np
import pytest

from independent_mkr.linear_models import (
    MODELS, ORDERS, StrongModelCase, IndependentStrongModel,
    _mkr_compatible_cell_operators,
)


def test_even_periodic_checkerboard_is_not_a_zero_gradient():
    """A two-step central first derivative would falsely annul this mode."""
    _, s=_mkr_compatible_cell_operators(9,10,2.0,1.0)
    checker=(-1.0)**np.arange(10)
    assert np.linalg.norm(s[1]@checker)>0.0
    # Direct second derivative is a one-cell Laplacian, not (central D)^2.
    dss=np.zeros((10,10))
    for j in range(10):
        dss[j,j]=-2
        dss[j,(j-1)%10]+=1
        dss[j,(j+1)%10]+=1
    step=2*pi/10
    dss/=step**2
    np.testing.assert_allclose(s[2],s[0]@dss,rtol=0.,atol=1.e-13)


@pytest.mark.parametrize("theory",MODELS)
def test_all_seven_strong_mkr_models_use_independent_collocated_faces(theory):
    model=IndependentStrongModel(StrongModelCase(theory=theory))
    c=model.case
    assert model.deriv_x[0].shape==(c.nx-1,c.nx)
    assert model.deriv_s[0].shape==(c.ns,c.ns)
    q=np.zeros((model.n,c.nx,c.ns))
    q[0]=np.linspace(0,c.length_m,c.nx)[:,None]
    xgrad=model._differentiate(q,1)
    np.testing.assert_allclose(xgrad[0],1.,rtol=0.,atol=3.e-15)
    constant=np.ones_like(q)
    for order in range(1,len(ORDERS)):
        assert np.max(np.abs(model._differentiate(constant,order)))<1.e-9


@pytest.mark.parametrize("theory",MODELS)
def test_own_mkr_discrete_energy_equals_nodal_flux_work(theory):
    """Cell energy derivative is conservative and algebraically symmetric."""
    model=IndependentStrongModel(StrongModelCase(theory=theory))
    rng=np.random.default_rng(712)
    x=model.unpack(rng.standard_normal(sum(model.dofs))*1.e-7)
    y=model.unpack(rng.standard_normal(sum(model.dofs))*1.e-7)
    for tensor,terms,action in (
        (model.stiffness,6,model.stiffness_action),
        (model.mass,3,model.mass_action)):
        fields_x=[model._differentiate(x,k) for k in range(terms)]
        fields_y=[model._differentiate(y,k) for k in range(terms)]
        expected=sum(
            np.sum(fields_x[i]*
                   np.einsum("ab,bxs->axs",tensor[i,j],fields_y[j],
                             optimize=True))
            for i in range(terms) for j in range(terms))
        actual=float(np.sum(x*action(y)))
        np.testing.assert_allclose(actual,expected,rtol=5.e-11,atol=1.e-6)
        reverse=float(np.sum(y*action(x)))
        np.testing.assert_allclose(actual,reverse,rtol=5.e-11,atol=1.e-6)
        assert float(np.sum(x*action(x)))>0.
