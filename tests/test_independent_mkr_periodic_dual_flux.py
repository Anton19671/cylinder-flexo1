"""Algebra-only verification of independent MKR angular node/face flux.

Tests elementary stencils, even-grid high modes, both discrete
adjoints, full 2D energy first variation and nondegenerate nodal
kinetic work. NO physical shell solve, time step, or MVI import.
"""
from __future__ import annotations
from math import pi
import numpy as np
import pytest

from independent_mkr.linear_models import (
    MODELS,StrongModelCase,IndependentStrongModel,
    _finite_difference_s,_periodic_primal_dual_angular,
)


@pytest.mark.parametrize("n",(8,9,10,11,16))
def test_periodic_one_cell_face_gradient_has_correct_adjoint(n):
    radius=1.4
    C,G,L,F=_periodic_primal_dual_angular(n,radius)
    step=2*pi*radius/n
    one=np.ones(n)
    np.testing.assert_allclose(C@one,one,atol=1.e-14)
    np.testing.assert_allclose(G@one,0.,atol=1.e-13)
    np.testing.assert_allclose(L@one,0.,atol=1.e-13)
    np.testing.assert_allclose(F,C@L,atol=0.,rtol=0.)
    rng=np.random.default_rng(710+n)
    values=rng.normal(size=n)
    flux=rng.normal(size=n)
    np.testing.assert_allclose(
        np.vdot(G@values,flux),np.vdot(values,G.T@flux),
        rtol=4.e-14,atol=2.e-13)
    np.testing.assert_allclose(G@values,
        (np.roll(values,-1)-values)/step,rtol=2.e-14,atol=2.e-14)
    np.testing.assert_allclose(L@values,
        (np.roll(values,-1)-2*values+np.roll(values,1))/step**2,
        rtol=3.e-14,atol=4.e-13)
    if n%2==0:
        checker=(-1.)**np.arange(n)
        np.testing.assert_allclose(C@checker,0.,atol=1.e-14)
        assert np.linalg.norm(G@checker)>1.
        assert np.linalg.norm(L@checker)>1.
        # A squared centered derivative would incorrectly annihilate it.
        center=_finite_difference_s(n,radius)
        np.testing.assert_allclose(center@checker,0.,atol=1.e-14)
        np.testing.assert_allclose(center@(center@checker),0.,atol=1.e-13)


def test_angular_face_kinematics_retains_second_order_smooth_harmonics():
    n=24
    C,G,L,F=_periodic_primal_dual_angular(n,1.)
    h=2*pi/n
    theta=np.arange(n)*h
    smooth=np.sin(theta)
    middle=np.sin(theta+h/2)
    slope=np.cos(theta+h/2)
    np.testing.assert_allclose(C@smooth,middle,rtol=0.,atol=.01)
    np.testing.assert_allclose(G@smooth,slope,rtol=0.,atol=.003)
    np.testing.assert_allclose(F@smooth,-middle,rtol=0.,atol=.016)
    np.testing.assert_allclose(L@smooth,-smooth,rtol=0.,atol=.007)


@pytest.mark.parametrize("theory",MODELS)
def test_mkr_angular_even_grid_stiffness_and_mass_have_no_checkerboard_hole(theory):
    case=StrongModelCase(theory=theory,nx=11,ns=10)
    model=IndependentStrongModel(case)
    checker=(-1.)**np.arange(case.ns)
    radial=(2 if theory in ("KL","FSDT","REDDY")
            else 2*(int(theory[-1])+1))
    tangent=(1 if theory in ("KL","FSDT","REDDY")
             else int(theory[-1])+1)
    for component in (0,tangent,radial):
        q=np.zeros((model.n,case.nx,case.ns))
        q[component,1:-1,:]=checker[None,:]*1.e-7
        kinetic=float(np.sum(q*model.mass_action(q)))
        elastic=float(np.sum(q*model.stiffness_action(q)))
        assert kinetic>0., (theory,component,kinetic)
        assert elastic>0., (theory,component,elastic)
    assert model.deriv_s_face[0].shape==(case.ns,case.ns)
    assert model.deriv_s_face[1].shape==(case.ns,case.ns)


@pytest.mark.parametrize("n",(9,10))
@pytest.mark.parametrize("theory",("KL","REDDY","CUF_TE4"))
def test_method_owned_balanced_discrete_work_exactly_equals_nodal_action(n,theory):
    case=StrongModelCase(theory=theory,nx=11,ns=n)
    model=IndependentStrongModel(case)
    rng=np.random.default_rng(191+n)
    q=model.unpack(rng.standard_normal(sum(model.dofs))*1.e-7)
    p=model.unpack(rng.standard_normal(sum(model.dofs))*1.e-7)
    wx=model.quadrature_x[None,:,None]
    expected=0.
    for placement in ("NODE","FACE"):
        dq=[model._differentiate(q,k,placement=placement)
            for k in range(6)]
        dp=[model._differentiate(p,k,placement=placement)
            for k in range(6)]
        expected+=.5*sum(
            float(np.sum(wx*dp[i]*np.einsum(
                "ab,bxs->axs",model.stiffness[i,j],dq[j],
                optimize=True))) for i in range(6)
            for j in range(6))
    actual=float(np.sum(p*model.stiffness_action(q)))
    swapped=float(np.sum(q*model.stiffness_action(p)))
    np.testing.assert_allclose(actual,expected,rtol=1.e-10,atol=2.e-7)
    np.testing.assert_allclose(actual,swapped,rtol=1.e-10,atol=2.e-7)
