"""Independent mathematical tests of rank-fixed X/Y dynamics (no PDE solve)."""
import numpy as np
import pytest

from cylinder_mvi_manifold_kinematics import (
    rank_factor_motion, rank_factor_tangent,
    rank_factor_first_variation, rank_factor_stationary_trace,
)


def factors(seed=12):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(3,2,7))
    y=rng.normal(size=(3,2,9))
    xd=rng.normal(size=x.shape)
    yd=rng.normal(size=y.shape)
    xdd=rng.normal(size=x.shape)
    ydd=rng.normal(size=y.shape)
    return x,y,xd,yd,xdd,ydd


def field(x,y):
    return np.einsum("crx,crs->cxs",x,y)


def test_exact_time_derivatives_of_moving_rank_two_product():
    x,y,xd,yd,xdd,ydd=factors()
    m=rank_factor_motion(x,y,xd,yd,xdd,ydd)
    assert m.rank==2
    h=1.e-4
    def q(t):
        xx=x+t*xd+.5*t*t*xdd
        yy=y+t*yd+.5*t*t*ydd
        return field(xx,yy)
    central_v=(q(h)-q(-h))/(2*h)
    central_a=(q(h)-2*q(0)+q(-h))/(h*h)
    np.testing.assert_allclose(m.field,q(0),atol=1.e-14)
    np.testing.assert_allclose(m.velocity,central_v,atol=3.e-8,rtol=3.e-8)
    np.testing.assert_allclose(m.acceleration,central_a,atol=2.e-6,rtol=2.e-6)


def test_mixed_acceleration_is_essential_even_when_factors_are_affine():
    x,y,xd,yd,_,_=factors()
    z=np.zeros_like(x)
    w=np.zeros_like(y)
    m=rank_factor_motion(x,y,xd,yd,z,w)
    expected=2.*field(xd,yd)
    np.testing.assert_allclose(m.acceleration,expected,rtol=0.,atol=1.e-14)
    assert np.max(abs(m.acceleration))>1.e-2


def test_variational_gradients_match_tangent_first_variation():
    x,y,xd,yd,*_=factors()
    rng=np.random.default_rng(31)
    residual=rng.normal(size=(3,7,9))
    tangent=rank_factor_tangent(x,y,xd,yd)
    gx,gy=rank_factor_first_variation(residual,x,y)
    lhs=np.einsum("cxs,cxs->",residual,tangent)
    rhs=np.einsum("crx,crx->",gx,xd)+np.einsum("crs,crs->",gy,yd)
    assert abs(lhs-rhs)<1.e-11


def test_gauge_velocity_cancellation_is_exact():
    x,y,_,_,_,_=factors()
    a=.3
    xd=a*x
    yd=-a*y
    xdd=a*a*x
    ydd=a*a*y
    m=rank_factor_motion(x,y,xd,yd,xdd,ydd)
    np.testing.assert_allclose(m.velocity,0.,atol=2.e-14)
    np.testing.assert_allclose(m.acceleration,0.,atol=3.e-14)


def test_rank_motion_stationary_clamped_displacement_velocity_acceleration():
    x,y,xd,yd,xdd,ydd=factors()
    for arr in (x,xd,xdd):
        arr[:,:,0]=0.
        arr[:,:,-1]=0.
    ncomp,r,nx=x.shape
    ns=y.shape[-1]
    c=np.zeros((ncomp*2*ns,ncomp*nx*ns))
    idx=0
    for comp in range(ncomp):
        for xidx in (0,nx-1):
            for sidx in range(ns):
                c[idx,comp*nx*ns+xidx*ns+sidx]=1.
                idx+=1
    defects=rank_factor_stationary_trace(c,x,y,xd,yd,xdd,ydd)
    np.testing.assert_allclose(defects,0.,atol=1.e-14)
    broken=x.copy()
    broken[:,:,0]=1.
    with pytest.raises(ValueError,match="STATIONARY_SUPPORT_VIOLATED"):
        rank_factor_stationary_trace(c,broken,y,xd,yd,xdd,ydd)


def test_invalid_shapes_and_nonfinite_fields_are_rejected():
    x,y,xd,yd,xdd,ydd=factors()
    with pytest.raises(ValueError,match="INVALID_RANK_OR_FIELD"):
        rank_factor_motion(x[:,0],y,xd,yd,xdd,ydd)
    with pytest.raises(ValueError,match="INVALID_XDOT"):
        rank_factor_motion(x,y,np.full_like(x,np.nan),yd,xdd,ydd)
    with pytest.raises(ValueError,match="INVALID_RESIDUAL"):
        rank_factor_first_variation(np.zeros((3,3,3)),x,y)
