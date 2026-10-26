"""Standalone checks of rank-fixed continuation and moving affine traces."""
from types import SimpleNamespace
import numpy as np
import pytest
from cylinder_mvi_continuation import continue_accepted_mvi_profiles


def make_example(rank=2, n=2, nx=5, ns=6):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(n,rank,nx))
    y=rng.normal(size=(n,rank,ns))
    x[:,:,0]=0.
    old_lift=np.zeros((n,nx,ns))
    old_lift[1,0,:]=2.
    scales=np.ones_like(old_lift)
    scales[0]=2.;scales[1]=3.
    represented=scales*(old_lift+np.einsum("crx,crs->cxs",x,y))
    prev=SimpleNamespace(converged=True,mvi_certificate={"accepted":True},
                         x_profiles=x,y_profiles=y,
                         lift_scaled=old_lift,scales=scales)
    return x,y,old_lift,scales,represented,prev


def test_fixed_rank_preserves_field_when_boundary_unchanged():
    x,y,lift,s,state,prev=make_example()
    class Space:
        def defect(self,u):
            return u.reshape(lift.shape)[:,0,:]-lift[:,0,:]
    a,b,d=continue_accepted_mvi_profiles(
        prev,state[:1],state[1:],new_lift_scaled=lift,
        new_scales=s.ravel(),full_space=Space(),rank=2)
    np.testing.assert_allclose(a,x)
    np.testing.assert_allclose(b,y)
    np.testing.assert_allclose(s*(lift+np.einsum("crx,crs->cxs",a,b)),state)
    assert d["rank"]==2


def test_changed_dirichlet_lift_preserves_free_factors():
    x,y,lift,s,state,prev=make_example()
    new=lift.copy()
    new[1,0,:]=3.
    class Space:
        def defect(self,u):
            return u.reshape(lift.shape)[:,0,:]-new[:,0,:]
    a,b,d=continue_accepted_mvi_profiles(
        prev,state[:1],state[1:],new_lift_scaled=new,new_scales=s.ravel(),
        full_space=Space(),rank=2)
    np.testing.assert_array_equal(a,x)
    np.testing.assert_array_equal(b,y)
    assert d["changed_boundary_lift_scaled"]==1.


def test_failed_or_changed_rank_cannot_continue():
    _,_,lift,s,state,prev=make_example()
    class Space:
        def defect(self,u):
            return np.array([0.])
    prev.converged=False
    with pytest.raises(ValueError,match="CONVERGED_PREVIOUS"):
        continue_accepted_mvi_profiles(prev,state[:1],state[1:],
            new_lift_scaled=lift,new_scales=s,full_space=Space(),rank=2)
    prev.converged=True
    with pytest.raises(ValueError,match="INCOMPATIBLE_FIXED_RANK"):
        continue_accepted_mvi_profiles(prev,state[:1],state[1:],
            new_lift_scaled=lift,new_scales=s,full_space=Space(),rank=3)
