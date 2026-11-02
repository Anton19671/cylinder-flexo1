"""Physical frozen-partner constraints: distinguish roundoff from real BC.

These are bounded, calculation-free algebraic regression tests. They preserve
the original MVI X/Y variation subspace rather than enriching it with nodal
vectors or increasing the eigenmode acceptance tolerance.
"""
from types import SimpleNamespace

import numpy as np

from cylinder_modal_separated_space import _coefficient_null, fixed_trace_xy_basis
from cylinder_modal_mvi import _modal_seed_plan


def _example(rank=3, scale=1.0):
    rng=np.random.default_rng(20261102)
    nx,ns=5,7
    x=rng.normal(size=(1,rank,nx))
    y=rng.normal(size=(1,rank,ns))
    x[0,:,0]=1.e-15 * scale * rng.normal(size=rank)
    x[0,:,-1]=1.e-15 * scale * rng.normal(size=rank)
    x[0,:,1:-1]*=scale
    fixed=np.zeros((1,nx,ns),bool)
    fixed[:,0,:]=fixed[:,-1,:]=True
    return x,y,fixed


def test_numerical_zero_boundary_does_not_remove_current_rank_three_y_space():
    x,y,fixed=_example()
    basis=fixed_trace_xy_basis(
        x,y,"Y",fixed.ravel(),np.ones(fixed.size))
    assert basis.shape==(fixed.size,3*7)
    represented=np.einsum("crx,crs->cxs",x,y).ravel()
    projected=basis @ (basis.T @ represented)
    assert np.linalg.norm(represented-projected) <= 1.e-12*np.linalg.norm(represented)
    assert np.max(np.abs(basis[fixed.ravel()])) == 0.


def test_roundoff_rank_decision_is_invariant_under_uniform_partner_scaling():
    for scale in (1.e-4,1.,1.e6):
        x,y,fixed=_example(scale=scale)
        basis=fixed_trace_xy_basis(
            x,y,"Y",fixed.ravel(),np.ones(fixed.size))
        assert basis.shape==(fixed.size,3*7)


def test_nonzero_physical_boundary_is_still_strictly_constrained():
    x=np.zeros((1,3,5))
    x[0,0]=[1.,2.,1.,0.,0.]
    x[0,1]=[0.,0.,1.,2.,1.]
    x[0,2]=[0.,1.,0.,1.,0.]
    y=np.ones((1,3,7))
    fixed=np.zeros((1,5,7),bool)
    fixed[:,0,:]=fixed[:,-1,:]=True
    basis=fixed_trace_xy_basis(
        x,y,"Y",fixed.ravel(),np.ones(fixed.size))
    assert basis.shape==(fixed.size,7)
    assert np.max(np.abs(basis[fixed.ravel()]))==0.
    q=np.einsum("crx,crs->cxs",x[0:1,2:3],y[0:1,2:3]).ravel()
    assert np.linalg.norm(basis@(basis.T@q)-q)<1.e-12*np.linalg.norm(q)


def test_true_near_boundary_constraint_not_discarded():
    partner=np.eye(5,3)
    partner[0,:]=[1.e-6,0.,0.]
    partner[-1,:]=[0.,1.e-6,0.]
    mask=np.zeros(5,dtype=bool)
    mask[0]=mask[-1]=True
    null=_coefficient_null(partner,mask)
    assert null.shape==(3,1)
    np.testing.assert_allclose(partner[mask]@null,0.,atol=1.e-14)


def test_rank_one_cuf_thin_high_n_seeds_have_u_v_quadrature():
    f=SimpleNamespace(theory="CUF_TE1",n_mechanical=6)
    op=SimpleNamespace(nx=13,ns=21,n_components=6,rank=1,f=f)
    seeds=_modal_seed_plan(op,10,12)
    targeted=[s for s in seeds if s.get("n") in (7,8)]
    assert len(targeted)==4
    assert all(s.get("component_phase")=="circumferential_quadrature" for s in targeted)
    assert {(s["n"],s["phase"]) for s in targeted}=={
        (7,"cos"),(7,"sin"),(8,"cos"),(8,"sin")}


def test_higher_rank_harmonics_remain_unrestricted():
    f=SimpleNamespace(theory="CUF_TE1",n_mechanical=6)
    for rank in (2,3,4,8):
        op=SimpleNamespace(nx=13,ns=21,n_components=6,rank=rank,f=f)
        seeds=_modal_seed_plan(op,10,12)
        targeted=[s for s in seeds if s.get("n") in (7,8)]
        assert len(targeted)==4
        assert all(s.get("component_phase") is None for s in targeted)
