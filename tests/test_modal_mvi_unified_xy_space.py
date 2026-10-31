"""No-solver contracts for unified unrestricted modal MVI on larger grids."""
import numpy as np
import pytest

from cylinder_modal_separated_space import fixed_trace_xy_basis
from cylinder_modal_reference import _check_unrestricted_modal_mvi_bounds


@pytest.mark.parametrize("nx,ns,rank,allowed", [
    (13,21,1,True), (13,21,3,True), (13,21,4,True),
    (13,21,8,False), (21,63,1,True), (21,63,2,False),
])
def test_cuf_te4_unrestricted_direction_limit(nx,ns,rank,allowed):
    if allowed:
        _check_unrestricted_modal_mvi_bounds(nx,ns,15,rank)
    else:
        with pytest.raises(ValueError,match="MODAL_MVI_UNRESTRICTED_DIRECTION_LIMIT"):
            _check_unrestricted_modal_mvi_bounds(nx,ns,15,rank)


@pytest.mark.parametrize("axis",["X","Y"])
def test_unrestricted_rank_one_preserves_actual_frozen_variations(axis):
    nx,ns=5,7
    x=np.zeros((1,1,nx))
    y=np.zeros((1,1,ns))
    x[0,0]=np.array([0.,1.,2.,1.,0.])
    y[0,0]=np.array([1.,2.,-1.,.5,-.5,2.,-1.])
    mask=np.zeros((1,nx,ns),bool)
    mask[:,0,:]=mask[:,-1,:]=True
    basis=fixed_trace_xy_basis(x,y,axis,mask.ravel(),
                               np.ones(mask.size))
    field=np.einsum("crx,crs->cxs",x,y).ravel()
    np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-12)
    np.testing.assert_allclose(basis[mask.ravel()],0.,atol=1e-12)
    np.testing.assert_allclose(basis@(basis.T@field),field,atol=1e-12)
    assert basis.shape[1]==(nx-2 if axis=="X" else ns)


def test_unrestricted_y_does_not_enrich_invalid_frozen_x():
    nx,ns=5,7
    x=np.ones((1,1,nx))
    y=np.ones((1,1,ns))
    mask=np.zeros((1,nx,ns),bool)
    mask[:,0,:]=True
    basis=fixed_trace_xy_basis(x,y,"Y",mask.ravel(),np.ones(mask.size))
    assert basis.shape==(mask.size,0)


def test_unrestricted_y_allows_exact_cancellation_of_fixed_x():
    nx,ns=5,7
    x=np.zeros((1,2,nx))
    x[0,0]=np.array([1.,1.,2.,3.,4.])
    x[0,1]=np.array([1.,-1.,1.,2.,1.])
    y=np.zeros((1,2,ns))
    mask=np.zeros((1,nx,ns),bool)
    mask[:,0,:]=True
    basis=fixed_trace_xy_basis(x,y,"Y",mask.ravel(),np.ones(mask.size))
    assert basis.shape==(mask.size,ns)
    np.testing.assert_allclose(basis[mask.ravel()],0.,atol=1e-12)
    np.testing.assert_allclose(basis.T@basis,np.eye(ns),atol=1e-12)
    expected=np.array([0.,2.,1.,1.,3.])
    for j in range(ns):
        col=basis[:,j].reshape(nx,ns)
        np.testing.assert_allclose(col[:,j]/col[1,j],expected/expected[1],
                                   atol=1e-12)


def test_unrestricted_x_enforces_nonuniform_angular_support_without_fourier():
    nx,ns=5,7
    x=np.ones((1,2,nx))
    y=np.zeros((1,2,ns))
    y[0,0]=np.arange(1.,ns+1)
    y[0,1]=np.arange(2.,ns+2)
    mask=np.zeros((1,nx,ns),bool)
    mask[:,2,3]=True
    basis=fixed_trace_xy_basis(x,y,"X",mask.ravel(),np.ones(mask.size))
    assert basis.shape[1]==(nx-1)*2+1
    np.testing.assert_allclose(basis[mask.ravel()],0.,atol=1e-12)
