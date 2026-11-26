"""No physical solve: prove natural SAT work with actual matched matrices."""
import numpy as np
import pytest
from cylinder_matched_axial_quartic import quartic_matched_axial_differences
from cylinder_matched_strong_sat import add_natural_trace_sat


@pytest.mark.parametrize('nx',(20,21,41))
@pytest.mark.parametrize('order',(1,2,3))
def test_matched_sat_full_work_with_fixed_displacement(nx,order):
    rng=np.random.default_rng(61126+order)
    op=quartic_matched_axial_differences(nx);h=1.2/(nx-1)
    w=h*op.weights
    q=rng.normal(size=(2,nx,4));q[:,[0,-1]]=0.
    A=rng.normal(size=q.shape)
    D=lambda matrix,k:np.einsum('ij,cjs->cis',matrix,A)/h**k
    bulk=(-1)**order*D(op.coefficient[order],order)
    traces=[];actions=[];targets=[]
    for k in range(1,order):
        normals=np.array([-1.,1.])**k
        traces.append(normals[:,None]*op.kinematic[k][[0,-1]]/h**k)
        flux=op.third_slope_flux if (order,k)==(3,1) else op.kinematic[0]
        action=D(flux,order-k-1)[:,[0,-1]].transpose(1,0,2)
        actions.append((-1)**(order-k-1)*np.array([-1.,1.])[:,None,None]**(k+1)*action)
        targets.append(rng.normal(size=action.shape))
    r=add_natural_trace_sat(bulk,w,traces,actions,targets)
    dq=np.einsum('ij,cjs->cis',op.kinematic[order],q)/h**order
    left=float(np.sum(q*r*w[None,:,None]))
    right=float(np.sum(dq*A*w[None,:,None]))
    for T,g in zip(traces,targets):
        right-=float(np.sum(np.einsum('si,cij->scj',T,q)*g))
    np.testing.assert_allclose(left,right,rtol=3.e-11,atol=3.e-9)
    assert not np.shares_memory(r,bulk)


def test_sat_rejects_invalid_weights_and_shape():
    bulk=np.zeros((1,21,4));w=np.ones(21)
    with pytest.raises(ValueError,match='BULK_OR_WEIGHTS'):
        add_natural_trace_sat(bulk,-w,[],[],[])
    with pytest.raises(ValueError,match='COUNT_MISMATCH'):
        add_natural_trace_sat(bulk,w,[np.zeros((2,21))],[],[])
    with pytest.raises(ValueError,match='INVALID_TRACE_OR_ACTION'):
        add_natural_trace_sat(bulk,w,[np.zeros((2,20))],
                              [np.zeros((2,1,4))],[np.zeros((2,1,4))])
