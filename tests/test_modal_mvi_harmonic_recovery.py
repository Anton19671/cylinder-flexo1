"""Rank-preserving Fourier-sector X/Y reconstruction, MVI-1 through MVI-8.

No shell equations are assembled; physical sector runs are checked separately.
"""
import inspect
import numpy as np
import pytest
from cylinder_modal_mvi import _SeparatedModalProblem, _harmonic_phase_basis
from cylinder_modal_harmonic_spectrum import _harmonic_variation_directions
from cylinder_variational_mvi import reconstruct_profiles


def _mock_operator(rank, nx=13, ns=21):
    op=object.__new__(_SeparatedModalProblem)
    op.n_components=15;op.nx=nx;op.ns=ns;op.rank=rank
    op.shape=(15,nx,ns)
    op.scales=np.ones(np.prod(op.shape))
    op.circumferential_harmonic=5
    return op


@pytest.mark.parametrize('nx,ns', [(13,21),(21,63)])
@pytest.mark.parametrize('rank', range(1,9))
@pytest.mark.parametrize('axis', ['X','Y'])
def test_exact_physical_sector_is_recovered_without_off_sector_qr_columns(rank,axis,nx,ns):
    op=_mock_operator(rank,nx,ns)
    phase=_harmonic_phase_basis(ns,5)
    xpos=np.linspace(-1,1,nx)
    scaled=np.empty(op.shape)
    for c in range(op.n_components):
        a=(1+c/15)*np.sin(np.pi*(xpos+1)/2)
        if rank==1:
            b=0.4*a
        else:
            b=(.2+c/17)*np.sin(2*np.pi*(xpos+1)/2)
        scaled[c]=a[:,None]*phase[0]+b[:,None]*phase[1]
    field=scaled.ravel()
    rng=np.random.default_rng(113)
    x=rng.normal(size=(15,rank,nx))
    y=rng.normal(size=(15,rank,ns))
    xx,yy=op.fit(x,y,axis,field,None)
    np.testing.assert_allclose(reconstruct_profiles(xx,yy),scaled,atol=1e-13,rtol=1e-13)
    assert np.allclose(yy,(yy@phase.T)@phase,atol=2e-14)
    for component in reconstruct_profiles(xx,yy):
        assert np.linalg.matrix_rank(component,tol=1e-10)<=min(2,rank)


def test_rank1_rejects_field_requiring_two_spatial_factors():
    op=_mock_operator(1);ph=_harmonic_phase_basis(21,5)
    x=np.linspace(0,1,13)
    field=np.zeros(op.shape)
    field[0]=x[:,None]*ph[0]+x[:,None]**2*ph[1]
    with pytest.raises(ValueError,match='harmonic factor reconstruction failed'):
        op.fit(np.zeros((15,1,13)),np.zeros((15,1,21)),'X',field.ravel(),None)


def test_halfstep_does_not_append_unconstrained_nodal_column():
    source=inspect.getsource(_SeparatedModalProblem.halfstep)
    assert 'MODAL_MVI_FROZEN_PARTNER_BASIS_LOST_CURRENT_FIELD' in source
    assert 'np.column_stack((bm, missing))' not in source
    assert 'previous field not in frozen-partner span' in source


@pytest.mark.parametrize('rank', range(1,9))
def test_harmonic_x_variations_keep_only_actual_angular_partner_span(rank):
    nx,ns,nc=21,63,3
    phase=_harmonic_phase_basis(ns,5)
    x=np.ones((nc,rank,nx))
    y=np.zeros((nc,rank,ns))
    for c in range(nc):
        for r in range(rank):
            y[c,r]=phase[r%2]
    directions=_harmonic_variation_directions(x,y,"X",5)
    expected=nc*nx*min(rank,2)
    assert directions.shape==(nc*nx*ns,expected)
    for col in directions.T.reshape(-1,nc,nx,ns):
        assert np.allclose(col,(col@phase.T)@phase,atol=1e-13)


@pytest.mark.parametrize('rank,expected_phases', [(1,1),(2,2),(4,2),(8,2)])
def test_x_halfstep_reaches_full_axial_sector_when_angular_partners_span(rank,expected_phases):
    nc,nx,ns=2,21,63
    phase=_harmonic_phase_basis(ns,4)
    x=np.ones((nc,rank,nx))
    y=np.zeros((nc,rank,ns))
    for c in range(nc):
        for r in range(rank):
            y[c,r]=phase[r%2]
    directions=_harmonic_variation_directions(x,y,"X",4)
    assert directions.shape[1]==nc*nx*expected_phases
    # Each axial/component site has its own independent phase coordinates.
    gram=directions.T@directions
    np.testing.assert_allclose(gram,np.eye(nc*nx*expected_phases),atol=1e-12,rtol=1e-12)
