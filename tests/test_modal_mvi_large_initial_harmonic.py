"""Large-grid CUF rank>2 modal initial Fourier conditioning tests.

The previous 13x21 dispatch ignored a temporary harmonic restriction and
produced off-sector mechanical fields, later rejected by fit() as
'MODAL MVI harmonic factor reconstruction failed'. The new initial basis
must remain strictly inside its REAL frozen-Y separated variation space.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_modal_mvi import _SeparatedModalProblem
from cylinder_modal_harmonic_spectrum import _harmonic_phase_basis
from cylinder_modal_separated_space import fixed_trace_harmonic_x_basis


@pytest.mark.parametrize("rank",[3,4,8])
def test_initial_harmonic_x_basis_is_exact_frozen_partner_intersection(rank):
    nx,ns,ncomp,n=13,21,6,7
    phases=_harmonic_phase_basis(ns,n)
    rng=np.random.default_rng(20331117+rank)
    x=rng.normal(size=(ncomp,rank,nx))
    y=np.empty((ncomp,rank,ns))
    for c in range(ncomp):
        frame=np.empty((ns,rank))
        frame[:,0]=phases[0]
        frame[:,1]=phases[1]
        for j in range(2,rank):
            frame[:,j]=rng.normal(size=ns)
        y[c]=np.linalg.qr(frame,mode="reduced")[0].T
    mask=np.zeros((ncomp,nx,ns),dtype=bool)
    mask[:,0,:]=True
    mask[:,-1,:]=True
    scales=np.full(ncomp*nx*ns,.000254)
    expected=2*ncomp*(nx-2)
    basis=fixed_trace_harmonic_x_basis(
        y,n,mask.ravel(),scales,max_directions=1600)
    assert basis.shape==(scales.size,expected)
    assert np.max(np.abs(basis[mask.ravel()]))==0.
    for j in sorted({0,basis.shape[1]//2,basis.shape[1]-1}):
        field=(basis[:,j]/scales).reshape(ncomp,nx,ns)
        sector=np.einsum("cxs,bs,bt->cxt",field,phases,phases)
        np.testing.assert_allclose(field,sector,rtol=1.e-11,atol=1.e-11)
        for c in range(ncomp):
            for i in range(nx):
                line=field[c,i]
                if not np.any(line):
                    continue
                frame=y[c].T
                proj=frame @ (frame.T @ line)
                np.testing.assert_allclose(line,proj,rtol=1.e-10,atol=1.e-10)
    stub=SimpleNamespace(
        shape=(ncomp,nx,ns),nx=nx,ns=ns,nm=scales.size,ne=0,
        large_harmonic=True,scales=scales,
        fixed_mask=mask.ravel(),circumferential_harmonic=n)
    dispatched,_=_SeparatedModalProblem.projected_bases(stub,x,y,"X")
    np.testing.assert_allclose(dispatched,basis,rtol=1.e-10,atol=1.e-11)


def test_fourier_direction_must_not_be_added_outside_frozen_y():
    nx,ns,ncomp,rank=13,21,1,3
    y=np.zeros((ncomp,rank,ns))
    y[0,0]=_harmonic_phase_basis(ns,1)[0]
    y[0,1]=_harmonic_phase_basis(ns,2)[0]
    y[0,2]=_harmonic_phase_basis(ns,3)[1]
    mask=np.zeros((ncomp,nx,ns),bool)
    mask[:,0,:]=mask[:,-1,:]=True
    basis=fixed_trace_harmonic_x_basis(
        y,7,mask.ravel(),np.ones(mask.size))
    assert basis.shape==(mask.size,0)


def test_full_angularly_fixed_x_node_is_removed_not_projected():
    nx,ns=5,21
    y=np.zeros((1,3,ns))
    y[0,:2]=_harmonic_phase_basis(ns,4)
    y[0,2]=_harmonic_phase_basis(ns,5)[0]
    fixed=np.zeros((1,nx,ns),bool)
    fixed[0,2,:]=True
    basis=fixed_trace_harmonic_x_basis(
        y,4,fixed.ravel(),np.ones(fixed.size))
    assert basis.shape==(fixed.size,2*(nx-1))
    assert np.max(np.abs(basis[fixed.ravel()]))==0.
