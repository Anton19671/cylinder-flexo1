"""Independent MVI nonuniform SBP/Fourier weighted Gateaux regression.

This test is purely algebraic: it patches local virtual-work coefficients
to a constant without integrating physical shell material laws. It catches
an omitted wx or ws factor in X/Y gradients even when individual quadrature
weights happen to be uniform along one coordinate.
"""
import numpy as np

from independent_mvi.coupled_surface import (
    CoupledMVIFactors,CoupledMVISurfaceGrid,NativeCoupledMVISurface,
)
from flexo_gradient_g3d import G3DMaterial


def test_nonuniform_sbp_quadrature_appears_in_both_factor_gradients():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(50.,.22,.02,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(11,7,1.6,2.),
        thickness_m=.18,n_gauss=2)
    x=np.zeros((surface.n_fields,1,surface.grid.nx))
    y=np.zeros((surface.n_fields,1,surface.grid.ns))
    x[:,0,:]=np.linspace(.13,.46,surface.grid.nx)
    y[:,0,:]=.2+np.cos(2*np.pi*np.arange(surface.grid.ns)/surface.grid.ns)*.13
    factors=CoupledMVIFactors(x,y)
    class Synthetic:
        def __init__(self,q):
            self.energy_j_per_m2=float(np.sum(q))
            self.mechanical=np.zeros((surface.n_mechanical,6))
            self.mechanical[:,0]=1.
            self.electrical=np.zeros((0,3))
    surface._local=lambda jets:Synthetic(jets.q)
    result=surface.weak_work(factors)
    weights=surface.grid.wx[:,None]*surface.grid.ws[None,:]
    node=np.broadcast_to(weights,(surface.n_fields,*weights.shape))
    np.testing.assert_allclose(result.nodal_gradient,node,rtol=0.,atol=1e-13)
    expected_x=np.einsum("cij,crj->cri",node,factors.y,optimize=True)
    expected_y=np.einsum("cij,cri->crj",node,factors.x,optimize=True)
    np.testing.assert_allclose(result.gradient_x,expected_x,atol=1e-13)
    np.testing.assert_allclose(result.gradient_y,expected_y,atol=1e-13)
    # Axial boundary SBP weights differ from interior weights.
    assert surface.grid.wx[0]!=surface.grid.wx[surface.grid.nx//2]
    assert not np.allclose(expected_x[:,:,0],expected_x[:,:,surface.grid.nx//2])


def test_nonuniform_quadrature_mixed_derivative_adjoint():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(50.,.22,.02,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(11,7,1.6,2.),
        thickness_m=.18,n_gauss=2)
    rng=np.random.default_rng(915)
    x=rng.normal(size=(surface.n_fields,2,surface.grid.nx))*.01
    y=rng.normal(size=(surface.n_fields,2,surface.grid.ns))*.01
    factors=CoupledMVIFactors(x,y)
    class Synthetic:
        def __init__(self,jets):
            qxs=np.asarray(jets.qxs).reshape(surface.n_mechanical)
            self.energy_j_per_m2=.5*float(qxs@qxs)
            self.mechanical=np.zeros((surface.n_mechanical,6))
            self.mechanical[:,4]=qxs
            self.electrical=np.zeros((0,3))
    surface._local=lambda jets:Synthetic(jets)
    work=surface.weak_work(factors)
    direction=rng.normal(size=x.shape)
    direction/=np.linalg.norm(direction)
    h=1.e-6
    def energy(step):
        return surface.weak_work(CoupledMVIFactors(x+step*direction,y)).energy_j
    numerical=(energy(h)-energy(-h))/(2*h)
    exact=float(np.sum(work.gradient_x*direction))
    np.testing.assert_allclose(numerical,exact,rtol=2e-6,atol=1e-10)
