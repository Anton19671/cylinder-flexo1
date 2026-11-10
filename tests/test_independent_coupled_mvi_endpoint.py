"""Synthetic endpoint rank preservation/rejection and MVI mass closure.

No physical cylinder solve. Exact rank cases use generated products; invalid
endpoint rank uses a sum of independent products.
"""
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
    CoupledMVIWork,
)
from independent_mvi.coupled_halfstep import CoupledMVIEndProgram
from independent_mvi.coupled_rank_endpoint import fit_mvi_endpoint_field
from independent_mvi.coupled_initial import close_mvi_instantaneous_state
from independent_mvi.coupled_newmark import NativeCoupledMVINewmark
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from flexo_gradient_g3d import G3DMaterial


def build(rank=1):
    surf=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(80.,.22,.03,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.3,2.),
        thickness_m=.2,n_gauss=2)
    rng=np.random.default_rng(45)
    x=np.zeros((surf.n_fields,rank,9))
    x[:,:,1:-1]=rng.normal(size=(surf.n_fields,rank,7))
    y=rng.normal(size=(surf.n_fields,rank,5))
    f=CoupledMVIFactors(x,y)
    prog=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    return surf,f,prog


def test_endpoint_exact_rank_reconstructed_without_svd():
    surf,seed,ends=build(rank=2)
    rng=np.random.default_rng(2026)
    x=seed.x.copy()
    y=seed.y.copy()
    # Keep the original independent Y span so one exact X-halfstep
    # suffices. This tests algebra, not convergence of a nonconvex ALS
    # search from two unrelated random factor subspaces.
    x[:,:,1:-1]=rng.normal(size=x[:,:,1:-1].shape)
    goal=surf.reconstruct(CoupledMVIFactors(x,y))
    result=fit_mvi_endpoint_field(
        surf,seed,goal,ends,relative_tolerance=1.e-9,
        max_iterations=100)
    assert result.accepted and result.factors is not None
    np.testing.assert_allclose(surf.reconstruct(result.factors),goal,
                               rtol=1e-8,atol=1e-8)
    assert result.factors.rank==2


def test_rank_one_endpoint_rejects_independent_rank_two_data():
    surf,seed,ends=build(rank=1)
    goal=surf.reconstruct(seed)
    x=np.linspace(0,1,9)
    y=np.arange(5,dtype=float)
    perturb=np.sin(np.pi*x)[:,None]*np.cos(2*np.pi*y[None]/5)
    goal[0]+=perturb
    result=fit_mvi_endpoint_field(
        surf,seed,goal,ends,relative_tolerance=1.e-10)
    assert not result.accepted
    assert result.factors is None


def test_endpoint_essential_axial_conflict_rejects_before_fitting():
    surf,seed,ends=build()
    goal=surf.reconstruct(seed)
    goal[:,0,:]=.2
    result=fit_mvi_endpoint_field(surf,seed,goal,ends)
    assert not result.accepted
    assert result.status=="MVI_ENDPOINT_ESSENTIAL_TRACE_CONFLICT"


def test_initial_mass_solver_fail_closed_at_coordinate_cap():
    surf,fac,ends=build()
    config=NativeCoupledMVINewmark(
        surf,NativeCoupledMVIInertia(surf,density_kg_m3=3.),
        program=ends)
    velocity=np.zeros((surf.n_mechanical,9,5))
    result=close_mvi_instantaneous_state(
        config,fac,velocity,time_s=0.,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        max_acceleration_coordinates=1)
    assert result.state is None
    assert result.status=="MVI_ACCELERATION_DENSE_COORDINATE_LIMIT"


def test_initial_moving_support_does_not_assemble_mass():
    surf,fac,ends=build()
    config=NativeCoupledMVINewmark(
        surf,NativeCoupledMVIInertia(surf,density_kg_m3=3.),
        program=ends)
    velocity=np.zeros((surf.n_mechanical,9,5))
    velocity[0,0,:]=1.
    with pytest.raises(ValueError,match="MOVING_SUPPORT"):
        close_mvi_instantaneous_state(
            config,fac,velocity,time_s=0.,
            field_scales=np.ones(surf.n_fields),energy_scale_j=1.)


def test_endpoint_fit_does_not_call_svd_and_preserves_rank(monkeypatch):
    surf,seed,ends=build(rank=2)
    target=surf.reconstruct(seed)
    target[:,1:-1,:]*=1.3
    def forbidden_svd(*args,**kwargs):
        raise AssertionError("MVI_ENDPOINT_MUST_NOT_USE_SVD")
    monkeypatch.setattr(np.linalg,"svd",forbidden_svd)
    result=fit_mvi_endpoint_field(
        surf,seed,target,ends,relative_tolerance=1.e-10)
    assert result.accepted
    assert result.factors.rank==seed.rank
    np.testing.assert_allclose(surf.reconstruct(result.factors),
                               target,atol=1.e-10,rtol=1.e-10)


def test_endpoint_rejects_small_nonzero_fixed_trace():
    surf,seed,ends=build()
    target=surf.reconstruct(seed)
    target[0,0,0]=5.e-10
    result=fit_mvi_endpoint_field(surf,seed,target,ends)
    assert result.factors is None
    assert result.status=="MVI_ENDPOINT_ESSENTIAL_TRACE_CONFLICT"
