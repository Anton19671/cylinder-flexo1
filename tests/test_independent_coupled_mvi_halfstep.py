"""Algebra-only tests of independent nonlinear coupled MVI frozen X/Y Newton.

Synthetic quadratic field work is used to exercise coordinate updates;
no physical cylinder, transient history or reference MKR solver is run.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVIFactors,CoupledMVISurfaceGrid,NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram,CoupledMVIHalfstepSettings,
    normalized_mvi_factors,solve_coupled_mvi_halfstep,
)
from flexo_gradient_g3d import G3DMaterial


def make_model(electric=False,rank=1):
    grid=CoupledMVISurfaceGrid(9,5,1.2,2.)
    material=G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                         -.002 if electric else 0.,
                         .8 if electric else 0.)
    model=NativeCoupledMVISurface(
        "CUF_TE1",material,grid=grid,thickness_m=.3,
        n_gauss=2,potential_order=0 if electric else None)
    x=np.zeros((model.n_fields,rank,9))
    y=np.zeros((model.n_fields,rank,5))
    x[:,:,1:-1]=.02
    y[:]=.1
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS",
                                 "GROUNDED_BOTH" if electric else "OFF")
    return model,CoupledMVIFactors(x,y),program


def fake_quadratic(model,target=0.):
    def work(factors,**kwargs):
        x=factors.x
        y=factors.y
        # This is a deliberately rank-separated coefficient energy,
        # not a physical continuum model.
        g_x=x-target
        g_y=y-target
        return SimpleNamespace(energy_j=.5*(np.sum(g_x*g_x)+np.sum(g_y*g_y)),
                               gradient_x=g_x,gradient_y=g_y)
    model.weak_work=work


def test_x_halfstep_solves_only_eligible_axial_coordinates():
    model,fac,ends=make_model()
    fake_quadratic(model)
    first=solve_coupled_mvi_halfstep(
        model,fac,axis="X",program=ends,
        field_scales=np.ones(model.n_fields),energy_scale_j=1.,
        settings=CoupledMVIHalfstepSettings(max_coordinates=128))
    assert first.converged
    assert first.axis=="X"
    np.testing.assert_allclose(first.factors.x[:,:,0],0.)
    np.testing.assert_allclose(first.factors.x[:,:,-1],0.)
    np.testing.assert_allclose(first.factors.x[:,:,1:-1],0.,atol=1e-8)
    np.testing.assert_array_equal(first.factors.y,
                                  normalized_mvi_factors(model,fac,ends,"X").y)


def test_y_halfstep_solves_only_angular_partner():
    model,fac,ends=make_model()
    fake_quadratic(model)
    result=solve_coupled_mvi_halfstep(
        model,fac,axis="Y",program=ends,
        field_scales=np.ones(model.n_fields),energy_scale_j=1.,
        settings=CoupledMVIHalfstepSettings(max_coordinates=128))
    assert result.converged
    np.testing.assert_allclose(result.factors.y,0.,atol=1e-8)
    assert not np.any(result.factors.x[:,:,0])
    assert not np.any(result.factors.x[:,:,-1])


def test_qr_reparameterization_preserves_product_and_ends():
    model,fac,ends=make_model(electric=True,rank=2)
    for axis in ("X","Y"):
        normalized=normalized_mvi_factors(model,fac,ends,axis)
        np.testing.assert_allclose(model.reconstruct(normalized),
                                   model.reconstruct(fac),atol=1e-13)
        np.testing.assert_allclose(normalized.x[:,:,0],0.)
        np.testing.assert_allclose(normalized.x[:,:,-1],0.)


def test_no_floating_electric_gauge_silently_accepted():
    model,fac,ends=make_model(electric=True)
    bad=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","INSULATED")
    with pytest.raises(ValueError,match="FLOATING_ELECTRIC_GAUGE"):
        solve_coupled_mvi_halfstep(
            model,fac,axis="X",program=bad,
            field_scales=np.ones(model.n_fields),energy_scale_j=1.)


@pytest.mark.parametrize("axis",("X","Y"))
def test_incompatible_axial_trace_rejected_before_material_evaluation(axis):
    model,fac,ends=make_model()
    x=fac.x.copy()
    x[:,0,0]=1.
    wrong=CoupledMVIFactors(x,fac.y)
    def never(*args,**kwargs):
        raise AssertionError("should not evaluate the material")
    model.weak_work=never
    with pytest.raises(ValueError,match="INCOMPATIBLE_X_TRACES"):
        solve_coupled_mvi_halfstep(
            model,wrong,axis=axis,program=ends,
            field_scales=np.ones(model.n_fields),energy_scale_j=1.)


def test_cancel_does_not_publish_partial_halfstep():
    model,fac,ends=make_model()
    fake_quadratic(model,target=2.)
    out=solve_coupled_mvi_halfstep(
        model,fac,axis="X",program=ends,
        field_scales=np.ones(model.n_fields),energy_scale_j=1.,
        cancelled=lambda:True)
    assert not out.converged and out.status=="MVI_HALFSTEP_CANCELLED"
    np.testing.assert_array_equal(out.factors.x,fac.x)
    np.testing.assert_array_equal(out.factors.y,fac.y)


def test_rank_cannot_expand_and_limit_is_explicit():
    model,fac,ends=make_model(rank=2)
    fake_quadratic(model)
    with pytest.raises(ValueError,match="COORDINATE_LIMIT"):
        solve_coupled_mvi_halfstep(
            model,fac,axis="X",program=ends,
            field_scales=np.ones(model.n_fields),energy_scale_j=1.,
            settings=CoupledMVIHalfstepSettings(max_coordinates=10))


def test_no_svd_or_mkr_or_shared_spatial_imports():
    src=(Path(__file__).resolve().parents[1]/
         "independent_mvi/coupled_halfstep.py").read_text(encoding="utf8")
    assert ".svd(" not in src and "pinv(" not in src
    imports=[]
    for node in ast.walk(ast.parse(src)):
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
        if isinstance(node,ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert not any(name.startswith((
        "independent_mkr","cylinder_mkr_","cylinder_direct_strong_",
        "cylinder_variational_fdm","cylinder_functional_assembly"))
        for name in imports)


@pytest.mark.parametrize("rank",range(1,9))
def test_zero_axial_seed_preserves_all_user_angular_modes_through_y_step(rank):
    """No silent loss of prescribed modes before the next X iteration."""
    nx,ns=11,9
    surface=NativeCoupledMVISurface(
        "KL",G3DMaterial(90.,.24,.06,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(nx,ns,1.2,2.),
        thickness_m=.3,n_gauss=2)
    theta=2.*np.pi*np.arange(ns)/ns
    profiles=[np.ones(ns)]
    for k in range(1,5):
        profiles.extend((np.cos(k*theta),np.sin(k*theta)))
    yy=np.broadcast_to(np.stack(profiles[:rank]),
                       (surface.n_fields,rank,ns)).copy()
    xx=np.zeros((surface.n_fields,rank,nx))
    original=CoupledMVIFactors(xx,yy)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    normalized=normalized_mvi_factors(surface,original,program,"Y")
    np.testing.assert_array_equal(normalized.y,original.y)
    np.testing.assert_array_equal(normalized.x,original.x)
    # In this state Y is dormant; no physical response or mode is
    # allowed to be fabricated merely by coordinate normalization.
    np.testing.assert_array_equal(surface.reconstruct(normalized),0.)
    # X is permitted to activate ANY of the original R angular
    # directions without a numerical rank reduction or substitution.
    updated_x=np.zeros_like(xx)
    updated_x[:,:,1:-1]=np.arange(1,rank+1)[None,:,None]*.001
    activated=CoupledMVIFactors(updated_x,normalized.y)
    assert activated.rank==rank
    assert np.linalg.matrix_rank(activated.y[0],tol=1.e-10)==rank
    assert np.max(np.abs(surface.reconstruct(activated)))>0.


@pytest.mark.parametrize("rank",(2,3,5,8))
def test_duplicate_or_zero_angular_modes_not_silently_replaced(rank):
    nx,ns=11,9
    surface=NativeCoupledMVISurface(
        "KL",G3DMaterial(90.,.24,.06,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(nx,ns,1.2,2.),
        thickness_m=.3,n_gauss=2)
    xx=np.zeros((surface.n_fields,rank,nx))
    yy=np.ones((surface.n_fields,rank,ns))
    xx[:,:,1:-1]=.001
    yy[:,0]=0.
    start=CoupledMVIFactors(xx,yy)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    for axis in ("X","Y"):
        out=normalized_mvi_factors(surface,start,program,axis)
        assert out.rank==rank
        np.testing.assert_allclose(surface.reconstruct(out),
                                   surface.reconstruct(start),atol=1.e-14)
        np.testing.assert_array_equal(out.x,start.x)
        np.testing.assert_array_equal(out.y,start.y)


@pytest.mark.parametrize("rank",range(1,9))
def test_term_diagnostics_keep_every_rank_slot_even_if_dormant(rank):
    from independent_mvi.coupled_surface import mvi_separated_term_norms
    nx,ns=11,9
    x=np.zeros((3,rank,nx))
    y=np.zeros((3,rank,ns))
    y[:]=1.
    x[:,0,1:-1]=.2
    fac=CoupledMVIFactors(x,y)
    mode=mvi_separated_term_norms(fac)
    assert mode.shape==(3,rank)
    np.testing.assert_allclose(mode[:,0],
        .2*np.sqrt(nx-2)*np.sqrt(ns),rtol=1e-14)
    if rank>1:
        np.testing.assert_array_equal(mode[:,1:],0.)
    assert fac.rank==rank


def test_physical_end_trace_cannot_hide_behind_factor_rescaling():
    """An almost-zero X_end times a huge Y is not a clamped end."""
    model,fac,ends=make_model()
    x=fac.x.copy()
    y=fac.y.copy()
    x[0,0,0]=1.e-12
    y[0,0,:]=1.e12
    wrong=CoupledMVIFactors(x,y)
    with pytest.raises(ValueError,match="INCOMPATIBLE_X_TRACES"):
        normalized_mvi_factors(model,wrong,ends,"X")


def test_homogeneous_end_gate_is_invariant_under_reciprocal_factor_scale():
    """A physically clamped state stays valid under X->aX, Y->Y/a."""
    model,fac,ends=make_model()
    x=fac.x.copy()
    y=fac.y.copy()
    x*=1.e-12
    y*=1.e12
    scaled=CoupledMVIFactors(x,y)
    before=model.reconstruct(scaled)
    normalized=normalized_mvi_factors(model,scaled,ends,"X")
    np.testing.assert_allclose(model.reconstruct(normalized),before,
                               rtol=1.e-12,atol=1.e-12)
