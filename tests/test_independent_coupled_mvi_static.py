"""Control-flow tests of rank-fixed coupled nonlinear MVI Y/X static driver.

All halfstep physics is mocked: no shell calculation is executed.
"""
import numpy as np
import pytest

import independent_mvi.coupled_static as engine
from independent_mvi.coupled_surface import (
    CoupledMVIFactors,CoupledMVISurfaceGrid,NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram,CoupledMVIHalfstepResult,
)
from flexo_gradient_g3d import G3DMaterial


def build():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(100.,.2,.01,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.3,n_gauss=2)
    x=np.zeros((surface.n_fields,1,9))
    y=np.zeros((surface.n_fields,1,5))
    x[:,:,1:-1]=.1
    y[:]=.2
    fac=CoupledMVIFactors(x,y)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    return surface,fac,program


def test_sweeps_y_then_x_and_require_two_stable_sweeps(monkeypatch):
    surface,fac,program=build()
    order=[]
    def half(surf,factors,*,axis,**kwargs):
        order.append(axis)
        return CoupledMVIHalfstepResult(factors,True,"STATIONARY",axis,0,0.,())
    def norms(*args,**kwargs):
        return 0.,0.,0.
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",norms)
    result=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.)
    assert result.converged
    assert order==["Y","X"]
    assert result.sweeps==1
    assert result.status=="MVI_FIXED_RANK_FULL_NODAL_STATIONARY"
    assert result.full_nodal_norm==0.
    assert result.factors.rank==fac.rank


@pytest.mark.parametrize("active_fields",[None,(1,)])
def test_large_other_field_does_not_hide_active_field_change(monkeypatch,active_fields):
    surface,fac,program=build()
    x=fac.x.copy()
    x[0]*=1.e12
    fac=CoupledMVIFactors(x,fac.y)
    def half(surf,factors,*,axis,**kwargs):
        x=factors.x.copy()
        if axis=="X":
            x[1]*=2.
        return CoupledMVIHalfstepResult(
            CoupledMVIFactors(x,factors.y),True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",lambda *a,**k:(0.,0.,0.))
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.,active_fields=active_fields,
        settings=engine.CoupledMVISweepSettings(max_sweeps=1))
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_SWEEPS_EXHAUSTED"
    assert out.field_change==pytest.approx(.5)


@pytest.mark.parametrize("failed_axis",("Y","X"))
def test_failed_halfstep_discards_incomplete_sweep(monkeypatch,failed_axis):
    surface,fac,program=build()
    seen=[]
    def half(surf,factors,*,axis,**kwargs):
        seen.append(axis)
        if axis==failed_axis:
            return CoupledMVIHalfstepResult(
                factors,False,"TANGENT_NOT_SOLVED",axis,1,1.,())
        return CoupledMVIHalfstepResult(factors,True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.)
    assert not out.converged
    assert out.sweeps==0
    np.testing.assert_array_equal(out.factors.x,fac.x)
    np.testing.assert_array_equal(out.factors.y,fac.y)
    assert out.status=="MVI_"+failed_axis+"_TANGENT_NOT_SOLVED"
    assert seen==(["Y"] if failed_axis=="Y" else ["Y","X"])


def test_cancel_before_first_sweep_does_not_touch_physics(monkeypatch):
    surface,fac,program=build()
    def never(*args,**kwargs):
        raise AssertionError("cancel must prevent halfstep")
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",never)
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.,cancelled=lambda:True)
    assert not out.converged
    assert out.status=="MVI_COUPLED_CANCELLED"


def test_no_stationarity_claim_from_nonzero_projected_directions(monkeypatch):
    surface,fac,program=build()
    def half(surf,factors,*,axis,**kwargs):
        return CoupledMVIHalfstepResult(factors,True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",
                        lambda *args,**kwargs:(.1,.1,.1))
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.,
        settings=engine.CoupledMVISweepSettings(max_sweeps=2))
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_SWEEPS_EXHAUSTED"
    assert out.sweeps==2


def test_rank_insufficient_rejected_even_when_both_factor_families_stationary(monkeypatch):
    surface,fac,program=build()
    def half(surf,factors,*,axis,**kwargs):
        return CoupledMVIHalfstepResult(factors,True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",
                        lambda *args,**kwargs:(0.,0.,.03))
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.)
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK"
    assert out.full_nodal_norm==pytest.approx(.03)
    assert out.factors.rank==fac.rank


def test_full_stationarity_requires_nodal_gradient_from_own_assembler(monkeypatch):
    surface,fac,program=build()
    from types import SimpleNamespace
    surface.weak_work=lambda *a,**k:SimpleNamespace(
        gradient_x=np.zeros_like(fac.x),gradient_y=np.zeros_like(fac.y))
    with pytest.raises(ValueError,match="FULL_NODAL_FIRST_VARIATION_REQUIRED"):
        engine._projected_norms(surface,fac,program,
                                np.ones(surface.n_fields),1.,None,None)


def test_excluded_essential_traces_do_not_enter_full_nodal_check():
    surface,fac,program=build()
    from types import SimpleNamespace
    node=np.zeros(surface.shape)
    node[:,0,:]=10.
    node[:,-1,:]=-10.
    surface.weak_work=lambda *a,**k:SimpleNamespace(
        gradient_x=np.zeros_like(fac.x),gradient_y=np.zeros_like(fac.y),
        nodal_gradient=node)
    assert engine._projected_norms(surface,fac,program,
                        np.ones(surface.n_fields),1.,None,None)==(0.,0.,0.)


def test_free_interior_nodal_direction_is_checked():
    surface,fac,program=build()
    from types import SimpleNamespace
    node=np.zeros(surface.shape)
    node[:,4,2]=.05
    surface.weak_work=lambda *a,**k:SimpleNamespace(
        gradient_x=np.zeros_like(fac.x),gradient_y=np.zeros_like(fac.y),
        nodal_gradient=node)
    assert engine._projected_norms(surface,fac,program,
                        np.ones(surface.n_fields),1.,None,None)==(0.,0.,.05)


def test_rank_five_keeps_five_separated_slots_at_every_sweep(monkeypatch):
    """MVI-5 cannot be renamed MVI-1 or accepted early on a short series."""
    surface,seed,program=build()
    x=np.repeat(seed.x,5,axis=1)
    y=np.repeat(seed.y,5,axis=1)
    factors=CoupledMVIFactors(x,y)
    requested_rank=5
    seen=[]
    def halfstep(surf,current,*,axis,**kwargs):
        seen.append((axis,current.rank))
        return CoupledMVIHalfstepResult(
            current,True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",halfstep)
    # This synthetic work is deliberately not stationary at any sweep.
    # A full MAX_SWEEPS result must remain an unsuccessful MVI-5, not
    # an artificial reduced-rank success.
    monkeypatch.setattr(engine,"_projected_norms",
                        lambda *args,**kwargs:(.05,.04,.03))
    out=engine.solve_coupled_mvi_static(
        surface,factors,program=program,
        field_scales=np.ones(surface.n_fields),energy_scale_j=1.,
        settings=engine.CoupledMVISweepSettings(max_sweeps=4))
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_SWEEPS_EXHAUSTED"
    assert out.sweeps==4
    assert out.factors.rank==requested_rank
    assert seen==[("Y",5),("X",5)]*4
    assert len(out.history)==4
    assert all(row["rank"]==5 for row in out.history)


def test_projected_norms_invariant_to_reciprocal_rank_one_scaling():
    """The same physical field and nodal work give the same X/Y gates."""
    from types import SimpleNamespace
    surface,fac,program=build()
    rng=np.random.default_rng(423)
    nodal=rng.normal(size=surface.shape)
    def work(current):
        return SimpleNamespace(
            nodal_gradient=nodal,
            gradient_x=np.einsum("cij,crj->cri",nodal,current.y),
            gradient_y=np.einsum("cij,cri->crj",nodal,current.x))
    scales=np.linspace(.5,2.,surface.n_fields)
    original=engine._projected_norms(
        surface,fac,program,scales,1.,None,None,work_evaluator=work)
    factors=np.linspace(.01,100.,surface.n_fields)[:,None,None]
    transformed=CoupledMVIFactors(fac.x*factors,fac.y/factors)
    np.testing.assert_allclose(surface.reconstruct(fac),
                               surface.reconstruct(transformed))
    actual=engine._projected_norms(
        surface,transformed,program,scales,1.,None,None,work_evaluator=work)
    np.testing.assert_allclose(actual,original,rtol=1.e-12,atol=1.e-12)


def test_projected_norms_invariant_to_rank_two_basis_change():
    """Equivalent rank-two factorizations must have identical work gates."""
    from types import SimpleNamespace
    surface,_,program=build()
    rng=np.random.default_rng(756)
    x=np.zeros((surface.n_fields,2,surface.grid.nx))
    x[:,:,1:-1]=rng.normal(size=x[:,:,1:-1].shape)
    y=rng.normal(size=(surface.n_fields,2,surface.grid.ns))
    factors=CoupledMVIFactors(x,y)
    transform=np.array([[1.7,.4],[-.3,.8]])
    other=CoupledMVIFactors(
        np.einsum("ab,cbi->cai",transform,x),
        np.einsum("ab,cbj->caj",np.linalg.inv(transform).T,y))
    np.testing.assert_allclose(surface.reconstruct(factors),
                               surface.reconstruct(other),atol=1.e-12)
    nodal=rng.normal(size=surface.shape)
    def work(current):
        return SimpleNamespace(
            nodal_gradient=nodal,
            gradient_x=np.einsum("cij,crj->cri",nodal,current.y),
            gradient_y=np.einsum("cij,cri->crj",nodal,current.x))
    scales=np.linspace(.5,2.,surface.n_fields)
    one=engine._projected_norms(surface,factors,program,scales,1.,None,None,
                                work_evaluator=work)
    two=engine._projected_norms(surface,other,program,scales,1.,None,None,
                                work_evaluator=work)
    np.testing.assert_allclose(one,two,atol=1.e-12,rtol=1.e-12)


@pytest.mark.parametrize("axis",("X","Y"))
@pytest.mark.parametrize("dormant",(False,True))
def test_dependent_frozen_profiles_keep_gauge_invariant_work_norms(axis,dormant):
    """Synthetic nodal work only; dependent and dormant rank slots remain.

    QR cannot invert a dependent partner. Reciprocal row scaling still
    must not change the projected stationarity decision by six orders.
    """
    from types import SimpleNamespace
    from independent_mvi.coupled_halfstep import normalized_mvi_factors
    surface,_,program=build()
    x=np.zeros((surface.n_fields,2,surface.grid.nx))
    x[:,:,1:-1]=1.e-6
    y=np.broadcast_to(np.array([[1.,1.,1.,1.,1.],
                               [1.,-1.,0.,1.,-1.]]),
                       (surface.n_fields,2,surface.grid.ns)).copy()
    if axis=="X":
        y[:,1]=y[:,0]
    if dormant:
        (y if axis=="X" else x)[:,1]=0.
    factors=CoupledMVIFactors(x,y)
    row_scaling=np.array([1.e6,1.e-4])[None,:,None]
    other=CoupledMVIFactors(x*row_scaling,y/row_scaling)
    rng=np.random.default_rng(12)
    nodal=rng.normal(size=surface.shape)
    def work(current):
        return SimpleNamespace(nodal_gradient=nodal,
            gradient_x=np.einsum("cij,crj->cri",nodal,current.y),
            gradient_y=np.einsum("cij,cri->crj",nodal,current.x))
    one=engine._projected_norms(surface,factors,program,np.ones(surface.n_fields),
                               1.,None,None,work_evaluator=work)
    two=engine._projected_norms(surface,other,program,np.ones(surface.n_fields),
                               1.,None,None,work_evaluator=work)
    np.testing.assert_allclose(one,two,rtol=1.e-12,atol=1.e-12)
    normalized=normalized_mvi_factors(surface,factors,program,axis)
    assert normalized.rank==2
    np.testing.assert_allclose(surface.reconstruct(normalized),
                               surface.reconstruct(factors),rtol=1.e-12,atol=1.e-18)
    if dormant:
        # A zero frozen row cannot erase its original active partner.
        np.testing.assert_array_equal(normalized.x[:,1] if axis=="X" else normalized.y[:,1],
                                      x[:,1] if axis=="X" else y[:,1])


def test_explicit_projected_static_result_preserves_full_residual(monkeypatch):
    surface,fac,program=build()
    def half(surf,factors,*,axis,**kwargs):
        return CoupledMVIHalfstepResult(factors,True,"STATIONARY",axis,0,0.,())
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",lambda *a,**k:(0.,0.,.03))
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.,settings=engine.CoupledMVISweepSettings(
            allow_projected_convergence=True))
    assert out.converged
    assert out.status=="CONVERGED_PROJECTED"
    assert out.full_nodal_norm==pytest.approx(.03)
    assert out.factors.rank==fac.rank
    assert out.history[-1]["full_nodal"]==pytest.approx(.03)


@pytest.mark.parametrize("value",(1,"true",None))
def test_projected_acceptance_requires_explicit_boolean(value):
    with pytest.raises(ValueError,match="FLAG_MUST_BE_BOOLEAN"):
        engine.CoupledMVISweepSettings(allow_projected_convergence=value)
