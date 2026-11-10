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
