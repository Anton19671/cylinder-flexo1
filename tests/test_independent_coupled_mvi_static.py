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
        return 0.,0.
    monkeypatch.setattr(engine,"solve_coupled_mvi_halfstep",half)
    monkeypatch.setattr(engine,"_projected_norms",norms)
    result=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.)
    assert result.converged
    assert order==["Y","X"]
    assert result.sweeps==1
    assert result.status=="MVI_FIXED_RANK_PROJECTED_STATIONARY_ONLY"
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
                        lambda *args,**kwargs:(.1,.1))
    out=engine.solve_coupled_mvi_static(
        surface,fac,program=program,field_scales=np.ones(surface.n_fields),
        energy_scale_j=1.,
        settings=engine.CoupledMVISweepSettings(max_sweeps=2))
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_SWEEPS_EXHAUSTED"
    assert out.sweeps==2
