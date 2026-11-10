"""No-solve Bathe MVI tests: two distinct stage times, atomic rejection.

All nonlinear Y/X work is replaced by a deterministic algebraic stage
callback. No physical cylinder calculation or time history is launched.
"""
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import CoupledMVIEndProgram
from independent_mvi.coupled_newmark import (
    CoupledMVIDynamicState,NativeCoupledMVINewmark,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_static import CoupledMVISweepResult
from independent_mvi.coupled_bathe import NativeCoupledMVIBathe,BATHE_GAMMA
from flexo_gradient_g3d import G3DMaterial


def configured():
    surf=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(90.,.24,.06,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.3,n_gauss=2)
    x=np.zeros((surf.n_fields,1,9))
    x[:,:,1:-1]=.03
    y=np.ones((surf.n_fields,1,5))
    fac=CoupledMVIFactors(x,y)
    z=np.zeros((surf.n_mechanical,9,5))
    state=CoupledMVIDynamicState(0.,fac,z,z)
    config=NativeCoupledMVINewmark(
        surf,NativeCoupledMVIInertia(surf,density_kg_m3=3.),
        program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF"))
    return surf,NativeCoupledMVIBathe(config),state


def accepted(factors):
    return CoupledMVISweepResult(
        factors,True,"MVI_FIXED_RANK_FULL_NODAL_STATIONARY",
        1,0.,0.,0.,(),0.)


def rejected(factors):
    return CoupledMVISweepResult(
        factors,False,"MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",
        1,0.,0.,0.,(),1.)


def test_two_stages_have_distinct_physical_times_and_kinematics(monkeypatch):
    import independent_mvi.coupled_bathe as module
    surf,solver,state=configured()
    stages=[]
    times=[]
    def fake_sweep(surface,factors,*,work_evaluator,**kw):
        stages.append((factors,work_evaluator))
        return accepted(factors)
    monkeypatch.setattr(module,"solve_coupled_mvi_static",fake_sweep)
    def loads(t):
        times.append(t)
        return None,None
    out=solver.advance(state,.1,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.,loads_at=loads)
    assert out.status=="MVI_BATHE_CONVERGED"
    assert out.state is not None and out.first_stage is not None
    assert out.state.time_s==pytest.approx(.1)
    assert out.first_stage.time_s==pytest.approx(BATHE_GAMMA*.1)
    assert times==pytest.approx([BATHE_GAMMA*.1,.1])
    assert len(stages)==2 and stages[0][1] is not stages[1][1]
    assert out.state.factors.rank==state.factors.rank


@pytest.mark.parametrize("failed_stage",(1,2))
def test_failed_stage_never_accepts_full_step(monkeypatch,failed_stage):
    import independent_mvi.coupled_bathe as module
    surf,solver,state=configured()
    count=[0]
    def fake_sweep(surface,factors,**kwargs):
        count[0]+=1
        return rejected(factors) if count[0]==failed_stage else accepted(factors)
    monkeypatch.setattr(module,"solve_coupled_mvi_static",fake_sweep)
    out=solver.advance(state,.01,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.)
    assert out.state is None
    assert count[0]==failed_stage
    assert out.first_stage is None if failed_stage==1 else out.first_stage is not None
    assert state.time_s==0.


def test_zero_dt_rejected_before_any_stage(monkeypatch):
    import independent_mvi.coupled_bathe as module
    surf,solver,state=configured()
    def never(*args,**kw):
        raise AssertionError("nonphysical dt must not run")
    monkeypatch.setattr(module,"solve_coupled_mvi_static",never)
    with pytest.raises(ValueError):
        solver.advance(state,0.,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.)


def test_bathe_viscosity_is_applied_on_both_stages_without_electric_inertia(monkeypatch):
    import independent_mvi.coupled_bathe as module
    from independent_mvi.coupled_surface import CoupledMVIWork
    surf,solver,state=configured()
    solver.configuration.viscosity.eta_s=.3
    stages=[]
    def weak(factors,**kwargs):
        q=surf.reconstruct(factors)
        nodal=np.zeros(surf.shape)
        return CoupledMVIWork(0.,q,None,
            np.zeros_like(factors.x),np.zeros_like(factors.y),
            factors.rank,nodal_gradient=nodal)
    monkeypatch.setattr(surf,"weak_work",weak)
    monkeypatch.setattr(solver.configuration.inertia,"apply",
                        lambda q,v,a:np.ones_like(q)*.1)
    def visc(q,v):
        stages.append(v.copy())
        return np.ones_like(q)*.2
    monkeypatch.setattr(solver.configuration.viscosity,"apply",visc)
    def halfstep(surface,factors,*,work_evaluator,**kw):
        work=work_evaluator(factors)
        np.testing.assert_allclose(work.nodal_gradient,.3)
        return accepted(factors)
    monkeypatch.setattr(module,"solve_coupled_mvi_static",halfstep)
    result=solver.advance(state,.05,field_scales=np.ones(surf.n_fields),
                          energy_scale_j=1.)
    assert result.state is not None
    assert len(stages)==2


@pytest.mark.parametrize("dt",(1.e-200,1.e-155,1.e-140))
def test_bathe_rejects_unrepresentable_implicit_coefficients_before_loads(
        monkeypatch,dt):
    import independent_mvi.coupled_bathe as module
    surf,solver,state=configured()
    def forbidden(*args,**kw):
        raise AssertionError("MVI_BATHE_MUST_NOT_SOLVE_UNREPRESENTABLE_DT")
    monkeypatch.setattr(module,"solve_coupled_mvi_static",forbidden)
    with pytest.raises(ValueError,match="UNREPRESENTABLE_TIME_COEFFICIENTS"):
        solver.advance(state,dt,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.,loads_at=forbidden)
    assert state.time_s==0.


def test_bathe_rejects_nonfinite_first_predictor_before_loads(monkeypatch):
    import independent_mvi.coupled_bathe as module
    surf,solver,state=configured()
    # The time coefficients remain finite, but q + dt*v overflows.
    velocity=np.zeros_like(state.velocity)
    velocity[:,1:-1,:]=1.e308
    modified=CoupledMVIDynamicState(
        state.time_s,state.factors,velocity,state.acceleration)
    def forbidden(*args,**kw):
        raise AssertionError("MVI_BATHE_MUST_NOT_SOLVE_NONFINITE_PREDICTOR")
    monkeypatch.setattr(module,"solve_coupled_mvi_static",forbidden)
    with pytest.raises(ValueError,match="NONFINITE_FIRST_PREDICTOR"):
        solver.advance(modified,10.,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.,loads_at=forbidden)
