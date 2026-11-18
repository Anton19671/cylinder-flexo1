"""Algebra-only certification of native energy-FDM constrained time drivers.

Fake unit-mass/harmonic weak work is supplied; NO material shell load
calculation, physical time simulation or MVI numerical backend is run.
"""
import numpy as np
import pytest

from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_native_energy_fdm import (
    NativeEnergyFDM,NativeEnergyEndSpace,NativeEnergyWork,
)
from cylinder_native_energy_dynamics import (
    NativeEnergyState,NativeEnergyDynamics,
)
from cylinder_direct_strong_newton import (
    StrongNewtonResult,StrongNewtonSettings,StrongElectricGauge,
)
from flexo_gradient_g3d import G3DMaterial


def _engine(monkeypatch,*,electric=False,gauge=False,clamped=False,order=1):
    model=DirectStrongShell("KL",G3DMaterial(
        90.,.24,.06,.004 if electric else 0.,
        -.002 if electric else 0.,.8 if electric else 0.),
        grid=DirectStrongGrid(7,4,1.2,2.),thickness_m=.3,n_gauss=2)
    fdm=NativeEnergyFDM(model)
    mechanical_ends={
        side:[[{"kind":"ESSENTIAL" if clamped else "NATURAL","value":0.}
               for _ in range(order)]
              for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    electrical_ends=(
        {side:[{"kind":"NATURAL" if gauge else "ESSENTIAL","value":0.}]
         for side in ("left","right")}
        if electric else None)
    electrical_gauge=None
    if gauge:
        weights=np.ones((1,model.grid.nx,model.grid.ns)) / (
            model.grid.nx*model.grid.ns)
        electrical_gauge=StrongElectricGauge(
            weights,target_v=.27,potential_scale_v=1.)
    space=NativeEnergyEndSpace(
        fdm,axial_orders=(order,)*model.n_mechanical,
        mechanical_ends=mechanical_ends,
        electrical_ends=electrical_ends,electric_gauge=electrical_gauge)
    def physical_work(q,p,*,velocity,acceleration,
                      mechanical_load=None,electrical_charge=None,**kwargs):
        # ODE at each free coordinate: qddot + 2 qdot + 3 q = forcing(t).
        value=acceleration+2.*velocity+3.*q
        if mechanical_load is not None:
            value=value-mechanical_load
        electric_work=None if p is None else p-.27 if gauge else p
        return NativeEnergyWork(value,electric_work)
    monkeypatch.setattr(fdm,"balance",physical_work)
    return NativeEnergyDynamics(space,density_kg_m3=3.),model


def _settings():
    return StrongNewtonSettings(max_dofs=512,max_iterations=7,
        scaled_absolute_tolerance=2.e-8,dense_lu_dof_limit=384)


@pytest.mark.parametrize("method",("newmark","bathe","gauss"))
@pytest.mark.parametrize("clamped",(False,True))
def test_all_three_energy_integrators_preserve_constraints_and_accept_true_ode(
        monkeypatch,method,clamped):
    solver,model=_engine(monkeypatch,clamped=clamped)
    q=np.zeros(model.shape)
    force=np.ones(model.shape)*.11
    loads_at=lambda t:(force,None)
    initial=solver.initialize(
        q,q,time_s=0.,loads_at=loads_at,
        reduced_state_scale=1.,reduced_equation_scale=1.,
        settings=_settings())
    assert initial.state is not None,initial.status
    assert initial.status=="CONVERGED"
    step=getattr(solver,method)(
        initial.state,.05,loads_at=loads_at,
        reduced_state_scale=1.,reduced_equation_scale=1.,
        settings=_settings())
    assert step.state is not None,step.status
    assert step.status=="CONVERGED"
    assert step.attempted_time_s==pytest.approx(.05)
    accepted=step.state
    assert accepted.time_s==pytest.approx(.05)
    assert np.isfinite(accepted.mechanical).all()
    assert np.isfinite(accepted.velocity).all()
    assert np.isfinite(accepted.acceleration).all()
    if clamped:
        for field in (accepted.mechanical,accepted.velocity,
                      accepted.acceleration):
            np.testing.assert_allclose(field[:,[0,-1],:],0.,atol=1.e-9)
    residual=solver.space.residual(
        solver.engine.balance(
            accepted.mechanical,accepted.electrical,
            velocity=accepted.velocity,acceleration=accepted.acceleration,
            mechanical_load=force))
    np.testing.assert_allclose(residual,0.,atol=1.e-6)


@pytest.mark.parametrize("method",("newmark","bathe","gauss"))
def test_energy_integrators_dont_accept_failed_newton_step(monkeypatch,method):
    import cylinder_native_energy_dynamics as module
    solver,model=_engine(monkeypatch)
    q=np.zeros(model.shape)
    initial=NativeEnergyState(0.,q,q,q)
    def fail(*args,**kwargs):
        return StrongNewtonResult(
            args[1].copy(),None,0,False,"MOCK_NEWTON_FAILED",1.)
    monkeypatch.setattr(module,"solve_strong_nonlinear_equations",fail)
    result=getattr(solver,method)(
        initial,.05,reduced_state_scale=1.,
        reduced_equation_scale=1.,settings=_settings())
    assert result.state is None
    assert "MOCK_NEWTON_FAILED" in result.status
    assert result.nonlinear.converged is False


def test_energy_gauge_initial_guess_is_projected_not_rejected(monkeypatch):
    solver,model=_engine(monkeypatch,electric=True,gauge=True)
    q=np.zeros(model.shape)
    pguess=np.zeros((1,model.grid.nx,model.grid.ns))
    initial=solver.initialize(
        q,q,pguess,reduced_state_scale=1.,
        reduced_equation_scale=1.,settings=_settings())
    assert initial.state is not None,initial.status
    np.testing.assert_allclose(
        initial.state.electrical.mean(),.27,atol=1.e-8)
    step=solver.newmark(
        initial.state,.05,reduced_state_scale=1.,
        reduced_equation_scale=1.,settings=_settings())
    assert step.state is not None,step.status
    np.testing.assert_allclose(
        step.state.electrical.mean(),.27,atol=1.e-8)


def test_energy_state_rejects_incompatible_support_velocity(monkeypatch):
    solver,model=_engine(monkeypatch,clamped=True)
    q=np.zeros(model.shape)
    v=np.zeros_like(q)
    v[:,0,:]=1.
    state=NativeEnergyState(0.,q,v,q)
    with pytest.raises(ValueError,match="VELOCITY_ESSENTIAL"):
        solver.newmark(state,.05,reduced_state_scale=1.,
                       reduced_equation_scale=1.,settings=_settings())


@pytest.mark.parametrize("amplitude",(1.e-12,1.e-150,1.e150))
@pytest.mark.parametrize("field",("mechanical","velocity","acceleration"))
def test_energy_constraints_cannot_erase_small_or_mixed_unit_initial_fields(
        monkeypatch,amplitude,field):
    solver,model=_engine(monkeypatch,clamped=True)
    fields={key:np.zeros(model.shape)
            for key in ("mechanical","velocity","acceleration")}
    # A large admissible component and angular column must not conceal a
    # support violation in a different component/column, even in SI nano units.
    fields[field][1,1:-1,:]=1.e150
    fields[field][0,1:-1,1]=1.e150
    fields[field][0,0,0]=amplitude
    state=NativeEnergyState(0.,**fields)
    with pytest.raises(ValueError,match="NATIVE_ENERGY_INCOMPATIBLE"):
        solver._coordinates(state)


@pytest.mark.parametrize("order",(1,2,3))
def test_energy_constraints_accept_roundoff_in_tiny_valid_fields(monkeypatch,order):
    solver,model=_engine(monkeypatch,clamped=True,order=order)
    rng=np.random.default_rng(918)
    coords=[rng.normal(size=solver.nm) for _ in range(3)]
    fields=[solver._mechanical_lift(c) for c in coords]
    amplitudes=np.array((1.e-150,1.e-12,1.e150))[:,None,None]
    state=NativeEnergyState(0.,*(a*amplitudes for a in fields))
    q,v,a,e=solver._coordinates(state)
    for restored,wanted in zip((q,v,a),
            (state.mechanical,state.velocity,state.acceleration)):
        np.testing.assert_allclose(
            solver._mechanical_lift(restored)/amplitudes,
            wanted/amplitudes,rtol=1.e-12,atol=1.e-12)


def test_energy_accepted_state_cannot_hide_tiny_grounding_violation(monkeypatch):
    solver,model=_engine(monkeypatch,electric=True)
    q=np.zeros(model.shape)
    p=np.zeros((1,*model.shape[1:]))
    p[0,0,0]=1.e-12
    state=NativeEnergyState(0.,q,q,q,p)
    with pytest.raises(ValueError,match="ELECTRIC_ESSENTIAL/GAUGE_TRACES"):
        solver._coordinates(state)


@pytest.mark.parametrize("method",("newmark","bathe","gauss"))
def test_energy_drivers_reject_unrepresentable_tiny_steps(monkeypatch,method):
    solver,model=_engine(monkeypatch)
    q=np.zeros(model.shape)
    state=NativeEnergyState(0.,q,q,q)
    with pytest.raises(ValueError,match="UNREPRESENTABLE"):
        getattr(solver,method)(
            state,1.e-300,reduced_state_scale=1.,
            reduced_equation_scale=1.,settings=_settings())


@pytest.mark.parametrize("method",("newmark","bathe","gauss"))
def test_gauge_cannot_hide_new_incompatible_charge_at_a_time_stage(monkeypatch,method):
    solver,model=_engine(monkeypatch,electric=True,gauge=True)
    q=np.zeros(model.shape)
    initial=solver.initialize(q,q,reduced_state_scale=1.,
        reduced_equation_scale=1.,settings=_settings())
    assert initial.state is not None
    with pytest.raises(ValueError,match="INCOMPATIBLE_INSULATED_CHARGE"):
        getattr(solver,method)(initial.state,.05,
            loads_at=lambda t:(None,np.ones((1,*model.shape[1:]))),
            reduced_state_scale=1.,reduced_equation_scale=1.,settings=_settings())
