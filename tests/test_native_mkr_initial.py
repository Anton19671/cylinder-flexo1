"""Algebra-only independent native MKR initial-state checks.

The controlled strong operator/solver tests do not run loaded shell cases.
"""
import numpy as np
import pytest

import cylinder_native_mkr_initial as native_initial
from cylinder_native_mkr_newmark import NativeStrongNewmark
from cylinder_direct_strong_fdm import DirectStrongGrid, DirectStrongBalance
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import StrongNewtonResult
from flexo_gradient_g3d import G3DMaterial


def configured(electric=False):
    model=DirectStrongShell("CUF_TE1",
        G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                    -.002 if electric else 0.,.8 if electric else 0.),
        grid=DirectStrongGrid(7,4,1.2,2.),thickness_m=.3,n_gauss=2)
    end={side: [[{"kind":"ESSENTIAL","value":0.}]
                for _ in range(model.n_mechanical)]
         for side in ("left","right")}
    electrical={side: [{"kind":"ESSENTIAL","value":0.} for _ in range(2)]
                for side in ("left","right")} if electric else None
    engine=NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
                               mechanical_ends=end,electrical_ends=electrical,
                               density_kg_m3=3.)
    return model,engine


def test_initial_state_solves_for_acceleration_not_displacement(monkeypatch):
    model,engine=configured()
    q=np.zeros(model.shape)
    v=np.zeros_like(q)
    seen=[]
    def balance(actual_q,p,**kwargs):
        np.testing.assert_array_equal(actual_q,q)
        np.testing.assert_array_equal(kwargs["velocity"],v)
        seen.append(kwargs["acceleration"].copy())
        return DirectStrongBalance(
            kwargs["acceleration"].copy(),None,
            np.zeros((2,model.n_mechanical,model.grid.ns)),None,None,None,
            tuple(range(1,model.grid.nx-1)),1)
    monkeypatch.setattr(model,"balance",balance)
    def solve(equation,mechanical,electrical,**kwargs):
        guess=np.full_like(q,.11)
        bulk,electric=equation(guess,None)
        assert electric is None
        np.testing.assert_allclose(bulk[:,1:-1],.11)
        # Essential acceleration equations are evaluated at the trial guess.
        # For a nonzero guess at fixed zero-support ends they equal .11.
        np.testing.assert_allclose(bulk[:,[0,-1],:],.11)
        return StrongNewtonResult(np.zeros_like(q),None,1,True,"CONVERGED",0.)
    monkeypatch.setattr(native_initial,"solve_strong_nonlinear_equations",solve)
    result=native_initial.initialize_consistent_native_mkr(
        engine,q,v,mechanical_state_scale=1.,mechanical_equation_scale=1.)
    assert result.state is not None
    assert len(seen)==2  # mass solve plus physical natural-trace compatibility
    np.testing.assert_array_equal(result.state.mechanical,q)
    np.testing.assert_array_equal(result.state.acceleration,q)


def test_initial_acceleration_essential_rows_use_support_scale(monkeypatch):
    """Initial acceleration is not a force: fixed rows need absolute control."""
    model,engine=configured()
    q=np.zeros(model.shape)
    captured=[]
    def fake(equation,mechanical,electrical,**kwargs):
        captured.append(np.asarray(kwargs["mechanical_equation_scale"]).copy())
        return StrongNewtonResult(q,None,0,False,"TEST_STOP",1.)
    monkeypatch.setattr(native_initial,"solve_strong_nonlinear_equations",fake)
    answer=native_initial.initialize_consistent_native_mkr(
        engine,q,q,mechanical_state_scale=1.,
        mechanical_equation_scale=1.e4,support_absolute_tolerance=2.e-9)
    assert answer.state is None
    assert len(captured)==1
    scale=captured[0]
    assert scale.shape==q.shape
    np.testing.assert_allclose(scale[:,[0,-1],:],.2)
    np.testing.assert_allclose(scale[:,1:-1,:],1.e4)


def test_initial_state_rejects_unmet_essential_displacement():
    model,engine=configured()
    q=np.zeros(model.shape)
    q[:,0,:]=1.
    with pytest.raises(ValueError,match="INITIAL_DISPLACEMENT_ESSENTIAL"):
        native_initial.initialize_consistent_native_mkr(
            engine,q,np.zeros_like(q),
            mechanical_state_scale=1.,mechanical_equation_scale=1.)


def test_initial_state_rejects_moving_supports():
    model,engine=configured()
    q=np.zeros(model.shape)
    v=np.zeros_like(q)
    v[:,-1,:]=.1
    with pytest.raises(ValueError,match="INITIAL_MOVING_SUPPORT"):
        native_initial.initialize_consistent_native_mkr(
            engine,q,v,mechanical_state_scale=1.,mechanical_equation_scale=1.)


def test_failed_initial_equilibrium_cannot_become_accepted_state(monkeypatch):
    model,engine=configured()
    q=np.zeros(model.shape)
    def fake(*args,**kwargs):
        return StrongNewtonResult(q,None,1,False,"LINEAR_TANGENT_NOT_SOLVED",1.)
    monkeypatch.setattr(native_initial,"solve_strong_nonlinear_equations",fake)
    answer=native_initial.initialize_consistent_native_mkr(
        engine,q,q,mechanical_state_scale=1.,mechanical_equation_scale=1.)
    assert answer.state is None
    assert answer.nonlinear.status=="LINEAR_TANGENT_NOT_SOLVED"


def test_initial_electric_unknown_is_solved_simultaneously(monkeypatch):
    model,engine=configured(electric=True)
    q=np.zeros(model.shape)
    p=np.zeros((2,*model.shape[1:]))
    def balance(actual_q,actual_p,**kwargs):
        np.testing.assert_array_equal(actual_p,p)
        return DirectStrongBalance(
            np.zeros_like(q),np.zeros_like(p),
            np.zeros((2,model.n_mechanical,model.grid.ns)),
            None,None,np.zeros((2,len(p),model.grid.ns)),
            tuple(range(1,model.grid.nx-1)),1)
    monkeypatch.setattr(model,"balance",balance)
    def solve(equation,a,p0,**kwargs):
        m,e=equation(a,p0)
        assert m.shape==q.shape and e.shape==p.shape
        return StrongNewtonResult(a,p0,0,True,"CONVERGED",0.)
    monkeypatch.setattr(native_initial,"solve_strong_nonlinear_equations",solve)
    out=native_initial.initialize_consistent_native_mkr(
        engine,q,q,p,mechanical_state_scale=1.,mechanical_equation_scale=1.,
        electrical_state_scale=1.,electrical_equation_scale=1.)
    assert out.state is not None and out.state.electrical.shape==p.shape


@pytest.mark.parametrize("natural_defect",[0.,.2])
def test_free_zeroth_order_natural_end_retains_initial_mass_equation(
        monkeypatch,natural_defect):
    """Free traction at t0 must not erase the acceleration equation."""
    model,_=configured()
    ends={side:[[{"kind":"NATURAL","value":0.}]
                for _ in range(model.n_mechanical)] for side in ("left","right")}
    engine=NativeStrongNewmark(
        model,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=ends,density_kg_m3=3.)
    q=np.zeros(model.shape)
    target=np.full_like(q,.125)
    def balance(actual_q,p,**kwargs):
        np.testing.assert_array_equal(actual_q,q)
        return DirectStrongBalance(
            kwargs["acceleration"]-target,None,
            np.full((2,model.n_mechanical,model.grid.ns),natural_defect),
            None,None,None,tuple(range(1,model.grid.nx-1)),1)
    monkeypatch.setattr(model,"balance",balance)
    answer=native_initial.initialize_consistent_native_mkr(
        engine,q,q,mechanical_state_scale=1.,mechanical_equation_scale=1.)
    if natural_defect:
        assert answer.state is None
        assert answer.nonlinear.status=="INITIAL_NATURAL_TRACE_INCOMPATIBLE"
    else:
        assert answer.state is not None
        np.testing.assert_allclose(answer.state.acceleration,target,atol=1.e-9)


@pytest.mark.parametrize("natural_defect",[0.,.2])
def test_higher_static_natural_rows_do_not_replace_initial_mass_balance(
        monkeypatch,natural_defect):
    """Manufactured diagonal mass: no physical cylinder calculation.

    The old initializer had identically zero slope rows and a singular
    Jacobian even though this prescribed mass problem is nonsingular.
    An incompatible original natural action must still reject the state.
    """
    model,base=configured()
    ends={side:[[{"kind":"ESSENTIAL","value":0.},
                 {"kind":"NATURAL","value":0.}]
                for _ in range(model.n_mechanical)] for side in ("left","right")}
    engine=NativeStrongNewmark(
        model,axial_orders=(2,)*model.n_mechanical,
        mechanical_ends=ends,density_kg_m3=3.)
    q=np.zeros(model.shape)
    target=np.ones_like(q)*.125
    target[:,[0,-1]]=0.
    def balance(actual_q,p,**kwargs):
        np.testing.assert_array_equal(actual_q,q)
        return DirectStrongBalance(
            kwargs["acceleration"]-target,None,
            np.zeros((2,model.n_mechanical,model.grid.ns)),
            np.full((2,model.n_mechanical,model.grid.ns),natural_defect),
            None,None,tuple(range(2,model.grid.nx-2)),2)
    monkeypatch.setattr(model,"balance",balance)
    answer=native_initial.initialize_consistent_native_mkr(
        engine,q,q,mechanical_state_scale=1.,mechanical_equation_scale=1.)
    if natural_defect:
        assert answer.state is None
        assert not answer.nonlinear.converged
        assert answer.nonlinear.status=="INITIAL_NATURAL_TRACE_INCOMPATIBLE"
        assert answer.nonlinear.scaled_equation_norm==pytest.approx(natural_defect)
    else:
        assert answer.state is not None
        np.testing.assert_allclose(answer.state.acceleration,target,atol=1.e-9)
        np.testing.assert_array_equal(answer.state.mechanical,q)
