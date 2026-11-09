"""Source/API tests for opt-in native MKR Newmark; no loaded shell runs.

A controlled algebraic callback checks Newmark at the proposed end state.
The physical coupled engine is never called for a time-history benchmark.
"""
import numpy as np
import pytest

from cylinder_direct_strong_fdm import DirectStrongGrid, DirectStrongBalance
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import StrongNewtonResult, StrongNewtonSettings
from cylinder_native_mkr_newmark import (
    NativeNewmarkState, NativeStrongNewmark,
)
from flexo_gradient_g3d import G3DMaterial


def make_engine(theory="CUF_TE1", electric=False):
    model=DirectStrongShell(
        theory,
        G3DMaterial(90., .24, .06, .004 if electric else 0.,
                    -.002 if electric else 0., .8 if electric else 0.),
        grid=DirectStrongGrid(7, 4, 1.2, 2.), thickness_m=.3, n_gauss=2)
    orders=(1,)*model.n_mechanical
    ends={side: [[{"kind":"ESSENTIAL","value":0.}] for _ in orders]
          for side in ("left","right")}
    electrical_ends=(
        {side: [{"kind":"ESSENTIAL","value":0.} for _ in range(2)] for side in ("left","right")}
        if electric else None)
    return model, NativeStrongNewmark(
        model, axial_orders=orders, mechanical_ends=ends,
        electrical_ends=electrical_ends, density_kg_m3=3.)


def test_state_copies_fields_and_rejects_invalid_dynamics():
    q=np.zeros((6,7,4))
    state=NativeNewmarkState(0.,q,q,q)
    q[:]=9.
    assert np.max(abs(state.mechanical))==0.
    assert not state.mechanical.flags.writeable
    with pytest.raises(ValueError,match="finite"):
        NativeNewmarkState(0.,np.full((6,7,4),np.nan),q,q)
    with pytest.raises(ValueError,match="shape"):
        NativeNewmarkState(0.,np.zeros((6,7,4)),q,np.zeros((6,6,4)))


def test_native_newmark_constructor_rejects_missing_physical_data():
    model,_=make_engine()
    with pytest.raises(ValueError,match="half-orders"):
        NativeStrongNewmark(model,axial_orders=(1,),mechanical_ends={},
                            density_kg_m3=3.)
    with pytest.raises(ValueError,match="density"):
        NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
                            mechanical_ends={"left":[],"right":[]},density_kg_m3=0.)
    with pytest.raises(ValueError,match="Newmark beta"):
        NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
                            mechanical_ends={"left":[],"right":[]},
                            density_kg_m3=3.,beta=0.)
    with pytest.raises(ValueError,match="NEWMARK_UNSTABLE_PARAMETER_PAIR"):
        NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
                            mechanical_ends={"left":[],"right":[]},
                            density_kg_m3=3.,beta=.1,gamma=.5)


def test_controlled_step_uses_newmark_kinematics_and_does_not_call_mvi(monkeypatch):
    import cylinder_native_mkr_newmark as module
    model,engine=make_engine()
    q=np.zeros(model.shape)
    v=np.full_like(q,.3)
    a=np.full_like(q,.2)
    state=NativeNewmarkState(.7,q,v,a)
    dt=.1
    target=q+dt*v+(.5-engine.beta)*dt**2*a
    trace={}

    def fake_balance(mechanical,electrical=None,**kw):
        trace["q"]=mechanical.copy()
        trace["v"]=kw["velocity"].copy()
        trace["a"]=kw["acceleration"].copy()
        zeros=np.zeros_like(q)
        return DirectStrongBalance(zeros,None,
            np.zeros((2,*q.shape[:1],q.shape[-1])),None,None,None,
            tuple(range(1,6)),1)

    def fake_newton(equation, mechanical, electrical, **kw):
        assert electrical is None
        result=equation(target,None)
        assert result[0].shape==q.shape and result[1] is None
        return StrongNewtonResult(target,None,0,True,"CONVERGED",0.)

    monkeypatch.setattr(model,"balance",fake_balance)
    monkeypatch.setattr(module,"solve_strong_nonlinear_equations",fake_newton)
    step=engine.advance(state,dt,mechanical_state_scale=1.,
                        mechanical_equation_scale=1.)
    assert step.state is not None
    np.testing.assert_allclose(trace["q"],target)
    np.testing.assert_allclose(trace["a"],0.,atol=1e-14)
    np.testing.assert_allclose(trace["v"],v+(1.-engine.gamma)*dt*a,atol=1e-14)
    np.testing.assert_allclose(step.state.velocity,trace["v"])
    np.testing.assert_allclose(step.state.acceleration,trace["a"])
    assert step.state.time_s==pytest.approx(.8)
    assert state.time_s==pytest.approx(.7)


def test_nonlinear_failure_never_accepts_time_step(monkeypatch):
    import cylinder_native_mkr_newmark as module
    model,engine=make_engine()
    q=np.zeros(model.shape)
    state=NativeNewmarkState(0.,q,q,q)

    def fail(*args,**kwargs):
        return StrongNewtonResult(q,None,2,False,"LINEAR_TANGENT_NOT_SOLVED",2.)

    monkeypatch.setattr(module,"solve_strong_nonlinear_equations",fail)
    result=engine.advance(state,.01,mechanical_state_scale=1.,
                          mechanical_equation_scale=1.)
    assert result.state is None
    assert result.nonlinear.status=="LINEAR_TANGENT_NOT_SOLVED"
    assert state.time_s==0.


def test_electric_modes_are_in_same_step_and_dont_get_inertia(monkeypatch):
    import cylinder_native_mkr_newmark as module
    model,engine=make_engine(electric=True)
    q=np.zeros(model.shape)
    p=np.zeros((2,*model.shape[1:]))
    state=NativeNewmarkState(0.,q,q,q,p)
    seen=[]

    def fake_balance(mechanical,electrical=None,**kw):
        seen.append((electrical.copy(),kw["acceleration"].copy(),
                     kw["density_kg_m3"]))
        return DirectStrongBalance(np.zeros_like(q),np.zeros_like(p),
            np.zeros((2,model.n_mechanical,model.grid.ns)),None,None,
            np.zeros((2,p.shape[0],model.grid.ns)),tuple(range(1,6)),1)

    def fake_newton(equation, mechanical, electrical, **kw):
        out=equation(mechanical,electrical)
        assert out[1].shape==electrical.shape
        return StrongNewtonResult(mechanical,electrical,0,True,"CONVERGED",0.)

    monkeypatch.setattr(model,"balance",fake_balance)
    monkeypatch.setattr(module,"solve_strong_nonlinear_equations",fake_newton)
    step=engine.advance(state,.01,mechanical_state_scale=1.,
                        mechanical_equation_scale=1.,
                        electrical_state_scale=1.,electrical_equation_scale=1.)
    assert step.state is not None
    assert step.state.electrical.shape==p.shape
    assert len(seen)==1 and seen[0][2]==3.
    np.testing.assert_allclose(seen[0][1],0.)


def test_no_spatial_mvi_or_reference_imports_in_native_step():
    import ast
    from pathlib import Path
    code=(Path(__file__).resolve().parents[1]/
          "numerics/cylinder_native_mkr_newmark.py").read_text(encoding="utf8")
    imported=[]
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,ast.Import):
            imported.extend(a.name for a in node.names)
        if isinstance(node,ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(x.startswith(("independent_mvi","cylinder_variational_",
                                  "cylinder_functional_assembly",
                                  "cylinder_newmark_mvi","cylinder_mvi_"))
                   for x in imported)
