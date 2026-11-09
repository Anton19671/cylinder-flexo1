"""Algebraic-only tests for the native strong MKR Bathe two-stage driver.

The tests mock Newton and the physical shell callback. No shell simulation,
time history, external benchmark or workflow is started by these tests.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import cylinder_native_mkr_bathe as bathe
from cylinder_native_mkr_newmark import NativeNewmarkState
from cylinder_direct_strong_newton import StrongNewtonResult
from cylinder_direct_strong_fdm import DirectStrongBalance
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_native_mkr_newmark import NativeStrongNewmark
from flexo_gradient_g3d import G3DMaterial
from cylinder_direct_strong_fdm import DirectStrongGrid


def setup(electric=False):
    model=DirectStrongShell("CUF_TE1",
        G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                    -.002 if electric else 0.,.8 if electric else 0.),
        grid=DirectStrongGrid(7,4,1.2,2.),
        thickness_m=.3,n_gauss=2)
    mechanical_ends={
        side: [[{"kind":"ESSENTIAL","value":0.}]
               for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    electrical_ends={
        side: [{"kind":"ESSENTIAL","value":0.} for _ in range(2)]
        for side in ("left","right")} if electric else None
    engine=NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=mechanical_ends,electrical_ends=electrical_ends,
        density_kg_m3=3.)
    q=np.zeros(model.shape);v=np.full_like(q,.12);a=np.full_like(q,.04)
    v[:,[0,-1],:]=0.
    a[:,[0,-1],:]=0.
    p=np.zeros((2,7,4)) if electric else None
    return model,bathe.NativeStrongBathe(engine),NativeNewmarkState(.3,q,v,a,p)


def fake_balance(model, history):
    def evaluate(q,p,**kw):
        history.append((q.copy(),None if p is None else p.copy(),
                        kw["velocity"].copy(),kw["acceleration"].copy(),
                        kw["mechanical_load"]))
        zeros=np.zeros(model.shape)
        return DirectStrongBalance(zeros,
            None if p is None else np.zeros_like(p),
            np.zeros((2,model.n_mechanical,model.grid.ns)),None,None,
            None if p is None else np.zeros((2,len(p),model.grid.ns)),
            tuple(range(1,6)),1)
    return evaluate


def test_bathe_two_stages_follow_three_point_formula(monkeypatch):
    model,engine,initial=setup()
    seen=[]
    monkeypatch.setattr(model,"balance",fake_balance(model,seen))
    loads=[]
    counter=[0]
    def fake_newton(equation,q,p,**kw):
        # Deliberate nonzero second-stage displacement increment.
        counter[0]+=1
        trial=q.copy() if counter[0]==1 else q+.003
        equation(trial,p)
        return StrongNewtonResult(trial,p,0,True,"CONVERGED",0.)
    monkeypatch.setattr(bathe,"solve_strong_nonlinear_equations",fake_newton)
    def loads_at(t):
        loads.append(t)
        return np.full(model.shape,t),None
    output=engine.advance(initial,.07,loads_at=loads_at,
        mechanical_state_scale=1.,mechanical_equation_scale=1.)
    assert output.status=="CONVERGED"
    assert output.first is not None and output.state is not None
    assert len(seen)==2 and len(loads)==2
    np.testing.assert_allclose(loads,[initial.time_s+bathe.BATHE_GAMMA*.07,
                                      initial.time_s+.07])
    assert output.state.time_s==pytest.approx(.37)
    assert initial.time_s==pytest.approx(.3)
    assert output.first.time_s<output.state.time_s
    for sample,time in zip(seen,loads):
        np.testing.assert_allclose(sample[4],time)


@pytest.mark.parametrize("stage", (1,2))
def test_bathe_failure_never_returns_final_state(stage,monkeypatch):
    model,engine,initial=setup()
    monkeypatch.setattr(model,"balance",fake_balance(model,[]))
    count=[0]
    def fake_newton(equation,q,p,**kw):
        count[0]+=1
        if count[0]==stage:
            return StrongNewtonResult(q,p,1,False,"NOT_CONVERGED",1.)
        equation(q,p)
        return StrongNewtonResult(q,p,0,True,"CONVERGED",0.)
    monkeypatch.setattr(bathe,"solve_strong_nonlinear_equations",fake_newton)
    out=engine.advance(initial,.02,mechanical_state_scale=1.,
                       mechanical_equation_scale=1.)
    assert out.state is None
    assert out.status==("FIRST_STAGE_" if stage==1 else "SECOND_STAGE_")+"NOT_CONVERGED"
    assert out.first is None if stage==1 else out.first is not None
    assert initial.time_s==.3


def test_bathe_electric_modes_remain_active_on_both_stages(monkeypatch):
    model,engine,initial=setup(electric=True)
    seen=[]
    monkeypatch.setattr(model,"balance",fake_balance(model,seen))
    count=[0]
    def fake_newton(equation,q,p,**kw):
        count[0]+=1
        assert p.shape==(2,7,4)
        equation(q,p)
        return StrongNewtonResult(q,p,0,True,"CONVERGED",0.)
    monkeypatch.setattr(bathe,"solve_strong_nonlinear_equations",fake_newton)
    out=engine.advance(initial,.02,mechanical_state_scale=1.,
                       mechanical_equation_scale=1.,
                       electrical_state_scale=1.,electrical_equation_scale=1.)
    assert out.state is not None and out.first is not None
    assert len(seen)==2 and count[0]==2
    assert out.state.electrical.shape==(2,7,4)


@pytest.mark.parametrize("field",("velocity","acceleration"))
def test_bathe_rejects_inconsistent_support_before_any_stage(field,monkeypatch):
    model,engine,state=setup()
    q=np.array(state.mechanical)
    v=np.array(state.velocity)
    a=np.array(state.acceleration)
    {"velocity":v,"acceleration":a}[...,0,:]=.1
    incompatible=NativeNewmarkState(state.time_s,q,v,a,state.electrical)
    def never(*args,**kwargs):
        raise AssertionError("strong Newton must not run")
    monkeypatch.setattr(bathe,"solve_strong_nonlinear_equations",never)
    with pytest.raises(ValueError,match="INITIAL_"):
        engine.advance(incompatible,.02,mechanical_state_scale=1.,
                       mechanical_equation_scale=1.)


def test_native_bathe_source_does_not_import_variational_or_mvi():
    path=Path(__file__).resolve().parents[1]/"numerics/cylinder_native_mkr_bathe.py"
    tree=ast.parse(path.read_text(encoding="utf8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(a.name for a in node.names)
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(x.startswith((
        "independent_mvi","cylinder_variational_","cylinder_bathe_mvi",
        "cylinder_functional_assembly","cylinder_newmark_mvi",
        "cylinder_mvi_")) for x in imports)
