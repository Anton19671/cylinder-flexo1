"""Contract tests for the experimental native MKR Gauss2 collocation route.

No loaded-shell calculation, transient case, or GitHub Actions invocation.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import cylinder_native_mkr_gauss2 as gauss
from cylinder_direct_strong_fdm import DirectStrongGrid, DirectStrongBalance
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import StrongNewtonResult
from cylinder_native_mkr_newmark import NativeNewmarkState, NativeStrongNewmark
from cylinder_native_mkr_initial import NativeInitialStateResult
from flexo_gradient_g3d import G3DMaterial


def setup(electric=False):
    model=DirectStrongShell(
        "CUF_TE1",
        G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                    -.002 if electric else 0.,.8 if electric else 0.),
        grid=DirectStrongGrid(7,4,1.2,2.),thickness_m=.3,n_gauss=2)
    ends={side: [[{"kind":"ESSENTIAL","value":0.}]
                 for _ in range(model.n_mechanical)]
          for side in ("left","right")}
    electric_ends=({side: [{"kind":"ESSENTIAL","value":0.}
                          for _ in range(2)] for side in ("left","right")}
                   if electric else None)
    engine=NativeStrongNewmark(model,axial_orders=(1,)*model.n_mechanical,
            mechanical_ends=ends,electrical_ends=electric_ends,
            density_kg_m3=3.)
    q=np.zeros(model.shape)
    p=np.zeros((2,7,4)) if electric else None
    return model,gauss.NativeStrongGauss2(engine),NativeNewmarkState(0.,q,q,q,p)


def test_stage_kinematics_exact_for_constant_acceleration():
    h=.12
    q=np.ones((3,7,4))*.1
    v=np.ones_like(q)*.2
    acc=np.ones_like(q)*.3
    stages=np.stack([q+h*c*v+.5*(h*c)**2*acc for c in gauss.GAUSS_C])
    sv,sa=gauss.gauss_stage_kinematics(stages,q,v,h)
    np.testing.assert_allclose(sa,np.broadcast_to(acc,sa.shape),rtol=1e-13,atol=1e-13)
    for i,c in enumerate(gauss.GAUSS_C):
        np.testing.assert_allclose(sv[i],v+h*c*acc,rtol=1e-13,atol=1e-13)
    qe,ve=gauss.gauss_end_kinematics(q,v,sv,sa,h)
    np.testing.assert_allclose(qe,q+h*v+.5*h*h*acc,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(ve,v+h*acc,rtol=1e-13,atol=1e-13)


def test_stage_electric_unknowns_are_simultaneous_and_no_mvi(monkeypatch):
    model,solver,initial=setup(electric=True)
    seen=[]
    def mock_balance(q,p,**kwargs):
        seen.append((p.copy(),kwargs["velocity"].copy(),
                     kwargs["acceleration"].copy()))
        return DirectStrongBalance(
            np.zeros_like(q),np.zeros_like(p),
            np.zeros((2,model.n_mechanical,model.grid.ns)),
            None,None,np.zeros((2,len(p),model.grid.ns)),
            tuple(range(1,6)),1)
    monkeypatch.setattr(model,"balance",mock_balance)
    def mock_newton(equation,q,p,**kwargs):
        result=equation(q,p)
        assert result[0].shape==(2*model.n_mechanical,7,4)
        assert result[1].shape==(4,7,4)
        return StrongNewtonResult(q.copy(),p.copy(),0,True,"CONVERGED",0.)
    monkeypatch.setattr(gauss,"solve_strong_nonlinear_equations",mock_newton)
    def mock_endpoint(engine,q,v,p,**kwargs):
        state=NativeNewmarkState(kwargs["time_s"],q,v,np.zeros_like(q),p)
        return NativeInitialStateResult(state,StrongNewtonResult(
            q,p,0,True,"CONVERGED",0.))
    monkeypatch.setattr(gauss,"initialize_consistent_native_mkr",mock_endpoint)
    result=solver.advance(initial,.02,mechanical_state_scale=1.,
                          mechanical_equation_scale=1.,
                          electrical_state_scale=1.,electrical_equation_scale=1.)
    assert result.status=="CONVERGED"
    assert len(result.stages)==2 and result.state is not None
    assert len(seen)==2
    assert all(v[0].shape==(2,7,4) for v in seen)
    assert result.state.time_s==pytest.approx(.02)


def test_gauss_stage_failure_rejects_endpoint(monkeypatch):
    model,solver,initial=setup()
    def fail(equation,q,p,**kwargs):
        return StrongNewtonResult(q,p,2,False,"LINEAR_TANGENT_NOT_SOLVED",1.)
    monkeypatch.setattr(gauss,"solve_strong_nonlinear_equations",fail)
    result=solver.advance(initial,.02,mechanical_state_scale=1.,
                          mechanical_equation_scale=1.)
    assert result.state is None and not result.stages
    assert result.status=="GAUSS_STAGES_LINEAR_TANGENT_NOT_SOLVED"


def test_gauss_endpoint_failure_rejects_full_step(monkeypatch):
    model,solver,initial=setup()
    def mock_balance(q,p,**kwargs):
        return DirectStrongBalance(
            np.zeros_like(q),None,
            np.zeros((2,model.n_mechanical,model.grid.ns)),
            None,None,None,tuple(range(1,6)),1)
    monkeypatch.setattr(model,"balance",mock_balance)
    def solve(equation,q,p,**kwargs):
        equation(q,p)
        return StrongNewtonResult(q,p,0,True,"CONVERGED",0.)
    monkeypatch.setattr(gauss,"solve_strong_nonlinear_equations",solve)
    def fail_endpoint(*args,**kwargs):
        return NativeInitialStateResult(None,StrongNewtonResult(
            np.zeros(model.shape),None,1,False,"FAILED",1.))
    monkeypatch.setattr(gauss,"initialize_consistent_native_mkr",fail_endpoint)
    out=solver.advance(initial,.02,mechanical_state_scale=1.,
                       mechanical_equation_scale=1.)
    assert out.state is None and len(out.stages)==2
    assert out.status=="GAUSS_ENDPOINT_FAILED"


def test_gauss_initial_acceleration_fixed_support_gate(monkeypatch):
    model,solver,state=setup()
    acceleration=np.zeros(model.shape)
    acceleration[:,-1,:]=.2
    inconsistent=NativeNewmarkState(state.time_s,state.mechanical,
                                     state.velocity,acceleration,state.electrical)
    def never(*args,**kwargs):
        raise AssertionError("Newton must not run")
    monkeypatch.setattr(gauss,"solve_strong_nonlinear_equations",never)
    with pytest.raises(ValueError,match="INITIAL_ACCELERATION_ESSENTIAL"):
        solver.advance(inconsistent,.02,mechanical_state_scale=1.,
                       mechanical_equation_scale=1.)


def test_source_excludes_cross_method_numerical_backends():
    path=Path(__file__).resolve().parents[1]/"numerics/cylinder_native_mkr_gauss2.py"
    imports=[]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
        if isinstance(node,ast.Import):
            imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(v.startswith(("cylinder_variational_","cylinder_mvi_",
                 "independent_mvi","cylinder_functional_assembly",
                 "cylinder_gauss2_variational")) for v in imports)
