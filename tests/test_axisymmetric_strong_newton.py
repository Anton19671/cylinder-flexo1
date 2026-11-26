"""Exact angular restriction of strong FDM, never a modal/MVI substitution."""
import json
import numpy as np
import pytest
from cylinder_direct_strong_newton import StrongNewtonSettings,solve_strong_nonlinear_equations
from cylinder_direct_strong_model import DirectStrongShell,THEORIES
from cylinder_direct_strong_fdm import DirectStrongGrid
from flexo_gradient_g3d import G3DMaterial
from independent_mkr import run_coupled_research as native
from test_native_coupled_mkr_case_dispatch import make_case
import cylinder_direct_strong_model as model_module

def test_axisymmetric_matches_full_coupled_nonlinear_equations_and_reduces_calls():
    shape=(1,7,4)
    target=np.broadcast_to(np.linspace(.1,.7,7)[None,:,None],shape).copy()
    calls=[]
    def equations(q,p):
        assert q.shape==p.shape==shape
        calls.append(1)
        return q+q**3+.3*p-(target+target**3+.6*target),p+.2*q-2.2*target
    kw=dict(mechanical_state_scale=1.,mechanical_equation_scale=1.,
            electrical_state_scale=1.,electrical_equation_scale=1.)
    q=np.zeros(shape)
    ordinary=solve_strong_nonlinear_equations(equations,q,q,**kw,
        settings=StrongNewtonSettings(max_dofs=56,linear_solver="DENSE_LU"))
    assert ordinary.converged
    full_calls=len(calls);calls.clear()
    reduced=solve_strong_nonlinear_equations(equations,q,q,**kw,
        settings=StrongNewtonSettings(max_dofs=14,linear_solver="DENSE_LU",axisymmetric=True))
    assert reduced.converged
    np.testing.assert_allclose(reduced.mechanical,ordinary.mechanical,atol=1.e-8,rtol=1.e-8)
    np.testing.assert_allclose(reduced.electrical,ordinary.electrical,atol=1.e-8,rtol=1.e-8)
    assert len(calls)<.4*full_calls
    assert np.all(reduced.mechanical==reduced.mechanical[..., :1])

@pytest.mark.parametrize("failure",("seed","scale","load"))
def test_no_silent_projection_of_nonaxisymmetric_input(failure):
    q=np.zeros((1,7,4));r=np.ones_like(q);load=np.ones_like(q)
    {"seed":q,"scale":r,"load":load}[failure][0,3,2]+=.01
    with pytest.raises(ValueError,match="AXISYMMETRIC_(INPUT_NOT_CONSTANT|EQUATIONS_NOT_INVARIANT)"):
        solve_strong_nonlinear_equations(lambda q,p:(q-load,None),q,
            mechanical_state_scale=1.,mechanical_equation_scale=r,
            settings=StrongNewtonSettings(axisymmetric=True))

def test_full_acceptance_not_first_angular_equation_only():
    q=np.zeros((1,7,4))
    def equations(q,p):
        out=np.full_like(q,.99e-8);out[...,1]=1.01e-8
        return out,None
    answer=solve_strong_nonlinear_equations(equations,q,
        mechanical_state_scale=1.,mechanical_equation_scale=1.,
        settings=StrongNewtonSettings(axisymmetric=True))
    assert not answer.converged
    assert answer.status=="AXISYMMETRIC_FULL_EQUILIBRIUM_FAILED"

@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","Gz"))
def test_identical_jet_reuse_matches_full_constitutive_assembly(theory,gradient,monkeypatch):
    grid=DirectStrongGrid(21,4,1.2,2.,axial_scheme="MATCHED_QUARTIC_SAT_V1")
    model=DirectStrongShell(theory,G3DMaterial(1.e9,.3,.002,.001,-.0005,1.e-8,gradient),
                            grid=grid,thickness_m=.08,n_gauss=3,
                            geometry_model="GREEN_LAGRANGE")
    rng=np.random.default_rng(18)
    q=np.repeat(rng.normal(scale=1.e-7,size=(model.n_mechanical,21,1)),4,axis=-1)
    p=np.repeat(rng.normal(scale=.01,size=(2,21,1)),4,axis=-1)
    kwargs=dict(velocity=.2*q,acceleration=-5*q,density_kg_m3=2500.,eta_s=.001,
                mass_damping_rate_per_s=.1)
    raw=model._local;count=[]
    def local(jets):
        count.append(1);return raw(jets)
    monkeypatch.setattr(model,"_local",local)
    got=model.coefficients(q,p,**kwargs)
    optimized=len(count);count.clear()
    monkeypatch.setattr(model_module,"_identical_angular_jets",lambda *a:False)
    expected=model.coefficients(q,p,**kwargs)
    assert len(count)==84
    assert optimized<=84
    for a,b in zip(got,expected):
        np.testing.assert_allclose(a,b,rtol=3.e-12,atol=1.e-10)

def test_explicit_case_preflight_rejects_angular_load_and_counts_reduced_dofs(tmp_path):
    path,case=make_case(tmp_path,theory="KL",kind="STATIC",electric=True)
    case["grid"].update(nx=21,axial_scheme="MATCHED_QUARTIC_SAT_V1")
    case["solver"].update(axisymmetric=True,max_dofs=84)
    q=np.zeros((3,21,5));p=np.zeros((1,21,5))
    np.savez_compressed(tmp_path/"initial.npz",mechanical=q,electrical=p)
    np.savez_compressed(tmp_path/"loads.npz",mechanical_load=q,electrical_charge=p)
    path.write_text(json.dumps(case))
    answer=native.run(path,validate_only=True)
    assert answer["axisymmetric"] is True
    q[2,10,1]=1.
    np.savez_compressed(tmp_path/"loads.npz",mechanical_load=q,electrical_charge=p)
    with pytest.raises(ValueError,match="AXISYMMETRIC_INPUT_NOT_CONSTANT"):
        native.run(path,validate_only=True)
