"""Algebra-only tests for independent MVI mass and implicit Newmark.

Uses synthetic work and a mocked Y/X nonlinear sweep. No physical loaded
cylinder, benchmark, long transient, MKR or reference FDM is executed.
"""
from types import SimpleNamespace
from pathlib import Path
import ast
import numpy as np
import pytest

import independent_mvi.coupled_newmark as dynamic
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_halfstep import CoupledMVIEndProgram
from independent_mvi.coupled_static import CoupledMVISweepResult
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
    CoupledMVIWork,
)
from flexo_gradient_g3d import G3DMaterial


def build(theory="CUF_TE1",electric=False):
    surf=NativeCoupledMVISurface(
        theory,G3DMaterial(90.,.24,.06,.003 if electric else 0.,
                           -.002 if electric else 0.,
                           .8 if electric else 0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.3,n_gauss=2,
        potential_order=0 if electric else None)
    x=np.zeros((surf.n_fields,1,9))
    x[:,:,1:-1]=.05
    y=np.ones((surf.n_fields,1,5))
    factor=CoupledMVIFactors(x,y)
    program=CoupledMVIEndProgram(
        "IMMOBILE_PARALLEL_ENDS","GROUNDED_BOTH" if electric else "OFF")
    inertia=NativeCoupledMVIInertia(surf,density_kg_m3=3.)
    native=dynamic.NativeCoupledMVINewmark(
        surf,inertia,program=program)
    state=dynamic.CoupledMVIDynamicState(
        0.,factor,np.zeros((surf.n_mechanical,9,5)),
        np.zeros((surf.n_mechanical,9,5)))
    return surf,native,state


@pytest.mark.parametrize("theory",("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def test_cuf_exact_mass_is_local_positive_for_nonzero_acceleration(theory):
    surf,driver,state=build(theory)
    a=np.zeros_like(state.acceleration)
    a[:,1:-1,:]=.5
    q=surf.reconstruct(state.factors)[:surf.n_mechanical]
    result=driver.inertia.apply(q,np.zeros_like(a),a)
    assert result.shape==a.shape
    assert np.isfinite(result).all()
    assert float(np.sum(a*result))>0.


@pytest.mark.parametrize("theory",("FSDT","REDDY","KL"))
def test_classical_and_director_mass_work_shapes(theory):
    surf,driver,state=build(theory)
    a=np.zeros_like(state.acceleration)
    a[:,1:-1,:]=.002
    q=surf.reconstruct(state.factors)[:surf.n_mechanical]
    result=driver.inertia.apply(q,np.zeros_like(a),a)
    assert result.shape==a.shape
    assert np.isfinite(result).all()


def test_newmark_keeps_electric_work_quasistatic_and_adds_mechanical_inertia(monkeypatch):
    surf,driver,state=build(electric=True)
    q0=surf.reconstruct(state.factors)
    call=[]
    def weak(factors,**kwargs):
        value=surf.reconstruct(factors)
        node=np.zeros(surf.shape)
        node[-1,:,:]=.3
        gx=np.einsum("cij,crj->cri",node,factors.y)
        gy=np.einsum("cij,cri->crj",node,factors.x)
        return CoupledMVIWork(0.,value[:surf.n_mechanical],
              value[surf.n_mechanical:],gx,gy,factors.rank,
              nodal_gradient=node)
    surf.weak_work=weak
    def inertial(q,v,a):
        call.append((q.copy(),v.copy(),a.copy()))
        return np.ones_like(q)*.12
    driver.inertia.apply=inertial
    def fake_sweep(surface,factors,*,work_evaluator,**kwargs):
        w=work_evaluator(factors)
        np.testing.assert_allclose(w.nodal_gradient[:surf.n_mechanical],.12)
        np.testing.assert_allclose(w.nodal_gradient[surf.n_mechanical:],.3)
        assert len(call)==1
        return CoupledMVISweepResult(factors,True,
            "MVI_FIXED_RANK_FULL_NODAL_STATIONARY",1,0.,0.,0.,(),0.)
    monkeypatch.setattr(dynamic,"solve_coupled_mvi_static",fake_sweep)
    step=driver.advance(state,.1,field_scales=np.ones(surf.n_fields),
                        energy_scale_j=1.)
    assert step.state is not None
    assert step.status=="MVI_NEWMARK_CONVERGED"
    assert step.state.time_s==pytest.approx(.1)
    assert step.state.factors.rank==state.factors.rank


def test_failure_never_accepts_unfinished_mvi_time_step(monkeypatch):
    surf,driver,state=build()
    def failed(*a,**kw):
        return CoupledMVISweepResult(state.factors,False,
            "MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",1,
            0.,0.,0.,(),.1)
    monkeypatch.setattr(dynamic,"solve_coupled_mvi_static",failed)
    step=driver.advance(state,.03,field_scales=np.ones(surf.n_fields),
                        energy_scale_j=1.)
    assert step.state is None
    assert not step.solver.converged
    assert state.time_s==0.


def test_moving_support_rejected_before_any_newton(monkeypatch):
    surf,driver,state=build()
    v=state.velocity.copy()
    v[:,0,:]=.2
    invalid=dynamic.CoupledMVIDynamicState(
        0.,state.factors,v,state.acceleration)
    def never(*a,**kw):
        raise AssertionError("bad boundary must fail before nonlinear solve")
    monkeypatch.setattr(dynamic,"solve_coupled_mvi_static",never)
    with pytest.raises(ValueError,match="MOVING_SUPPORT"):
        driver.advance(invalid,.1,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.)


def test_invalid_parameters_rejected():
    surf,driver,state=build()
    with pytest.raises(ValueError,match="UNSTABLE_PARAMETER_PAIR"):
        dynamic.NativeCoupledMVINewmark(
            surf,driver.inertia,program=driver.program,beta=.1,gamma=.5)
    with pytest.raises(ValueError,match="finite physical value"):
        driver.advance(state,0.,field_scales=np.ones(surf.n_fields),
                       energy_scale_j=1.)


def test_new_mvi_dynamic_has_no_mkr_or_reference_numeric_imports():
    root=Path(__file__).resolve().parents[1]
    for relative in ("independent_mvi/coupled_newmark.py",
                     "independent_mvi/coupled_inertia.py"):
        imports=[]
        for node in ast.walk(ast.parse((root/relative).read_text(encoding="utf8"))):
            if isinstance(node,ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node,ast.Import):
                imports.extend(v.name for v in node.names)
        assert not any(p.startswith((
            "cylinder_direct_strong_","cylinder_mkr_","independent_mkr",
            "cylinder_variational_dynamic","cylinder_variational_fdm",
            "cylinder_functional_assembly","cylinder_surface_linear_algebra"))
            for p in imports)
