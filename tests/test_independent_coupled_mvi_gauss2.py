"""No physical solve: independent Gauss2 tableau, fixed-rank endpoint guards.

Stage nonlinear solves and endpoint closure are mocked. An algebraic
constant-acceleration test validates both coupled stage identities.
"""
import numpy as np
import pytest

import independent_mvi.coupled_gauss2 as gauss
from independent_mvi.coupled_bathe import BATHE_GAMMA
from independent_mvi.coupled_surface import (
    NativeCoupledMVISurface,CoupledMVISurfaceGrid,CoupledMVIFactors,
)
from independent_mvi.coupled_halfstep import CoupledMVIEndProgram
from independent_mvi.coupled_static import CoupledMVISweepResult
from independent_mvi.coupled_initial import MVIAccelerationResult
from independent_mvi.coupled_rank_endpoint import MVIRankEndpointFit
from independent_mvi.coupled_newmark import (
    NativeCoupledMVINewmark,CoupledMVIDynamicState,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from flexo_gradient_g3d import G3DMaterial


def configured():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(90.,.24,.06,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.3,n_gauss=2)
    x=np.zeros((surface.n_fields,1,9))
    x[:,:,1:-1]=.04
    y=np.ones((surface.n_fields,1,5))
    f=CoupledMVIFactors(x,y)
    z=np.zeros((surface.n_mechanical,9,5))
    state=CoupledMVIDynamicState(0.,f,z,z)
    config=NativeCoupledMVINewmark(
        surface,NativeCoupledMVIInertia(surface,density_kg_m3=3.),
        program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF"))
    return surface,gauss.NativeCoupledMVIGauss2(config),state


def sweep(f,accepted=True):
    return CoupledMVISweepResult(
        f,accepted,"MVI_FIXED_RANK_FULL_NODAL_STATIONARY"
        if accepted else "MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",
        1,0.,0.,0.,(),0. if accepted else 1.)


def test_gauss_collocation_kinematics_constant_acceleration():
    h=.08
    q=np.ones((3,9,5))*.04
    v=np.ones_like(q)*.2
    a=np.ones_like(q)*.3
    stages=np.stack([
        q+h*c*v+.5*(h*c)**2*a for c in gauss.GAUSS_C])
    vs,acc=gauss.mvi_gauss_stage_kinematics(stages,q,v,h)
    np.testing.assert_allclose(acc,np.broadcast_to(a,acc.shape),
                               rtol=1e-13,atol=1e-13)
    for i,c in enumerate(gauss.GAUSS_C):
        np.testing.assert_allclose(vs[i],v+h*c*a,
                                   rtol=1e-13,atol=1e-13)
    endq,endv=gauss.mvi_gauss_endpoint(q,v,vs,acc,h)
    np.testing.assert_allclose(endq,q+h*v+.5*h*h*a,atol=1e-13)
    np.testing.assert_allclose(endv,v+h*a,atol=1e-13)


def test_gauss_joint_stages_and_endpoint_are_both_required(monkeypatch):
    surf,solver,state=configured()
    times=[]
    calls=[]
    monkeypatch.setattr(
        gauss,"solve_coupled_mvi_static",
        lambda surface,factors,**kw:
            calls.append(kw["work_evaluator"]) or sweep(factors))
    monkeypatch.setattr(gauss,"_projected_norms",
                        lambda *a,**kw:(0.,0.,0.))
    def fit(surface,seed,field,program,**kw):
        return MVIRankEndpointFit(seed,True,"MVI_ENDPOINT_EXACT_RANK",0.,0)
    monkeypatch.setattr(gauss,"fit_mvi_endpoint_field",fit)
    def closure(config,factors,velocity,**kw):
        result=CoupledMVIDynamicState(
            kw["time_s"],factors,velocity,np.zeros_like(velocity))
        return MVIAccelerationResult(result,"MVI_INITIAL_EQUILIBRIUM_CLOSED",None,None)
    monkeypatch.setattr(gauss,"close_mvi_instantaneous_state",closure)
    def loads(t):
        times.append(t)
        return None,None
    result=solver.advance(
        state,.02,field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        loads_at=loads)
    assert result.status=="MVI_GAUSS_CONVERGED"
    assert result.state is not None
    assert len(result.stages)==2 and len(result.stage_solvers)==2
    assert len(calls)==2 and calls[0] is not calls[1]
    assert result.state.time_s==pytest.approx(.02)
    assert times==pytest.approx([.02*gauss.GAUSS_C[0],
                               .02*gauss.GAUSS_C[1],.02])


@pytest.mark.parametrize("failed_stage",(1,2))
def test_failed_stage_prevents_endpoint(monkeypatch,failed_stage):
    surf,solver,state=configured()
    count=[0]
    def stage(surface,factors,**kw):
        count[0]+=1
        return sweep(factors,count[0]!=failed_stage)
    monkeypatch.setattr(gauss,"solve_coupled_mvi_static",stage)
    monkeypatch.setattr(gauss,"fit_mvi_endpoint_field",
                        lambda *a,**kw:pytest.fail("endpoint fit on failed stage"))
    result=solver.advance(
        state,.02,field_scales=np.ones(surf.n_fields),energy_scale_j=1.)
    assert result.state is None
    assert result.status.startswith("MVI_GAUSS_STAGE_"+str(failed_stage))
    assert state.time_s==0.


def test_rank_insufficient_endpoint_cannot_be_accepted(monkeypatch):
    surf,solver,state=configured()
    monkeypatch.setattr(gauss,"solve_coupled_mvi_static",
                        lambda surface,factors,**kw:sweep(factors))
    monkeypatch.setattr(gauss,"_projected_norms",
                        lambda *a,**kw:(0.,0.,0.))
    monkeypatch.setattr(gauss,"fit_mvi_endpoint_field",
        lambda *a,**kw:MVIRankEndpointFit(
            None,False,"MVI_RANK_R_ENDPOINT_NOT_REPRESENTABLE",.3,4))
    monkeypatch.setattr(gauss,"close_mvi_instantaneous_state",
                        lambda *a,**kw:pytest.fail("closure on bad rank"))
    result=solver.advance(
        state,.02,field_scales=np.ones(surf.n_fields),energy_scale_j=1.)
    assert result.state is None
    assert "RANK_R_ENDPOINT_NOT_REPRESENTABLE" in result.status
    assert result.endpoint_rank_error==pytest.approx(.3)


def test_endpoint_acceleration_closure_failure_rejects_step(monkeypatch):
    surf,solver,state=configured()
    monkeypatch.setattr(gauss,"solve_coupled_mvi_static",
                        lambda surface,factors,**kw:sweep(factors))
    monkeypatch.setattr(gauss,"_projected_norms",
                        lambda *a,**kw:(0.,0.,0.))
    monkeypatch.setattr(gauss,"fit_mvi_endpoint_field",
        lambda surf,seed,target,program,**kw:
            MVIRankEndpointFit(seed,True,"MVI_ENDPOINT_EXACT_RANK",0.,0))
    monkeypatch.setattr(gauss,"close_mvi_instantaneous_state",
        lambda *a,**kw:MVIAccelerationResult(
            None,"MVI_ACCELERATION_MASS_NOT_SOLVED",None,None))
    result=solver.advance(
        state,.02,field_scales=np.ones(surf.n_fields),energy_scale_j=1.)
    assert result.state is None
    assert result.status=="MVI_GAUSS_ENDPOINT_MVI_ACCELERATION_MASS_NOT_SOLVED"


@pytest.mark.parametrize("rank",range(1,9))
def test_gauss_rank_preserving_relaxation_in_same_angular_span(rank):
    """Exact physical interpolation from Gram solve, no SVD or rank change."""
    rng=np.random.default_rng(923+rank)
    surface=NativeCoupledMVISurface(
        "KL",G3DMaterial(90.,.24,.01,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(11,9,1.2,2.),
        thickness_m=.3,n_gauss=2)
    nx,ns=surface.grid.nx,surface.grid.ns
    theta=2.*np.pi*np.arange(ns)/ns
    family=[np.ones(ns)]
    for k in range(1,5):
        family.extend((np.cos(k*theta),np.sin(k*theta)))
    angular=np.broadcast_to(
        np.array(family[:rank]),(surface.n_fields,rank,ns)).copy()
    x0=np.zeros((surface.n_fields,rank,nx))
    x1=np.zeros_like(x0)
    x0[:,:,1:-1]=rng.normal(size=(surface.n_fields,rank,nx-2))*.001
    x1[:,:,1:-1]=rng.normal(size=(surface.n_fields,rank,nx-2))*.001
    old=CoupledMVIFactors(x0,angular)
    proposed=CoupledMVIFactors(x1,angular)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    weight=.25
    output=gauss._relax_gauss_fixed_rank_stage(
        surface,old,proposed,weight,program)
    assert output.rank==rank
    actual=surface.reconstruct(output)
    expected=(1.-weight)*surface.reconstruct(old)+weight*surface.reconstruct(proposed)
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
    np.testing.assert_array_equal(output.x[:,:,0],0.)
    np.testing.assert_array_equal(output.x[:,:,-1],0.)


def test_gauss_relaxation_never_changes_rank():
    surface,_,_=configured()
    x=np.zeros((surface.n_fields,2,surface.grid.nx))
    y=np.ones((surface.n_fields,2,surface.grid.ns))
    old=CoupledMVIFactors(x[:,:1],y[:,:1])
    proposed=CoupledMVIFactors(x,y)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    with pytest.raises(ValueError,match="RANK_CHANGED"):
        gauss._relax_gauss_fixed_rank_stage(
            surface,old,proposed,.25,program)
