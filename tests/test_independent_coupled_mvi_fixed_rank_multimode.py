"""Exact rank-5 and rank-8 synthetic nonzero multi-mode work checks.

This is a deterministic algebraic variational potential with a
known five/eight-mode field, NOT a fabricated physical shell solution.
A ZERO initial axial factor must not destroy the user's nonzero angular
basis. All requested separated slots become nonzero by actual X/Y
iterations. The rank is never reduced; a truncated sweep is not
falsely marked as converged.
"""
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,CoupledMVIWork,
    NativeCoupledMVISurface,mvi_separated_term_norms,
)
from independent_mvi.coupled_static import (
    CoupledMVISweepSettings,solve_coupled_mvi_static,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram,CoupledMVIHalfstepSettings,
)
from flexo_gradient_g3d import G3DMaterial


def setup(rank):
    nx,ns=11,9
    surface=NativeCoupledMVISurface(
        "KL",G3DMaterial(100.,.25,.01,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(nx,ns,1.2,2.),
        thickness_m=.2,n_gauss=2)
    theta=2.*np.pi*np.arange(ns)/ns
    angular=[np.ones(ns)]
    for k in range(1,5):
        angular.extend((np.cos(k*theta),np.sin(k*theta)))
    y=np.broadcast_to(np.asarray(angular[:rank]),
                       (surface.n_fields,rank,ns)).copy()
    x=np.zeros((surface.n_fields,rank,nx))
    desired_x=x.copy()
    u=np.arange(nx)/(nx-1)
    for index in range(rank):
        desired_x[2,index]=.01*np.sin((index+1)*np.pi*u)
    # Enforce the essential mechanical end traces exactly.
    desired_x[:,:,0]=0.
    desired_x[:,:,-1]=0.
    target=np.einsum("cri,crj->cij",desired_x,y,optimize=True)
    seed=CoupledMVIFactors(x,y)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")

    def work(factors):
        field=surface.reconstruct(factors)
        grad=field-target
        return CoupledMVIWork(
            .5*float(np.sum(grad*grad)),field,None,
            np.einsum("cij,crj->cri",grad,factors.y,optimize=True),
            np.einsum("cij,cri->crj",grad,factors.x,optimize=True),
            factors.rank,nodal_gradient=grad)
    return surface,seed,program,target,work


@pytest.mark.parametrize("rank",(5,8))
def test_all_requested_dormant_modes_activate_under_nonzero_multimode_work(rank):
    surface,seed,program,target,work=setup(rank)
    output=solve_coupled_mvi_static(
        surface,seed,program=program,
        field_scales=np.ones(surface.n_fields),energy_scale_j=1.,
        active_fields=(2,),work_evaluator=work,
        settings=CoupledMVISweepSettings(
            max_sweeps=5,relative_field_change=1.e-8,
            halfstep=CoupledMVIHalfstepSettings(
                max_coordinates=96,max_iterations=6,
                linear_solver="DENSE_LU",tolerance_scaled=1.e-8)))
    assert output.converged,output.status
    assert output.factors.rank==rank
    assert output.sweeps>=2
    np.testing.assert_allclose(surface.reconstruct(output),
                               target,atol=2.e-8,rtol=2.e-8)
    contributions=mvi_separated_term_norms(output.factors)
    assert contributions.shape==(surface.n_fields,rank)
    assert np.all(contributions[2]>1.e-10)
    assert all(h["rank"]==rank for h in output.history)


def test_rank_five_finite_sweep_limit_is_not_fake_success():
    surface,seed,program,target,work=setup(5)
    out=solve_coupled_mvi_static(
        surface,seed,program=program,
        field_scales=np.ones(surface.n_fields),energy_scale_j=1.,
        active_fields=(2,),work_evaluator=work,
        settings=CoupledMVISweepSettings(
            max_sweeps=1,
            halfstep=CoupledMVIHalfstepSettings(
                max_coordinates=96,max_iterations=6,
                linear_solver="DENSE_LU")))
    assert not out.converged
    assert out.status=="MVI_FIXED_RANK_SWEEPS_EXHAUSTED"
    assert out.sweeps==1
    assert out.factors.rank==5
