"""Independent MVI acceleration closure: dense-small and matrix-free-large.

These are synthetic mass-operator tests, not a loaded shell calculation.
The ONLY solved grid unknown is physical acceleration; the rank-R
displacement factors are unchanged and the strong MKR is never imported.
"""
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,CoupledMVIWork,
    NativeCoupledMVISurface,
)
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_newmark import NativeCoupledMVINewmark
from independent_mvi.coupled_halfstep import CoupledMVIEndProgram
from independent_mvi.coupled_initial import close_mvi_instantaneous_state
from flexo_gradient_g3d import G3DMaterial
import independent_mvi.coupled_initial as initial


def build():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(100.,.23,.02,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.),
        thickness_m=.2,n_gauss=2)
    x=np.zeros((surface.n_fields,1,9))
    y=np.ones((surface.n_fields,1,5))
    factors=CoupledMVIFactors(x,y)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")
    inertia=NativeCoupledMVIInertia(surface,density_kg_m3=2.)
    configuration=NativeCoupledMVINewmark(
        surface,inertia,program=program)
    def fake_work(factors,**kw):
        q=surface.reconstruct(factors)
        g=np.zeros(surface.shape)
        g[:surface.n_mechanical,1:-1]=.2
        return CoupledMVIWork(0.,q,None,
            np.einsum("cij,crj->cri",g,factors.y),
            np.einsum("cij,cri->crj",g,factors.x),
            factors.rank,nodal_gradient=g)
    surface.weak_work=fake_work
    inertia.apply=lambda q,v,a:2.*a
    velocity=np.zeros((surface.n_mechanical,9,5))
    return configuration,factors,velocity


@pytest.mark.parametrize("mode,dense_limit",[
    ("AUTO",20),("MATRIX_FREE_GMRES",20),("DENSE_LU",250),
])
def test_own_initial_acceleration_closes_same_mass_equation(mode,dense_limit):
    configuration,factors,velocity=build()
    solved=close_mvi_instantaneous_state(
        configuration,factors,velocity,time_s=0.,
        field_scales=np.ones(configuration.surface.n_fields),
        energy_scale_j=1.,max_acceleration_coordinates=220,
        acceleration_linear_solver=mode,
        acceleration_dense_limit=dense_limit)
    assert solved.state is not None,solved.status
    q=configuration.surface.reconstruct(solved.state.factors)
    np.testing.assert_array_equal(q,0.)
    np.testing.assert_allclose(
        solved.state.acceleration[:,1:-1],-.1,rtol=0.,atol=1.e-11)
    np.testing.assert_array_equal(solved.state.acceleration[:,0],0.)
    np.testing.assert_array_equal(solved.state.acceleration[:,-1],0.)


def test_large_matrix_free_acceleration_never_calls_dense_solver(monkeypatch):
    configuration,factors,velocity=build()
    def forbidden(*args,**kwargs):
        pytest.fail("Dense LU forbidden in matrix-free MVI mass closure")
    monkeypatch.setattr(initial,"_checked_lu_newton",forbidden)
    solved=close_mvi_instantaneous_state(
        configuration,factors,velocity,time_s=0.,
        field_scales=np.ones(configuration.surface.n_fields),
        energy_scale_j=1.,max_acceleration_coordinates=220,
        acceleration_linear_solver="AUTO",acceleration_dense_limit=20)
    assert solved.state is not None,solved.status


def test_unfinished_matrix_free_acceleration_is_never_accepted(monkeypatch):
    configuration,factors,velocity=build()
    monkeypatch.setattr(initial,"gmres",
                        lambda op,rhs,**kw:(np.zeros_like(rhs),1))
    solved=close_mvi_instantaneous_state(
        configuration,factors,velocity,time_s=0.,
        field_scales=np.ones(configuration.surface.n_fields),
        energy_scale_j=1.,max_acceleration_coordinates=220,
        acceleration_linear_solver="MATRIX_FREE_GMRES",
        acceleration_dense_limit=20)
    assert solved.state is None
    assert solved.status=="MVI_ACCELERATION_KRYLOV_NOT_SOLVED"


def test_dense_explicit_method_rejects_excess_before_allocating_matrix():
    configuration,factors,velocity=build()
    solved=close_mvi_instantaneous_state(
        configuration,factors,velocity,time_s=0.,
        field_scales=np.ones(configuration.surface.n_fields),
        energy_scale_j=1.,max_acceleration_coordinates=220,
        acceleration_linear_solver="DENSE_LU",
        acceleration_dense_limit=20)
    assert solved.state is None
    assert solved.status=="MVI_ACCELERATION_DENSE_METHOD_LIMIT"
