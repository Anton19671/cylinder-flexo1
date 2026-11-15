"""Pure algebraic matrix-free MVI large-halfstep tests.

Synthetic positive quadratic frozen-partner work; no physical cylinder
state, loaded benchmark, MKR solver, or timestep calculation.
"""
import numpy as np
import pytest

import independent_mvi.coupled_halfstep as half
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
)
from flexo_gradient_g3d import G3DMaterial


def setup():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(80.,.23,.03,0.,0.,0.),
        grid=CoupledMVISurfaceGrid(11,9,1.4,2.),
        thickness_m=.2,n_gauss=2)
    rng=np.random.default_rng(211)
    x=np.zeros((surface.n_fields,2,surface.grid.nx))
    x[:,:,1:-1]=rng.normal(0.,.3,size=x[:,:,1:-1].shape)
    y=rng.normal(0.,.3,size=(surface.n_fields,2,surface.grid.ns))
    factors=CoupledMVIFactors(x,y)
    program=half.CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","OFF")

    def quadratic_work(factors,**kw):
        return type("QuadraticWork",(),{
            "gradient_x":factors.x,
            "gradient_y":factors.y,
        })()
    surface.weak_work=quadratic_work
    return surface,factors,program


@pytest.mark.parametrize("axis",("X","Y"))
def test_matrix_free_krylov_solves_own_factor_equations_without_dense_lu(
        axis,monkeypatch):
    surf,fac,program=setup()
    def forbidden(*a,**kw):
        raise AssertionError("dense LU called on matrix-free MVI halfstep")
    monkeypatch.setattr(half,"_checked_lu_newton",forbidden)
    settings=half.CoupledMVIHalfstepSettings(
        max_coordinates=300,linear_solver="MATRIX_FREE_GMRES",
        max_iterations=3,difference_step=1.e-6)
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis=axis,program=program,
        field_scales=np.ones(surf.n_fields),
        energy_scale_j=1.,settings=settings)
    assert out.converged
    assert out.factors.rank==fac.rank
    assert out.history[-1]["linear_solver"]=="MATRIX_FREE_GMRES"
    if axis=="X":
        np.testing.assert_allclose(out.factors.x,0.,atol=2.e-8)
    else:
        np.testing.assert_allclose(out.factors.y,0.,atol=2.e-8)
    assert not np.any(out.factors.x[:,:,0])
    assert not np.any(out.factors.x[:,:,-1])


def test_auto_uses_matrix_free_above_small_dense_threshold(monkeypatch):
    surf,fac,program=setup()
    monkeypatch.setattr(half,"_checked_lu_newton",
                        lambda *a,**kw:pytest.fail("unexpected dense path"))
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis="Y",program=program,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        settings=half.CoupledMVIHalfstepSettings(
            max_coordinates=300,dense_coordinate_limit=3))
    assert out.converged
    assert out.history[-1]["linear_solver"]=="MATRIX_FREE_GMRES"


def test_nonconverged_krylov_never_accepts_partial_correction(monkeypatch):
    surf,fac,program=setup()
    def no_convergence(operator,rhs,**kwargs):
        return np.zeros_like(rhs),1
    monkeypatch.setattr(half,"gmres",no_convergence)
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis="Y",program=program,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        settings=half.CoupledMVIHalfstepSettings(
            max_coordinates=300,linear_solver="MATRIX_FREE_GMRES"))
    assert not out.converged
    assert out.status=="MVI_HALFSTEP_TANGENT_NOT_SOLVED"
    np.testing.assert_array_equal(out.factors.x,fac.x)
    np.testing.assert_array_equal(out.factors.y,fac.y)


def test_invalid_krylov_controls_rejected():
    with pytest.raises(ValueError,match="LINEAR_SOLVER"):
        half.CoupledMVIHalfstepSettings(linear_solver="SVD")
    with pytest.raises(ValueError,match="KRYLOV"):
        half.CoupledMVIHalfstepSettings(krylov_relative_tolerance=-1.)
    with pytest.raises(ValueError,match="KRYLOV_RESTART"):
        half.CoupledMVIHalfstepSettings(krylov_restart=0)



def test_failed_reduced_krylov_records_actual_reason_and_preserves_original(monkeypatch):
    surf,fac,program=setup()
    def no_convergence(operator,rhs,**kwargs):
        return np.zeros_like(rhs),3
    monkeypatch.setattr(half,"gmres",no_convergence)
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis="Y",program=program,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        settings=half.CoupledMVIHalfstepSettings(
            max_coordinates=300,linear_solver="MATRIX_FREE_GMRES"))
    assert not out.converged
    assert out.status=="MVI_HALFSTEP_TANGENT_NOT_SOLVED"
    assert out.history[-1]["linear_failure"]=="MVI_GMRES_NOT_CONVERGED"
    assert out.history[-1]["linear_dimension"]>0
    assert out.history[-1]["linear_solver"]=="MATRIX_FREE_GMRES"
    np.testing.assert_array_equal(out.factors.x,fac.x)
    np.testing.assert_array_equal(out.factors.y,fac.y)



@pytest.mark.parametrize("axis",("X","Y"))
def test_opt_in_krylov_breakdown_recovers_from_own_checked_reduced_lu(axis,monkeypatch):
    surf,fac,program=setup()
    def forced_stagnation(operator,rhs,**kwargs):
        return np.zeros_like(rhs),7
    monkeypatch.setattr(half,"gmres",forced_stagnation)
    settings=half.CoupledMVIHalfstepSettings(
        max_coordinates=300,max_iterations=3,
        linear_solver="MATRIX_FREE_GMRES",
        krylov_dense_fallback_limit=300)
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis=axis,program=program,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        settings=settings)
    assert out.converged
    assert out.factors.rank==fac.rank
    assert any(h.get("recovered") is True and
               h.get("krylov_failure")=="MVI_GMRES_NOT_CONVERGED"
               for h in out.history)
    assert any(h.get("linear_solver")=="DENSE_LU_AFTER_GMRES"
               for h in out.history)
    if axis=="X":
        np.testing.assert_allclose(out.factors.x,0.,atol=2.e-8)
    else:
        np.testing.assert_allclose(out.factors.y,0.,atol=2.e-8)
    assert not np.any(out.factors.x[:,:,0])
    assert not np.any(out.factors.x[:,:,-1])


def test_opt_in_lu_failure_fails_closed_and_keeps_original_rank(monkeypatch):
    surf,fac,program=setup()
    def forced_stagnation(operator,rhs,**kwargs):
        return np.zeros_like(rhs),7
    monkeypatch.setattr(half,"gmres",forced_stagnation)
    def singular_lu(*args,**kwargs):
        raise np.linalg.LinAlgError("SYNTHETIC_ILL_CONDITIONED_REDUCED_TANGENT")
    monkeypatch.setattr(half,"_checked_lu_newton",singular_lu)
    out=half.solve_coupled_mvi_halfstep(
        surf,fac,axis="X",program=program,
        field_scales=np.ones(surf.n_fields),energy_scale_j=1.,
        settings=half.CoupledMVIHalfstepSettings(
            max_coordinates=300,linear_solver="MATRIX_FREE_GMRES",
            krylov_dense_fallback_limit=300))
    assert not out.converged
    assert out.status=="MVI_HALFSTEP_TANGENT_NOT_SOLVED"
    assert out.history[-1]["krylov_failure"]=="MVI_GMRES_NOT_CONVERGED"
    assert out.history[-1]["linear_failure"]=="SYNTHETIC_ILL_CONDITIONED_REDUCED_TANGENT"
    assert out.factors.rank==fac.rank
    np.testing.assert_array_equal(out.factors.x,fac.x)
    np.testing.assert_array_equal(out.factors.y,fac.y)


def test_default_and_invalid_fallback_bounds_preserve_fail_closed_behavior():
    assert half.CoupledMVIHalfstepSettings().krylov_dense_fallback_limit==0
    for bad in (-1,1.0,True,"300"):
        with pytest.raises(ValueError,match="KRYLOV_DENSE_FALLBACK_LIMIT"):
            half.CoupledMVIHalfstepSettings(krylov_dense_fallback_limit=bad)


@pytest.mark.parametrize("axis",("X","Y"))
def test_explicit_dense_reduced_mvi_newton_has_no_gmres_or_fdm(axis,monkeypatch):
    surf,fac,program=setup()
    def forbidden_gmres(*args,**kwargs):
        raise AssertionError("explicit reduced LU must not call GMRES")
    monkeypatch.setattr(half,"gmres",forbidden_gmres)
    result=half.solve_coupled_mvi_halfstep(
        surf,fac,axis=axis,program=program,
        field_scales=np.ones(surf.n_fields),
        energy_scale_j=1.,
        settings=half.CoupledMVIHalfstepSettings(
            max_coordinates=300,linear_solver="DENSE_LU",
            krylov_dense_fallback_limit=0,max_iterations=3))
    assert result.converged
    assert result.factors.rank==fac.rank
    assert result.history[-1]["linear_solver"]=="DENSE_LU"
    if axis=="X":
        np.testing.assert_allclose(result.factors.x,0.,atol=2.e-8)
    else:
        np.testing.assert_allclose(result.factors.y,0.,atol=2.e-8)
