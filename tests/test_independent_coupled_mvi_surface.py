"""Algebra-only checks for independent nonlinear coupled MVI surface weak work.

No loaded shell, no time series, no Newton solve and no GitHub Actions run.
Synthetic local energy avoids using a physical solution as its own benchmark.
"""
from types import SimpleNamespace
from pathlib import Path
import ast
import numpy as np
import pytest

from flexo_gradient_g3d import G3DMaterial
from independent_mvi.coupled_surface import (
    THEORIES, CoupledMVISurfaceGrid, CoupledMVIFactors,
    NativeCoupledMVISurface,
)


def setup(theory="CUF_TE1",electric=False,gradient="G3D",rank=2):
    grid=CoupledMVISurfaceGrid(9,5,1.2,2.)
    material=G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                         -.002 if electric else 0.,.8 if electric else 0.,gradient)
    model=NativeCoupledMVISurface(
        theory,material,grid=grid,thickness_m=.3,n_gauss=2,
        potential_order=1 if electric else None)
    rng=np.random.default_rng(20261109)
    x=rng.normal(0,.01,(model.n_fields,rank,grid.nx))
    y=rng.normal(0,.01,(model.n_fields,rank,grid.ns))
    return model,CoupledMVIFactors(x,y)


def _fake_local(model,kind):
    """Independent synthetic local scalar functional and exact derivatives.

    Includes the once-only mixed derivative x-s and a negative electric
    quadratic term, with optional reciprocal mechanical/electric coupling.
    """
    def evaluate(jets):
        n=model.n_mechanical
        p=model.n_electric
        q=np.asarray(jets.q).reshape(n)
        qxs=np.asarray(jets.qxs).reshape(n)
        m=np.zeros((n,len(model.mechanical_multi)))
        e=np.zeros((p,3))
        energy=.5*float(q@q)+.2*float(qxs@qxs)
        m[:,0]=q
        m[:,4]=.4*qxs
        if p:
            psi=np.asarray(jets.psi)
            psix=np.asarray(jets.psix)
            energy-=.35*float(psi@psi)
            energy+=.12*float(psix@psix)
            energy+=.07*float(q[0]*psi[0])
            m[0,0]+=.07*psi[0]
            e[:,0]=-.7*psi
            e[:,1]=.24*psix
            e[0,0]+=.07*q[0]
        return SimpleNamespace(energy_j_per_m2=energy,mechanical=m,electrical=e)
    return evaluate


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","GZ"))
def test_all_seven_theories_own_ranked_field_shapes(theory,gradient):
    model,fac=setup(theory,gradient=gradient)
    assert model.reconstruct(fac).shape==model.shape
    xd,yd,jets=model._surface_derivatives(fac)
    for order in model.mechanical_multi:
        assert jets[order].shape==model.shape
    assert xd[3].shape==fac.x.shape
    assert yd[3].shape==fac.y.shape


@pytest.mark.parametrize("electric",(False,True))
@pytest.mark.parametrize("theory",("KL","FSDT","REDDY","CUF_TE1","CUF_TE4"))
def test_independent_mvi_weak_work_equals_direct_factor_variation(
        theory,electric):
    model,fac=setup(theory,electric=electric)
    model._local=_fake_local(model,"synthetic")
    rng=np.random.default_rng(17)
    dm=rng.normal(size=(model.n_mechanical,model.grid.nx,model.grid.ns))*.02
    de=(rng.normal(size=(model.n_electric,model.grid.nx,model.grid.ns))*.01
        if electric else None)
    result=model.weak_work(fac,mechanical_load=dm,free_charge=de)
    for family in ("X","Y"):
        direction=rng.normal(size=getattr(fac,family.lower()).shape)
        direction/=np.linalg.norm(direction)
        h=1.e-6
        def varied(step):
            x=fac.x.copy()
            y=fac.y.copy()
            if family=="X":
                x+=step*direction
            else:
                y+=step*direction
            return model.weak_work(CoupledMVIFactors(x,y),
                    mechanical_load=dm,free_charge=de).energy_j
        numerical=(varied(h)-varied(-h))/(2*h)
        analytic=float(np.sum(
            (result.gradient_x if family=="X" else result.gradient_y)*direction))
        np.testing.assert_allclose(analytic,numerical,atol=2.e-8,rtol=1.e-6)


def test_electricity_off_never_accepts_free_charge():
    model,fac=setup()
    model._local=_fake_local(model,"synthetic")
    with pytest.raises(ValueError,match="OFF"):
        model.weak_work(fac,free_charge=np.zeros((1,9,5)))


def test_rank_capacity_and_factors_are_immutable():
    model,fac=setup(rank=2)
    assert not fac.x.flags.writeable
    assert not fac.y.flags.writeable
    x=fac.x.copy()
    with pytest.raises(ValueError,match="rank"):
        model.weak_work(CoupledMVIFactors(np.zeros((model.n_fields,6,9)),
                                              np.zeros((model.n_fields,6,5))))
    np.testing.assert_array_equal(fac.x,x)


def test_material_switch_is_local_not_method_substitution():
    a,_=setup(gradient="G3D")
    b,_=setup(gradient="GZ")
    assert a.material.gradient_model=="G3D"
    assert b.material.gradient_model=="GZ"
    assert a.grid.dx.shape==b.grid.dx.shape


def test_no_mkr_or_shared_spatial_operator_imports():
    src=(Path(__file__).resolve().parents[1]/
         "independent_mvi/coupled_surface.py").read_text(encoding="utf8")
    imports=[]
    for node in ast.walk(ast.parse(src)):
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node,ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert not any(name.startswith((
        "independent_mkr","cylinder_mkr_","cylinder_direct_strong_",
        "cylinder_variational_fdm","cylinder_functional_assembly",
        "cylinder_sbp_derivatives","cylinder_surface_linear_algebra"))
        for name in imports)
