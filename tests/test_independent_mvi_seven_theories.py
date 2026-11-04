"""Seven independent MVI shell theories: algebraic tests, no physical solves."""
import ast
from pathlib import Path

import numpy as np
import pytest

from independent_mvi.linear_models import (
    MODELS,VariationalModelCase,IndependentVariationalModel,
    _kinematic_coefficients,_strain_matrices,_reddy_warps)

COUNTS={"KL":3,"FSDT":5,"REDDY":5,"CUF_TE1":6,"CUF_TE2":9,
        "CUF_TE3":12,"CUF_TE4":15}


@pytest.mark.parametrize("theory",MODELS)
def test_mvi_theory_operator_mass_factors_boundary_constraints(theory):
    c=VariationalModelCase(theory=theory)
    model=IndependentVariationalModel(c,rank=1)
    assert model.n==COUNTS[theory]
    assert model.stiffness.shape==(6,6,model.n,model.n)
    assert model.mass.shape==(3,3,model.n,model.n)
    rng=np.random.default_rng(55481)
    a=rng.normal(size=(model.n,c.nx,c.ns))*1.e-6
    b=rng.normal(size=a.shape)*1.e-6
    for k,B in enumerate(model.basis):
        a[k]=B@(B.T@a[k])
        b[k]=B@(B.T@b[k])
    np.testing.assert_allclose(
        sum(np.sum(a*model.stiffness_action(b)) for _ in [0]),
        np.sum(b*model.stiffness_action(a)),rtol=2.e-11,atol=1.e-7)
    np.testing.assert_allclose(
        np.sum(a*model.mass_action(b)),np.sum(b*model.mass_action(a)),
        rtol=1.e-11,atol=1.e-10)
    assert np.sum(a*model.mass_action(a))>0
    assert np.sum(a*model.stiffness_action(a))>0
    X,Y=model._first_factors()
    assert X.shape==(model.n,1,c.nx)
    assert Y.shape==(model.n,1,c.ns)
    assert np.max(np.abs(X[:,:,[0,-1]]))<2.e-13
    for frozen in ("X","Y"):
        XX,YY=model._normalize(X,Y,frozen)
        np.testing.assert_allclose(model.field(XX,YY),model.field(X,Y),
                                   atol=2.e-13)
        T,widths=model._directions(XX,YY,"Y" if frozen=="X" else "X")
        assert T.shape==(model.n*c.nx*c.ns,sum(widths))
    if theory=="KL":
        test=model.unpack(rng.standard_normal(sum(model.dofs)))
        edge=np.einsum("ij,js->is",model.dx,test[2])[[0,-1]]
        np.testing.assert_allclose(edge,0.,atol=2.e-12)


@pytest.mark.parametrize("theory",MODELS)
def test_mvi_seven_models_have_physical_kinematic_dimensions(theory):
    c=VariationalModelCase(theory=theory)
    for z in (-c.thickness_m/2,0,c.thickness_m/2):
        disp,dz=_kinematic_coefficients(theory,z,c.radius_m,c.thickness_m)
        assert disp.shape==dz.shape==(3,3,COUNTS[theory])
        mat=_strain_matrices(disp,dz,c.radius_m,z)
        assert mat.shape==(6,6,COUNTS[theory])
        assert np.isfinite(mat).all()


def test_mvi_reddy_shear_zero_at_both_surfaces():
    c=VariationalModelCase(theory="REDDY")
    for z in (-c.thickness_m/2,c.thickness_m/2):
        ax,axp,bs,bsp=_reddy_warps(z,c.radius_m,c.thickness_m)
        assert abs(1+axp)<1.e-12
        H=1+z/c.radius_m
        assert abs(1+bsp-(z+bs)/(c.radius_m*H))<2.e-11


def test_kl_rank8_cannot_be_falsely_enabled_on_small_clamped_grid():
    c=VariationalModelCase(theory="KL",nx=11,ns=9)
    with pytest.raises(ValueError,match="MVI_RANK_EXCEEDS_CLAMPED_FIELD_CAPACITY"):
        IndependentVariationalModel(c,rank=8)
    model=IndependentVariationalModel(
        VariationalModelCase(theory="KL",nx=15,ns=9),rank=8)
    assert model.rank==8



@pytest.mark.parametrize("theory",MODELS)
def test_all_seven_theories_accept_eight_terms_on_clamped_15_by_9_grid(theory):
    case=VariationalModelCase(theory=theory,nx=15,ns=9)
    model=IndependentVariationalModel(case,rank=8)
    x,y=model._first_factors()
    assert x.shape==(model.n,8,15)
    assert y.shape==(model.n,8,9)
    assert np.max(np.abs(x[:,:,[0,-1]]))<1.e-13
    if theory=="KL":
        edge=np.einsum("ij,crj->cri",model.dx[[0,-1]],x[2:3])
        np.testing.assert_allclose(edge,0.,atol=2.e-12)
    field=model.field(x,y)
    assert field.shape==(model.n,15,9)
    assert np.allclose(field,0.)
    # Frozen X rank-eight space remains one-dimensional in Y per factor.
    _,widths=model._directions(x,y,"Y")
    assert sum(widths)==model.n*8*9

def test_mvi_never_imports_mkr_or_legacy_functional():
    p=Path(__file__).resolve().parents[1]/"independent_mvi"/"linear_models.py"
    tree=ast.parse(p.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(s.startswith(("independent_mkr","cylinder_","cuf_","flexo_"))
                   for s in imports)
