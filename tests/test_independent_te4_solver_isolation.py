"""No shell simulation: certify isolation and discrete algebra for TE4 prototypes.

The test uses operator actions and manufactured nodal fields, not time stepping.
It does not claim independent physical validation of either implementation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from independent_mkr.linear_te4 import PhysicalCase, StrongDifferenceTE4
from independent_mvi.linear_te4 import VariationalCase, VariationalTE4

ROOT=Path(__file__).resolve().parents[1]


def test_solver_sources_do_not_import_each_other_or_common_variational_backend():
    forbidden=("cylinder_functional_assembly",
               "cylinder_variational_dynamic", "cylinder_variational_mvi",
               "cylinder_variational_fdm", "cylinder_boundary_spatial",
               "cylinder_bathe_mvi", "cuf_g3d_weak",
               "independent_mkr", "independent_mvi")
    for folder in ("independent_mkr","independent_mvi"):
        file=ROOT/folder/"linear_te4.py"
        tree=ast.parse(file.read_text(encoding="utf-8"),filename=str(file))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imports.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:
                imports.append(node.module)
        for name in imports:
            assert not name.startswith(forbidden), (file,name)


def test_independent_fd_and_variational_angular_derivatives_act_on_columns():
    nx,ns=9,9
    cfd=PhysicalCase(nx=nx,ns=ns)
    cmvi=VariationalCase(nx=nx,ns=ns)
    fdm=StrongDifferenceTE4(cfd)
    mvi=VariationalTE4(cmvi,rank=1)
    theta=2*np.pi*np.arange(ns)/ns
    u=np.zeros((15,nx,ns))
    u[10]=np.cos(theta)[None,:]
    # Both implementations act with D_s on the angular (last) dimension,
    # not D_s^T. Their derivative approximations are deliberately different.
    expected_fd=np.einsum("ij,cxj->cxi",fdm.ds,u)
    expected_vi=np.einsum("ij,cxj->cxi",mvi.ds,u)
    np.testing.assert_allclose(fdm._s(u),expected_fd,atol=1.e-15)
    np.testing.assert_allclose(mvi._s(u),expected_vi,atol=1.e-15)
    target=-np.sin(theta)/cfd.radius_m
    np.testing.assert_allclose(mvi._s(u)[10,3],target,atol=1.e-12)
    assert np.max(np.abs(fdm._s(u)[10,3]-target))<.1
    assert np.linalg.norm(fdm.ds-mvi.ds)>1.e-3


def test_independent_weak_te4_stiffness_is_symmetric_in_weighted_energy():
    mvi=VariationalTE4(VariationalCase(),rank=1)
    rng=np.random.default_rng(91641)
    a=rng.normal(size=(15,9,9))*1.e-5
    b=rng.normal(size=(15,9,9))*1.e-5
    for q in (a,b):
        q[:,0,:]=0
        q[:,-1,:]=0
    lhs=float(np.sum(a*mvi.weak_stiffness(b)))
    rhs=float(np.sum(b*mvi.weak_stiffness(a)))
    np.testing.assert_allclose(lhs,rhs,rtol=2.e-11,atol=1.e-9)
    assert float(np.sum(a*mvi.weak_stiffness(a)))>0


def test_mkr_uses_strong_derivative_not_discrete_weak_adjoint():
    fdm=StrongDifferenceTE4(PhysicalCase())
    mvi=VariationalTE4(VariationalCase(),rank=1)
    # Their first derivative operators are intentionally different.
    assert not np.allclose(fdm.dx,mvi.dx)
    assert not np.allclose(fdm.ds,mvi.ds)
    # MKR uses a pointwise strong divergence, MVI uses quadrature-weighted
    # variational adjoints. They cannot silently share the discrete operator.
    rng=np.random.default_rng(118)
    q=rng.normal(size=(15,9,9))*1.e-8
    q[:,0,:]=q[:,-1,:]=0
    a=fdm.elastic_strong_action(q)
    b=mvi.weak_stiffness(q)
    assert a.shape==b.shape==q.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.allclose(a,b,rtol=1.e-12,atol=1.e-12)


def test_exact_rank_two_cannot_be_expressed_as_one_MVI_factor():
    nx=ns=9
    x=np.linspace(0.,1.,nx)
    theta=2*np.pi*np.arange(ns)/ns
    axial=np.stack((np.sin(np.pi*x),np.sin(2*np.pi*x)))
    angular=np.stack((np.cos(theta),np.sin(theta)))
    q=np.outer(axial[0],angular[0])+.4*np.outer(axial[1],angular[1])
    singular=np.linalg.svd(q,compute_uv=False)
    assert singular[1]>0.2*singular[0]
    model=VariationalTE4(VariationalCase(),rank=1)
    f1,f2=model._initial()
    assert f1.shape==(15,1,nx) and f2.shape==(15,1,ns)
    assert np.linalg.matrix_rank(model._field(f1,f2)[10])==0
    best_rank_one_error=singular[1]
    assert best_rank_one_error>1.


def test_rank_eight_requires_more_than_nine_axial_nodes_for_clamped_ends():
    with pytest.raises(ValueError,match="MVI_RANK_EXCEEDS_CLAMPED_SPATIAL_CAPACITY"):
        VariationalTE4(VariationalCase(nx=9,ns=9),rank=8)
    model=VariationalTE4(VariationalCase(nx=11,ns=9),rank=8)
    assert model.rank==8
