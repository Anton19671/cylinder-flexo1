"""Independent seven-model MKR nodal energy and axial end adjoints.

These tests construct small algebraic operators only. They do NOT run
a loaded shell simulation, physical time step, or MVI computation.
Source: Krysko-Zhong corrected 2026-10-24, §§39–41.
"""
import numpy as np
import pytest

from independent_mkr.linear_models import (
    MODELS, IndependentStrongModel, StrongModelCase, ORDERS,
)


def _own_discrete_work(model, trial, virtual, tensor, terms):
    """Independent scalar construction: derivative of MKR's discrete energy."""
    wx=model.quadrature_x[None,:,None]
    a=[model._differentiate(virtual,j) for j in range(terms)]
    b=[model._differentiate(trial,j) for j in range(terms)]
    return sum(float(np.sum(wx*a[i]*np.einsum(
        "ab,bxs->axs",tensor[i,j],b[j],optimize=True)))
        for i in range(terms) for j in range(terms))


@pytest.mark.parametrize("theory",MODELS)
@pytest.mark.parametrize("kind",("stiffness","mass"))
def test_mkr_operator_is_exact_discrete_energy_first_variation(theory,kind):
    model=IndependentStrongModel(StrongModelCase(theory=theory,nx=11,ns=9))
    rng=np.random.default_rng(20261107)
    trial=model.unpack(1.e-7*rng.standard_normal(sum(model.dofs)))
    virtual=model.unpack(1.e-7*rng.standard_normal(sum(model.dofs)))
    tensor=(model.stiffness if kind=="stiffness" else model.mass)
    terms=(6 if kind=="stiffness" else 3)
    action=(model.stiffness_action if kind=="stiffness" else model.mass_action)
    expected=_own_discrete_work(model,trial,virtual,tensor,terms)
    actual=float(np.sum(virtual*action(trial)))
    reverse=float(np.sum(trial*action(virtual)))
    # The two independently assembled contractions must agree.
    np.testing.assert_allclose(actual,expected,rtol=4.e-11,atol=2.e-8)
    np.testing.assert_allclose(actual,reverse,rtol=4.e-11,atol=2.e-8)
    assert float(np.sum(trial*action(trial)))>0.


@pytest.mark.parametrize("theory",("KL","REDDY"))
def test_mkr_axial_bending_uses_own_adjoint_not_naive_d2_squared(theory):
    model=IndependentStrongModel(StrongModelCase(theory=theory,nx=11,ns=9))
    B=model.basis[2]             # radial displacement/free axial coordinates
    D2=model.deriv_x[2]          # two successive MKR differences
    w=model.quadrature_x
    correct=B.T@D2.T@np.diag(w)@D2@B
    old_strong=B.T@(D2@D2)@B
    np.testing.assert_allclose(correct,correct.T,rtol=3.e-14,atol=1.e-9)
    assert float(np.linalg.norm(old_strong-old_strong.T))>1.e-3
    assert float(np.linalg.norm(correct-old_strong))>1.e-3


def test_mkr_own_mass_has_nonzero_energy_for_alternating_angular_velocity():
    """The nodal zero-order kinetic term cannot be cell-averaged away."""
    model=IndependentStrongModel(StrongModelCase(theory="CUF_TE4",nx=11,ns=9))
    velocity=np.zeros((model.n,11,9))
    velocity[10,1:-1,:]=(-1.)**np.arange(9)[None,:]
    kinetic=float(np.sum(velocity*model.mass_action(velocity)))
    assert kinetic>0.


def test_discrete_mkr_operators_do_not_import_mvi_or_reference_backend():
    import ast
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"independent_mkr"/"linear_models.py"
    root=ast.parse(path.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(root):
        if isinstance(node,ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(value.startswith((
        "independent_mvi","numerics","mechanics","cylinder_",
        "cuf_g3d","flexoelectric")) for value in imports)
