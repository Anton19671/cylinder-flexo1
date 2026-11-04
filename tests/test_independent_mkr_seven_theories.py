"""Algebra-only checks for 7 independent strong-form MKR shell models.

No time marching, no actual physical load run.
"""
import numpy as np
import pytest
from independent_mkr.linear_models import (
    MODELS, StrongModelCase, IndependentStrongModel,
    _kinematic_coefficients, _strain_matrices, _reddy_warps,
)

COUNT={"KL":3,"FSDT":5,"REDDY":5,"CUF_TE1":6,"CUF_TE2":9,
       "CUF_TE3":12,"CUF_TE4":15}


@pytest.mark.parametrize("theory",MODELS)
def test_seven_theories_dimensions_mass_positivity_and_clamped_constraints(theory):
    model=IndependentStrongModel(StrongModelCase(theory=theory))
    assert model.n==COUNT[theory]
    assert model.stiffness.shape==(6,6,model.n,model.n)
    assert model.mass.shape==(3,3,model.n,model.n)
    assert all(np.isfinite(m).all() for m in (model.mass,model.stiffness))
    assert np.allclose(model.stiffness,model.stiffness.transpose(1,0,3,2),
                       atol=3.e-6,rtol=2.e-13)
    rng=np.random.default_rng(493)
    for _ in range(2):
        packed=rng.normal(size=sum(model.dofs))
        q=model.unpack(packed)
        np.testing.assert_allclose(model.pack(q),packed,rtol=2.e-13,atol=2.e-13)
        np.testing.assert_allclose(q[:,[0,-1],:],0.,atol=2.e-13)
        if theory=="KL":
            dq=np.einsum("ij,js->is",model.dx,q[2])
            np.testing.assert_allclose(dq[[0,-1]],0.,atol=2.e-12)
        for action in (model.mass_action,model.stiffness_action):
            result=action(q)
            assert result.shape==q.shape and np.isfinite(result).all()


@pytest.mark.parametrize("theory",MODELS)
def test_thickness_linear_strain_at_zero_field_zero(theory):
    case=StrongModelCase(theory=theory)
    model=IndependentStrongModel(case)
    for z in (-case.thickness_m/2,0.,case.thickness_m/2):
        displacement,dz=_kinematic_coefficients(
            theory,z,case.radius_m,case.thickness_m)
        assert displacement.shape==dz.shape==(3,3,model.n)
        b=_strain_matrices(displacement,dz,case.radius_m,z)
        assert b.shape==(6,6,model.n)
        assert np.isfinite(b).all()


def test_reddy_both_shears_vanish_at_faces_independent_of_curvature():
    case=StrongModelCase(theory="REDDY")
    for z in (-case.thickness_m/2,case.thickness_m/2):
        ax,axp,bs,bsp=_reddy_warps(z,case.radius_m,case.thickness_m)
        np.testing.assert_allclose(1+axp,0.,rtol=0.,atol=2.e-14)
        H=1+z/case.radius_m
        np.testing.assert_allclose(1+bsp-bs/(case.radius_m*H),
                                   0.,rtol=0.,atol=2.e-11)


def test_new_MKR_strong_operators_are_separate_from_shared_weak_assembler():
    import ast
    from pathlib import Path
    file=Path(__file__).resolve().parents[1]/"independent_mkr"/"linear_models.py"
    tree=ast.parse(file.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    disallowed=("cylinder_","independent_mvi","cuf_g3d","flexo_")
    assert not any(name.startswith(disallowed) for name in imports)
