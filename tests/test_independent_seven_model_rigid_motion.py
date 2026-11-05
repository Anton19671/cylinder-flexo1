"""Independent geometric objectivity checks without any shell simulation.

A Cartesian in-plane rigid translation of a cylindrical shell is
V0(theta)=-A*sin(theta), W0(theta)=A*cos(theta), all other surface
coefficients zero. At every thickness point, all six *linear* strains
must vanish for KL, FSDT, Reddy and CUF TE1-4. Fourier angular
differentiation reproduces this identity on the odd periodic grid;
centered angular finite differences have a known nonzero consistency
error. These checks expose missing curvature and crossed derivatives.
"""
import numpy as np
import pytest
from independent_mkr.linear_models import (
    MODELS,StrongModelCase,IndependentStrongModel
)
from independent_mvi.linear_models import (
    VariationalModelCase,IndependentVariationalModel
)


def _rigid_translation(model):
    c=model.case
    theta=2*np.pi*np.arange(c.ns)/c.ns
    n=model.n
    q=np.zeros((n,c.nx,c.ns))
    if c.theory in ("KL","FSDT","REDDY"):
        tangential,radial=1,2
    else:
        order=int(c.theory[-1])+1
        tangential,radial=order,2*order
    q[tangential]=-np.sin(theta)[None,:]*1.e-6
    q[radial]=np.cos(theta)[None,:]*1.e-6
    return q


def _unweighted_stiffness_energy(model,q):
    derivatives=[model._differentiate(q,k) for k in range(6)]
    # For a constant homogeneous material, tensor is the independent
    # 12-point thickness integral of the exact strain-to-stress map.
    value=sum(float(np.sum(
        derivatives[i]*np.einsum(
            "ab,bxs->axs",model.stiffness[i,j],derivatives[j],
            optimize=True)
    )) for i in range(6) for j in range(6))
    return value


@pytest.mark.parametrize("theory",MODELS)
def test_seven_linear_kinematics_are_objective_for_in_plane_rigid_shift(theory):
    cm=StrongModelCase(theory=theory,nx=11,ns=9)
    cv=VariationalModelCase(theory=theory,nx=11,ns=9)
    mkr=IndependentStrongModel(cm)
    mvi=IndependentVariationalModel(cv,rank=1)
    qfd=_rigid_translation(mkr)
    qvi=_rigid_translation(mvi)
    np.testing.assert_array_equal(qfd,qvi)
    vi_energy=_unweighted_stiffness_energy(mvi,qvi)
    fd_energy=_unweighted_stiffness_energy(mkr,qfd)
    # Evaluate an explicit non-rigid radial wave as a dimensional scale.
    radial=(2 if theory in ("KL","FSDT","REDDY") else
            2*(int(theory[-1])+1))
    comparison=np.zeros_like(qvi)
    comparison[radial]=1.e-6*np.cos(
        2*np.pi*np.arange(cm.ns)/cm.ns)[None,:]
    typical=_unweighted_stiffness_energy(mvi,comparison)
    assert typical>0 and np.isfinite(typical)
    assert abs(vi_energy)<1.e-20*typical, (
        "MVI_WRONG_CYLINDRICAL_LINEAR_RIGID_TRANSLATION",theory,
        vi_energy,typical)
    # The MKR angular derivative is a deliberately *different*, second
    # order centered stencil. It cannot represent sin/cos exactly at
    # ns=9; its nonzero rigid-shift energy is a spatial truncation effect.
    assert fd_energy>1.e-8*typical, (
        "MKR_SILENTLY_USES_SPECTRAL_WEAK_OPERATOR",theory,fd_energy)
    assert fd_energy<typical, (
        "MKR_RIGID_TRANSLATION_DISCRETIZATION_BROKEN",theory,fd_energy)


@pytest.mark.parametrize("theory",MODELS)
def test_seven_model_axial_uniform_translation_is_strain_free(theory):
    mkr=IndependentStrongModel(StrongModelCase(theory=theory))
    mvi=IndependentVariationalModel(
        VariationalModelCase(theory=theory),rank=1)
    for model in (mkr,mvi):
        field=np.zeros((model.n,model.case.nx,model.case.ns))
        field[0]=1.e-6
        reference=np.zeros_like(field)
        radial=(2 if theory in ("KL","FSDT","REDDY") else
                2*(int(theory[-1])+1))
        reference[radial]=1.e-6
        energy=_unweighted_stiffness_energy(model,field)
        reference_energy=_unweighted_stiffness_energy(model,reference)
        assert reference_energy>0
        assert abs(energy)<=1.e-22*reference_energy
