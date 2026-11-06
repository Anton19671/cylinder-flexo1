"""Independent analytic constitutive verification for cylinder shells.

Pure algebra and low-order tensor assembly only: no physical time stepping.
The reference plane-stress tensor is the Schur complement of isotropic 3D
elasticity with sigma_zz eliminated; CUF retains 3D thickness stretch.
"""
import numpy as np
import pytest

from independent_mkr.linear_models import (
    StrongModelCase, IndependentStrongModel,
    _elastic_tensor as mkrelastic,
    _kinematic_coefficients as mkrkin,
    _strain_matrices as mkrstrains,
)
from independent_mvi.linear_models import (
    VariationalModelCase, IndependentVariationalModel,
    _elastic_tensor as mvielastic,
    _kinematic_coefficients as mvikin,
    _strain_matrices as mvistrains,
)

CLASSICAL=("KL","FSDT","REDDY")
CUF=("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
THEORIES=CLASSICAL+CUF
ROUTES=(
    (StrongModelCase, IndependentStrongModel, mkrelastic, mkrkin, mkrstrains),
    (VariationalModelCase, IndependentVariationalModel, mvielastic, mvikin, mvistrains),
)


def independent_3d_tensor(E, nu):
    mu=E/(2*(1+nu))
    lam=E*nu/((1+nu)*(1-2*nu))
    out=np.diag([2*mu,2*mu,2*mu,mu,mu,mu])
    out[:3,:3]+=lam
    return out


@pytest.mark.parametrize("case_cls,model_cls,elastic,kin,strains",ROUTES)
@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("nu",(0., .22, .3, .48))
def test_shell_constitutive_is_analytic_schur_complement(
    case_cls,model_cls,elastic,kin,strains,theory,nu
):
    E=210.e9
    actual=elastic(theory,E,nu)
    c3=independent_3d_tensor(E,nu)
    assert actual.shape==(6,6)
    np.testing.assert_allclose(actual,actual.T,rtol=0,atol=0)
    np.testing.assert_allclose(actual[3:,3:],np.eye(3)*E/(2*(1+nu)),
                               rtol=2.e-15,atol=1.e-6)
    if theory in CLASSICAL:
        active=[0,1,3,4,5]
        reference=(c3[np.ix_(active,active)]
                   -np.outer(c3[active,2],c3[2,active])/c3[2,2])
        np.testing.assert_allclose(actual[np.ix_(active,active)],reference,
                                   rtol=1.e-14,atol=1.e-3)
        np.testing.assert_array_equal(actual[2],np.zeros(6))
        np.testing.assert_array_equal(actual[:,2],np.zeros(6))
        case=case_cls(theory=theory,poisson=nu)
        for z in (-case.thickness_m/2,0.,case.thickness_m/2):
            c,dz=kin(theory,z,case.radius_m,case.thickness_m)
            b=strains(c,dz,case.radius_m,z)
            np.testing.assert_array_equal(b[:,2,:],np.zeros_like(b[:,2,:]))
        strain=np.array([.001,-.002,0.,.003,-.004,.005])
        assert (actual@strain)[2]==0.
    else:
        np.testing.assert_allclose(actual,c3,rtol=2.e-15,atol=1.e-3)
        assert actual[2,2]>0.
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("case_cls,model_cls,elastic,kin,strains",ROUTES)
@pytest.mark.parametrize("theory",THEORIES)
def test_axial_membrane_stiffness_has_exact_thickness_moment(
    case_cls,model_cls,elastic,kin,strains,theory
):
    case=case_cls(theory=theory,poisson=.3)
    model=(model_cls(case) if model_cls is IndependentStrongModel
           else model_cls(case,rank=1))
    # The axial-only u_0,x coefficient is one through thickness;
    # integral (1+z/R) dz = h exactly.
    expected=elastic(theory,case.young_pa,case.poisson)[0,0]*case.thickness_m
    np.testing.assert_allclose(model.stiffness[1,1,0,0],expected,
                               rtol=3.e-12,atol=1.e-3)


@pytest.mark.parametrize("case_cls,model_cls,elastic,kin,strains",ROUTES)
@pytest.mark.parametrize("theory",CLASSICAL)
def test_classical_shell_stiffness_uses_plane_stress_not_3d_lambda(
    case_cls,model_cls,elastic,kin,strains,theory
):
    case=case_cls(theory=theory,poisson=.3)
    model=(model_cls(case) if model_cls is IndependentStrongModel
           else model_cls(case,rank=1))
    expected=case.young_pa*case.thickness_m/(1-case.poisson**2)
    full3d=independent_3d_tensor(case.young_pa,case.poisson)[0,0]*case.thickness_m
    actual=model.stiffness[1,1,0,0]
    np.testing.assert_allclose(actual,expected,rtol=3.e-12,atol=1.e-3)
    assert abs(actual-full3d)/expected>.1
