"""No-solve physical inner/outer electric face work and induction tests.

No cylinder loading, nonlinear iteration or time history is executed.
The exact thickness basis, curved face area, physical constitutive
induction and outward normal sign are checked for all seven shell
kinematics and both G3D/Gz material projections.
"""
import json
import numpy as np
import pytest

from independent_mvi.coupled_electric_faces import (
    face_generalized_free_charge,recover_mvi_electric_faces,
    summarize_mvi_electric_faces,
)
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
)
from flexo_gradient_g3d import G3DMaterial


THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")


def case(theory,mode="G3D",potential_order=1):
    surf=NativeCoupledMVISurface(
        theory,G3DMaterial(100.,.25,.0,0.,0.,3.,gradient_model=mode),
        grid=CoupledMVISurfaceGrid(9,5,1.2,2.0),
        thickness_m=.2,n_gauss=2,potential_order=potential_order)
    x=np.zeros((surf.n_fields,1,surf.grid.nx))
    y=np.ones((surf.n_fields,1,surf.grid.ns))
    # phi(x,s,z) = alpha * (2*z/h), with q=0.
    x[surf.n_mechanical+1,0,:]=.12
    return surf,CoupledMVIFactors(x,y)


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("mode",("G3D","GZ"))
def test_physical_face_induction_and_normal_sign_for_seven_theories(theory,mode):
    surf,factors=case(theory,mode)
    # The zero-displacement, linear-through-thickness potential generates
    # E_z=-2*alpha/h and D_z=kappa*E_z.
    expected_dz=-3.*2.*.12/.2
    raw=recover_mvi_electric_faces(surf,factors)
    for side,normal in (("inner",-1),("outer",1)):
        face=raw[side]
        np.testing.assert_allclose(face.induction3_c_per_m2[:,:,2],
                                   expected_dz,atol=2.e-12)
        np.testing.assert_allclose(face.outward_induction_c_per_m2,
                                   normal*expected_dz,atol=2.e-12)
        assert face.condition=="INSULATED"
        np.testing.assert_allclose(face.natural_flux_difference_c_per_m2,
                                   normal*expected_dz,atol=2.e-12)
        assert face.status.startswith("DIAGNOSTIC_ONLY")
        area=(1.+normal*.2/(2*2.))*np.sum(surf.grid.weights)
        assert face.physical_area_m2==pytest.approx(area)
        assert not face.induction3_c_per_m2.flags.writeable
    summary=summarize_mvi_electric_faces(raw)
    assert summary["pointwise_condition_enforced"] is False
    assert summary["physical_face_flux_certified"] is False
    json.dumps(summary)


@pytest.mark.parametrize("theory",THEORIES)
def test_generalized_charge_is_curved_face_area_times_te_basis(theory):
    surf,_=case(theory,potential_order=1)
    inward=2.0
    outward=-.5
    total=face_generalized_free_charge(
        surf,
        inner={"mode":"CHARGE","charge_c_per_m2":inward},
        outer={"mode":"CHARGE","charge_c_per_m2":outward})
    Hm=1.-surf.thickness_m/(2*surf.grid.radius_m)
    Hp=1.+surf.thickness_m/(2*surf.grid.radius_m)
    expected_even=Hm*inward+Hp*outward
    expected_odd=-Hm*inward+Hp*outward
    np.testing.assert_allclose(total[0],expected_even)
    np.testing.assert_allclose(total[1],expected_odd)
    # Positive surface charge is +sigma*phi work regardless of side.
    assert expected_even>0


@pytest.mark.parametrize("theory",THEORIES)
def test_recovered_face_charge_cancels_normal_flux_when_prescribed(theory):
    surf,factors=case(theory)
    dz=-3.*2.*.12/.2
    # Outer n=+1; inner n=-1: sigma=-n*D_z.
    inner={"mode":"CHARGE","charge_c_per_m2":dz}
    outer={"mode":"CHARGE","charge_c_per_m2":-dz}
    report=recover_mvi_electric_faces(
        surf,factors,inner=inner,outer=outer)
    for face in report.values():
        np.testing.assert_allclose(face.natural_flux_difference_c_per_m2,
                                   0.,atol=2.e-12)


@pytest.mark.parametrize("side",("inner","outer"))
def test_unsupported_face_potential_is_rejected(side):
    surf,factors=case("CUF_TE1")
    faces=dict(inner=None,outer=None)
    faces[side]={"mode":"POTENTIAL","charge_c_per_m2":None}
    with pytest.raises(ValueError,match="DIRICHLET_POTENTIAL_NOT_IMPLEMENTED"):
        face_generalized_free_charge(surf,**faces)
    with pytest.raises(ValueError,match="DIRICHLET_POTENTIAL_NOT_IMPLEMENTED"):
        recover_mvi_electric_faces(surf,factors,**faces)


def test_nonuniform_charge_field_is_accepted_in_physical_face_units():
    surf,_=case("KL")
    rng=np.random.default_rng(511)
    field=rng.normal(size=(surf.grid.nx,surf.grid.ns))
    total=face_generalized_free_charge(
        surf,inner={"mode":"CHARGE","charge_c_per_m2":field})
    np.testing.assert_allclose(
        total[0],(1.-surf.thickness_m/(2.*surf.grid.radius_m))*field)
    np.testing.assert_allclose(
        total[1],-(1.-surf.thickness_m/(2.*surf.grid.radius_m))*field)


def test_face_charge_mode_requires_finite_scalar_or_field():
    surf,_=case("REDDY")
    for value in (True,float("nan"),float("inf"),[[1.],[2.]]):
        with pytest.raises(ValueError,match="FACE_CHARGE_"):
            face_generalized_free_charge(
                surf,outer={"mode":"CHARGE","charge_c_per_m2":value})
