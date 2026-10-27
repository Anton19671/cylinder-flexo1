"""Shell-to-face integration checks for an independent modal electric operator.

The independent face scheme has its own spatial quadrature. The exact
comparison with the established weak solver is restricted to a dielectric
state constant in x, which deliberately shares the same periodic forward
face gradient as the legacy reference. Nonzero mechanical fields need not
produce an identical nodal action: their polarization collocation differs.
"""
import numpy as np
import pytest

from cylinder_electric_modal_faces import (
    modal_electric_face_weak, shell_flexopolarization_face_samples)
from cylinder_functional_assembly import (
    CylinderFunctional, MECHANICAL_DERIVATIVES,
    ELECTRICAL_DERIVATIVES, surface_trapezoid)
from flexo_gradient_g3d import G3DMaterial, enthalpy_derivatives


@pytest.mark.parametrize('model',('KL','FSDT','REDDY',
                                  'CUF_TE1','CUF_TE2','CUF_TE3','CUF_TE4'))
@pytest.mark.parametrize('gradient_model',('G3D','Gz'))
def test_all_shell_models_modal_dielectric_matches_legacy_weak(model,gradient_model):
    n_gauss=(2*int(model[-1])+1 if model.startswith('CUF_TE')
             else 7 if model=='REDDY' else 5)
    nm=3
    n_gauss=max(n_gauss,5)
    op=surface_trapezoid(4,6,1.2,2.)
    material=G3DMaterial(100.,.25,.08,.001,-.002,1.5,
                         gradient_model=gradient_model)
    f=CylinderFunctional(model,material,radius_m=2.,thickness_m=.2,
                         surface_operators=op,n_gauss=n_gauss)
    q=np.zeros((f.n_mechanical,4,6))
    j=np.arange(6)
    p=np.zeros((nm,4,6))
    for m in range(nm):
        p[m]=.03*(m+1)*np.cos(2*np.pi*(m+1)*j/6)[None,:]
    ref=f.evaluate(q,p,energy_breakdown=False)
    face=modal_electric_face_weak(
        p,length_m=1.2,radius_m=2.,thickness_m=.2,
        kappa_f_per_m=1.5,n_gauss=n_gauss)
    np.testing.assert_allclose(face.electrical_residual_c,
                               ref.electrical_residual,rtol=2e-9,atol=2e-10)
    np.testing.assert_allclose(face.electric_potential_enthalpy_j,
                               ref.total_enthalpy_j,rtol=2e-9,atol=2e-10)


@pytest.mark.parametrize('gradient_model',('G3D','Gz'))
def test_nonlinear_shell_polarization_is_sampled_from_same_material(gradient_model):
    op=surface_trapezoid(4,6,1.2,2.)
    material=G3DMaterial(100.,.25,.08,.002,-.001,1.5,
                         gradient_model=gradient_model)
    f=CylinderFunctional('CUF_TE1',material,radius_m=2.,thickness_m=.2,
                         surface_operators=op,n_gauss=5)
    rng=np.random.default_rng(725)
    q=1e-3*rng.normal(size=(f.n_mechanical,4,6))
    samples=shell_flexopolarization_face_samples(f,q,n_gauss=6)
    assert samples.axial_c_per_m2.shape==(6,3,6)
    assert samples.circumferential_c_per_m2.shape==(6,4,6)
    assert samples.transverse_c_per_m2.shape==(6,4,6)
    qjets=f._field_derivatives(q,MECHANICAL_DERIVATIVES[:f.n_derivatives])
    pjets=f._field_derivatives(np.zeros((1,4,6)),ELECTRICAL_DERIVATIVES)
    i,j=1,2
    jets=f._make_jets(qjets,pjets,i,j)
    for z,expected in ((-.1,samples.inner_transverse_c_per_m2[i,j]),
                       (.1,samples.outer_transverse_c_per_m2[i,j]),
                       (0.,None)):
        point=f._point(jets,z)
        _,_,induction=enthalpy_derivatives(
            point.strain6,point.gradient6x3,np.zeros(3),material)
        if expected is not None:
            np.testing.assert_allclose(expected,induction[2],rtol=1e-12,atol=1e-14)
    p=np.zeros((4,4,6))
    p[0]=.01*np.cos(2*np.pi*np.arange(6)/6)
    p[1]=.005
    r=modal_electric_face_weak(p,length_m=1.2,radius_m=2.,
        thickness_m=.2,kappa_f_per_m=1.5,n_gauss=6,polarization=samples)
    assert np.isfinite(r.electrical_residual_c).all()
    assert np.isfinite(r.inner_outward_induction_c_per_m2).all()
    assert np.isfinite(r.outer_outward_induction_c_per_m2).all()
