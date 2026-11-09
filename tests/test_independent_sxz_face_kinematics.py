"""Exact algebraic S_xz = G*gamma_xz checks; no field solve."""
import numpy as np
import pytest
from independent_mkr.linear_models import (
    _kinematic_coefficients as mk_kin,
    _strain_matrices as mk_strain,
    _elastic_tensor as mk_hooke,
)
from independent_mvi.linear_models import (
    _kinematic_coefficients as mv_kin,
    _strain_matrices as mv_strain,
    _elastic_tensor as mv_hooke,
)


@pytest.mark.parametrize("kin,strain,hooke", [
    (mk_kin,mk_strain,mk_hooke),
    (mv_kin,mv_strain,mv_hooke),
])
def test_reddy_sxz_zero_for_every_coefficient_at_both_faces(kin,strain,hooke):
    R,h,E,nu=1.,.05,210.e9,.3
    C=hooke("REDDY",E,nu)
    assert C[4,4]==pytest.approx(E/(2.*(1.+nu)))
    for z in (-h/2.,h/2.):
        c,dz=kin("REDDY",z,R,h)
        B=strain(c,dz,R,z)
        # B is indexed [surface derivative, strain component, unknown].
        # Engineering gamma_xz is strain component 4, across ALL
        # derivative slots. Checking B[4] would accidentally check
        # the mixed x-s derivative rather than transverse shear.
        np.testing.assert_allclose(B[:,4,:],0.,rtol=0.,atol=1.e-12)
        trial=np.random.default_rng(313).normal(size=(B.shape[0],B.shape[2]))
        eps=np.einsum("ijk,ik->j",B,trial)
        sxz=C[4,4]*eps[4]
        assert abs(sxz)<1.e-2  # 210 GPa times floating roundoff


@pytest.mark.parametrize("kin,strain,hooke", [
    (mk_kin,mk_strain,mk_hooke),
    (mv_kin,mv_strain,mv_hooke),
])
def test_fsd_t_sxz_not_falsely_forced_to_zero_on_faces(kin,strain,hooke):
    R,h,E,nu=1.,.05,210.e9,.3
    for z in (-h/2.,h/2.):
        c,dz=kin("FSDT",z,R,h)
        B=strain(c,dz,R,z)
        # An independent axial rotation produces gamma_xz = 1.
        assert B[0,4,3]==pytest.approx(1.)
        assert hooke("FSDT",E,nu)[4,4]==pytest.approx(E/(2*(1+nu)))
