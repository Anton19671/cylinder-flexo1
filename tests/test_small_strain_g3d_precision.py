"""Precision regression for the physical CUF/KL/FSDT/Reddy GL kernels.

Analytic homogeneous simple shear and a prescribed shear gradient are
evaluated locally; no spatial shell solve or time history is executed.
"""
import numpy as np
import pytest
from cylinder_paths import activate
activate()
from cuf_g3d_weak import _strain_from_deformation_jets
from cylinder_kl_g3d_weak import _strain as kl_strain
from cylinder_director_g3d import _strain as director_strain


@pytest.mark.parametrize("kernel",(_strain_from_deformation_jets,kl_strain,director_strain))
@pytest.mark.parametrize("gamma",(1.e-6,1.e-8,-1.e-8,1.e-10))
def test_small_simple_shear_retains_quadratic_normal_strain(kernel,gamma):
    f=np.eye(3)
    f[0,1]=gamma
    fx=np.zeros((3,3))
    fx[0,1]=2.*gamma
    zero=np.zeros((3,3))
    strain,derivatives=kernel(f,fx,zero,zero)
    expected=np.array([0.,.5*gamma**2,0.,gamma,0.,0.])
    dx=np.array([0.,2.*gamma**2,0.,2.*gamma,0.,0.])
    np.testing.assert_allclose(strain,expected,rtol=1.e-14,atol=0.)
    np.testing.assert_allclose(derivatives[0],dx,rtol=1.e-14,atol=0.)
    np.testing.assert_array_equal(derivatives[1],0.)
    np.testing.assert_array_equal(derivatives[2],0.)


@pytest.mark.parametrize("kernel",(_strain_from_deformation_jets,kl_strain,director_strain))
def test_finite_simple_shear_keeps_full_green_lagrange_terms(kernel):
    gamma=.35
    f=np.eye(3)
    f[0,1]=gamma
    zero=np.zeros((3,3))
    strain,_=kernel(f,zero,zero,zero)
    np.testing.assert_allclose(strain,[0.,.5*gamma**2,0.,gamma,0.,0.],
                               rtol=1.e-14,atol=0.)
