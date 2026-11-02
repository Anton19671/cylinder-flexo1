"""Regression for cancellation in the linear KL local modal tangent."""
from types import SimpleNamespace
import numpy as np
from cylinder_modal_mvi import _kl_linear_local_tangent
from cylinder_kl_g3d_weak import KLSurfaceJets, FIELDS, thickness_weak_coefficients
from flexo_gradient_g3d import G3DMaterial


def test_thin_shell_linear_tangent_has_reciprocal_bending_coupling():
    material = G3DMaterial(207e9, .3, 0., 0., 0., 0.)
    f = SimpleNamespace(R=.076, h=.000254, n_gauss=9, material=material)
    tangent = _kl_linear_local_tangent(f)
    # Entries 20,25 are tiny relative to the membrane terms, so a global
    # norm alone would conceal the loss of precision that broke Fourier modes.
    assert abs(tangent[20, 25] - tangent[25, 20]) < 1.e-9
    assert abs(tangent[20, 25] - 65.8807430376) < 1.e-7

    # Independently compare one column with the existing weak-force code;
    # the larger perturbation avoids F-I cancellation in this diagnostic.
    jets = [np.zeros(3) for _ in FIELDS]
    jets[5][2] = .01
    electric = [np.zeros(1) for _ in range(3)]
    def force(sign):
        return thickness_weak_coefficients(
            KLSurfaceJets(*[sign*x for x in jets], *electric), material,
            radius_m=f.R, thickness_m=f.h, n_gauss=f.n_gauss,
            geometry_model="LINEAR").mechanical.ravel()
    fd = (force(1.) - force(-1.)) / .02
    np.testing.assert_allclose(tangent[:, 25], fd, rtol=1.e-8, atol=3.e-6)
