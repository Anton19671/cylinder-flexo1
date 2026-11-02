"""Independent, calculation-free tests for Cammalleri plane-stress diagnostic.

Only the isolated Cammalleri diagnostic uses the Schur-reduced material;
the production KL material law and the 1e-6 modal certification stay intact.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import cylinder_modal_mvi
from cylinder_modal_mvi import _kl_linear_local_tangent
from flexo_gradient_g3d import G3DMaterial
import modal_cammalleri_plane_stress_diagnostic as diagnostic


def _material():
    return G3DMaterial(207e9, .3, 0., 0., 0., 0.)


def _full_elastic_matrix(material):
    lam, shear = material.lame_lambda, material.shear_modulus
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] += 2 * shear
    c[3, 3] = c[4, 4] = c[5, 5] = shear
    return c


def _functional(material=None, **extra):
    kwargs = dict(R=.076, h=.000254, n_gauss=9,
                  material=material or _material(), theory="KL",
                  geometry_model="LINEAR", electrical_on=False)
    kwargs.update(extra)
    return SimpleNamespace(**kwargs)


def test_plane_stress_schur_is_exact_constitutive_elimination():
    material = _material()
    c = _full_elastic_matrix(material)
    q = diagnostic.reduced_plane_stress_matrix(material)
    active = [0, 1, 3, 4, 5]
    for strain2 in ([.013, -.007, .02, 0., 0.],
                    [-.001, .004, -.009, .003, .011],
                    [.003, .004, -.005, .002, -.012]):
        reduced = np.asarray(strain2, float)
        full = np.zeros(6)
        full[active] = reduced
        full[2] = -(c[2, active] @ reduced) / c[2, 2]
        stress = c @ full
        assert abs(stress[2]) < 1.e-5  # Pa; exactly sigma_zz=0
        np.testing.assert_allclose(
            stress[active], q[np.ix_(active,active)] @ reduced,
            rtol=5.e-14, atol=5.e-6)
        np.testing.assert_allclose(
            full @ c @ full, reduced @ q[np.ix_(active,active)] @ reduced,
            rtol=5.e-14, atol=5.e-6)
    np.testing.assert_allclose(q[0,0],207e9/(1-.3**2),rtol=1.e-13)
    np.testing.assert_allclose(q[0,1],.3*207e9/(1-.3**2),rtol=1.e-13)
    assert np.max(np.abs(q[2]))==0


def test_plane_stress_variational_kl_hessian_is_symmetric_and_no_stiffer():
    f = _functional()
    three_dimensional = _kl_linear_local_tangent(f)
    plane_stress = diagnostic.plane_stress_local_tangent(f)
    assert plane_stress.shape == three_dimensional.shape == (30,30)
    assert np.isfinite(plane_stress).all()
    np.testing.assert_allclose(plane_stress,plane_stress.T,rtol=1.e-12,atol=1.e-7)
    rng = np.random.default_rng(20261102)
    for _ in range(12):
        v = rng.normal(size=30)
        e3 = float(v @ three_dimensional @ v)
        ep = float(v @ plane_stress @ v)
        assert ep >= -1.e-9*max(abs(e3),1.)
        assert ep <= e3 + 1.e-9*max(abs(e3),1.)


@pytest.mark.parametrize("updates",[
    {"theory":"FSDT"}, {"geometry_model":"GREEN_LAGRANGE"},
    {"electrical_on":True},
    {"material":G3DMaterial(207e9,.3,1.e-6,0.,0.,0.)},
])
def test_diagnostic_refuses_models_without_plane_stress_derivation(updates):
    with pytest.raises(ValueError,match="REQUIRES_LINEAR_MACRO_KL"):
        diagnostic.plane_stress_local_tangent(_functional(**updates))


def test_diagnostic_restores_production_tangent_on_exception(monkeypatch):
    initial=cylinder_modal_mvi._kl_linear_local_tangent
    def injected_failure(case):
        assert cylinder_modal_mvi._kl_linear_local_tangent is diagnostic.plane_stress_local_tangent
        raise RuntimeError("intentional-test-solver-failure")
    monkeypatch.setattr(diagnostic,"solve_modal_case",injected_failure)
    with pytest.raises(RuntimeError,match="intentional-test-solver-failure"):
        diagnostic.run_case({"modal":{}},13,21,"plane_stress_diagnostic")
    assert cylinder_modal_mvi._kl_linear_local_tangent is initial
