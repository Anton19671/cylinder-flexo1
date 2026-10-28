"""Audit K-2: check the bending-energy limitation without rewriting physics."""
import numpy as np
import pytest
from flexo_gradient_g3d import G3DMaterial, enthalpy_parts
from cylinder_thickness_locking_diagnostic import thickness_locking_diagnostic


@pytest.mark.parametrize("nu", [0., .3, .45])
def test_flat_pure_bending_energy_ratio(nu):
    m = G3DMaterial(100., nu, 0., 0., 0., 0.)
    z, w = np.polynomial.legendre.leggauss(12)
    lam, g = m.lame_lambda, m.shear_modulus
    def energy(ex, ez):
        return enthalpy_parts([ex, 0., ez, 0., 0., 0.],
                              np.zeros((6, 3)), [0., 0., 0.], m)["classical"]
    restricted = sum(weight * energy(zz, 0.) for zz, weight in zip(z, w))
    relaxed = sum(weight * energy(zz, -lam/(lam+2*g)*zz)
                  for zz, weight in zip(z, w))
    assert restricted/relaxed == pytest.approx((1.-nu)**2/(1.-2.*nu), rel=1.e-13)


@pytest.mark.parametrize("theory", ["KL", "FSDT", "REDDY", "CUF_TE1"])
def test_low_order_reports_known_risk(theory):
    d = thickness_locking_diagnostic(theory, .3)
    assert d["thickness_locking"] == "YES"
    assert d["flat_limit_pure_bending_energy_ratio"] == pytest.approx(1.225)
    assert d["plane_stress_reduction_applied"] is False


@pytest.mark.parametrize("theory", ["CUF_TE2", "CUF_TE3", "CUF_TE4"])
def test_higher_order_is_not_falsely_certified(theory):
    d = thickness_locking_diagnostic(theory, .3)
    assert d["thickness_locking"] == "NOT_ESTABLISHED"
    assert d["quantitative_bending_validation_complete"] is False


def test_bad_input_rejected():
    with pytest.raises(ValueError): thickness_locking_diagnostic("CUF_TE8", .3)
    with pytest.raises(ValueError): thickness_locking_diagnostic("KL", .5)


def test_gui_exposes_warning():
    from cylinder_reference_gui import reference_locking_message
    assert "1.225" in reference_locking_message(
        {"theory":"CUF","cuf_model":"TE1","material":{"nu0":.3}})
    assert reference_locking_message(
        {"theory":"CUF","cuf_model":"TE2","material":{"nu0":.3}}) is None
