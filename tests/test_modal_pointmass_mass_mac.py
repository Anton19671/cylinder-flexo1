"""Mass-MAC metric regression for the Zhong point-mass modal audit."""
import numpy as np
import pytest
from modal_mvi_te1_pointmass_tolerance_audit import mass_weighted_mac


def test_mass_mac_uses_squared_normalized_overlap_not_sqrt_mac():
    m = np.diag([2.0, 5.0])
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 1.0])
    assert mass_weighted_mac(lambda x: m @ x, u, v) == pytest.approx(2.0 / 7.0)


def test_mass_mac_is_one_for_rescaled_identical_mode_and_zero_for_orthogonal():
    m = np.diag([2.0, 5.0])
    u = np.array([1.0, 0.0])
    assert mass_weighted_mac(lambda x: m @ x, u, -3.0 * u) == pytest.approx(1.0)
    assert mass_weighted_mac(lambda x: m @ x, u, np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_mass_mac_rejects_bad_mass():
    with pytest.raises(ValueError, match="NONPOSITIVE"):
        mass_weighted_mac(lambda x: -x, [1.0, 0.0], [1.0, 0.0])
