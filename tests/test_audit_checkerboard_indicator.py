"""Audit Zh-12: checkerboard is detected from potential, not grid parity."""
import numpy as np
import pytest
from cylinder_electric_face_flux import dielectric_face_flux_fdm

def evaluate(phi):
    return dielectric_face_flux_fdm(phi,length_m=1.2,radius_m=2.,
                                    thickness_m=.2,kappa_f_per_m=1.5)

@pytest.mark.parametrize("ns",[4,6,8,12,32])
def test_nyquist_present(ns):
    p=np.broadcast_to((-1.)**np.arange(ns),(5,ns)).copy()
    a=evaluate(p)
    assert a.circumferential_checkerboard_detected
    assert a.circumferential_checkerboard_energy_fraction==pytest.approx(1.,abs=1e-14)
    assert evaluate(p+1200.).circumferential_checkerboard_energy_fraction==pytest.approx(1.,abs=1e-14)

@pytest.mark.parametrize("ns",[4,6,8,12,32])
def test_even_grid_without_nyquist(ns):
    p=np.full((5,ns),2.3)
    assert not evaluate(p).circumferential_checkerboard_detected
    sine=np.sin(2*np.pi*np.arange(ns)/ns)
    a=evaluate(np.broadcast_to(sine,(5,ns)).copy())
    assert not a.circumferential_checkerboard_detected
    assert a.circumferential_checkerboard_energy_fraction<1e-20

def test_odd_grid_has_no_nyquist():
    a=evaluate(np.random.default_rng(4).normal(size=(5,7)))
    assert not a.circumferential_checkerboard_detected
    assert a.circumferential_checkerboard_energy_fraction==0.
