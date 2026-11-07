"""Algebra-only geometric center sampling checks for isolated MKR and MVI."""
import numpy as np
import pytest

from independent_mkr.run_models import _central_grid_value as mkr_center
from independent_mvi.run_models import _central_grid_value as mvi_center


@pytest.mark.parametrize("sample", [mkr_center, mvi_center])
@pytest.mark.parametrize("nx,ns", [(9, 8), (10, 9), (10, 8), (9, 9), (11, 5)])
def test_geometric_center_for_affine_axial_and_even_theta_field(sample, nx, ns):
    # The exporter uses piecewise linear periodic theta interpolation.
    # For odd ns, this differs from the exact trigonometric point value.
    x=np.linspace(0., 2., nx)[:, None]
    theta=(2.*np.pi*np.arange(ns)/ns)[None, :]
    field=3.*x+2.*np.cos(theta)+7.
    expected=8. if ns%2==0 else 10.+2.*np.cos(np.pi-np.pi/ns)
    assert sample(field)==pytest.approx(expected, abs=1.e-12)


@pytest.mark.parametrize("sample", [mkr_center, mvi_center])
def test_point_sample_not_nearest_node_on_even_axial_grid(sample):
    field=np.arange(10., dtype=float)[:, None]*np.ones((1,9))
    assert sample(field)==pytest.approx(4.5)


@pytest.mark.parametrize("sample", [mkr_center, mvi_center])
def test_invalid_shape_rejected(sample):
    with pytest.raises(ValueError, match="CENTRAL_SAMPLE_REQUIRES_2D_FIELD"):
        sample(np.arange(9))
