"""Physical campaign input and comparable field recovery, without solves."""
import numpy as np
import pytest
from ci.matched_sat_physical_pilot import make_case,radial_surfaces
from independent_mkr import run_coupled_research as native

@pytest.mark.parametrize("theory",native.THEORIES)
@pytest.mark.parametrize("gradient",("G3D","Gz"))
def test_matrix_inputs_and_radial_surfaces(tmp_path,theory,gradient):
    path,nm,pe=make_case(tmp_path,theory,gradient,"NEWMARK",n_gauss=12,steps=2)
    a=native.load_case(path)
    assert a["model"].n_gauss==12
    assert a["opts"]["steps"]==2
    assert a["settings"].axisymmetric
    assert a["model"].geometry_model=="GREEN_LAGRANGE"
    q=a["seed"]["mechanical"].copy()
    q[2*(pe+1) if theory.startswith("CUF_TE") else 2]=-.001
    m,o,i=radial_surfaces(a["model"],q,pe)
    np.testing.assert_allclose(m,-.001,atol=1.e-13)
    np.testing.assert_allclose(o,-.001,atol=1.e-13)
    np.testing.assert_allclose(i,-.001,atol=1.e-13)
    if theory.startswith("CUF_TE"):
        q[2*(pe+1)+1]=.0002
        m,o,i=radial_surfaces(a["model"],q,pe)
        np.testing.assert_allclose(m,-.001,atol=1.e-13)
        np.testing.assert_allclose(o,-.0008,atol=1.e-13)
        np.testing.assert_allclose(i,-.0012,atol=1.e-13)
