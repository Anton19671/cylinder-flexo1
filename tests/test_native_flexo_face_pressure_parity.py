"""Regression: every native seven-theory physical face pressure uses H(+h/2).

No mechanical equation is solved in this test. It prevents accidental
comparison of different physical loads under identical nominal pressure.
"""
from pathlib import Path

import numpy as np
import pytest

from ci.native_flexo_loaded_models import (
    THEORIES, GRADIENTS, PRESSURE, H, R, NX, NS,
    e_order, n_mech, case_and_npz,
)


@pytest.mark.parametrize("theory", THEORIES)
@pytest.mark.parametrize("gradient", GRADIENTS)
def test_outer_face_pressure_same_physical_load_across_theories(
    tmp_path, theory, gradient
):
    folder=tmp_path / (theory+"_"+gradient)
    folder.mkdir()
    case_file, case, forcing=case_and_npz(folder,theory,gradient)
    assert case_file.is_file()
    assert forcing.shape == (n_mech(theory),NX,NS)
    expected=-PRESSURE*(1.+H/(2.*R))
    nonzero=np.flatnonzero(np.max(np.abs(forcing),axis=(1,2)))
    indices=(list(range(2*(e_order(theory)+1),3*(e_order(theory)+1)))
             if theory.startswith("CUF_TE") else [2])
    assert nonzero.tolist()==indices
    np.testing.assert_allclose(forcing[indices],expected,rtol=0.,atol=1.e-14)
    with np.load(folder/"load.npz",allow_pickle=False) as data:
        np.testing.assert_allclose(data["mechanical_load"],forcing)
    assert case["physics"]["theory"]==theory
    assert case["physics"]["material"]["gradient_model"]==gradient
