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


def test_failed_reduced_tangent_capture_restores_hook(tmp_path,monkeypatch):
    from ci import native_flexo_loaded_models as campaign
    from independent_mvi import coupled_halfstep as half
    matrix=np.diag([1.,1.e-16])
    gradient=np.ones(2)
    original=half._checked_lu_newton
    def fake_run(*args,**kwargs):
        return half._checked_lu_newton(matrix,gradient,0.)
    monkeypatch.setattr(campaign.mvi,"run",fake_run)
    output=tmp_path/"FAILED_REDUCED_TANGENT.npz"
    with pytest.raises(np.linalg.LinAlgError,match="MVI_ILL_CONDITIONED"):
        campaign.run_with_failed_tangent_capture("unused",output)
    assert half._checked_lu_newton is original
    with np.load(output,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["tangent"],matrix)
        np.testing.assert_array_equal(saved["gradient"],gradient)
