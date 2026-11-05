"""Check physical equivalence and truly non-axisymmetric patch loading.

No time-dependent physics calculation is executed by this test.
"""
from __future__ import annotations

import numpy as np

from ci.generate_cuf_te4_nonaxis_bathe import experiment_cases
from cylinder_reference_case import prepare_reference_case
from cylinder_variational_loads import project_radial_load


def test_patch_inputs_are_identical_except_solver_choices():
    cases=experiment_cases(steps=1)
    assert sorted(cases)==["FDM","MVI1","MVI8"]
    for name,case in cases.items():
        assert case["calculation_kind"]=="DYNAMIC"
        assert case["geometry_model"]=="LINEAR"
        assert case["dynamics"]["integrator"]=="BATHE"
        assert case["dynamics"]["time_step_s"]==.002
        assert case["dynamics"]["time_law"]["frequency_hz"]==25
        assert case["load_definition"]["q"]==100000
        assert case["load_definition"]["type"]=="PATCH_RADIAL_FACE"
        assert case["load_definition"]["full_circumference"] is False
        assert case["load_definition"]["x1"]>0
        assert case["load_definition"]["x2"]<case["geometry"]["L"]
        if name=="MVI8":
            assert case["reference_mvi"]["dynamic_krylov_halfsteps"] is True
        elif name=="MVI1":
            assert not case["reference_mvi"].get("dynamic_krylov_halfsteps",False)


def test_patch_is_nonaxisymmetric_on_actual_nine_by_nine_solver_grid():
    case=experiment_cases(steps=1)["FDM"]
    prepared=prepare_reference_case(case)
    projection=project_radial_load(prepared.problem.functional,case["load_definition"])
    p=projection.face_pressure_pa
    assert p.shape==(9,9)
    assert np.isfinite(p).all()
    assert np.any(p<0)
    assert np.any(p==0)
    assert np.linalg.matrix_rank(p,tol=1.e-10)>=1
    assert np.max(np.ptp(p,axis=1))>1000
    assert np.max(np.ptp(p,axis=0))>1000
    # The pressure must genuinely force more than just one Fourier
    # circumferential harmonic, despite using a separable patch indicator.
    harmonics=np.fft.rfft(p,axis=1)
    amplitude=np.linalg.norm(harmonics,axis=0)
    assert np.count_nonzero(amplitude>1.e-6*np.max(amplitude))>=3
