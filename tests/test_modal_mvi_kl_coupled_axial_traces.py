"""Real KL C-C 13x21 exact slope trace and separated-space checks."""
import json
from pathlib import Path
import numpy as np
from cylinder_modal_reference import _modal_spec, prepare_modal_case
from cylinder_modal_axial_traces import (
    uniform_axial_mechanical_traces, axial_trace_xy_basis,
    axial_trace_harmonic_x_basis)
from cylinder_modal_mvi import _SeparatedModalProblem

ROOT=Path(__file__).resolve().parents[1]


def kl_case():
    f=ROOT/"config/verification/rawat2016/modal_linear/mvi_13x13/RAWAT2016_KL_13x13_MODAL_MVI1.json"
    case=json.loads(f.read_text())
    case["mvi_grid"]={"nx":13,"ns":21,"operator_scheme":"SBP42_FOURIER_ODD"}
    case["modal"]["n_modes"]=1
    return case


def test_real_kl_clamp_keeps_exact_end_slopes():
    case=kl_case()
    prepared,density,_=_modal_spec(case)
    bases,fixed,cm=uniform_axial_mechanical_traces(
        prepared.problem,prepared.mechanical)
    assert cm.shape[0]>int(fixed.sum())
    assert len(bases)==3
    assert fixed.sum()>0
    assert prepare_modal_case(case)["method"]=="MVI"
    problem=_SeparatedModalProblem(prepared,density,1)
    assert problem.axial_trace_bases is not None
    assert problem.n_mechanical_free_dofs==sum(b.shape[1] for b in bases)*21
    rng=np.random.default_rng(441)
    x=rng.normal(size=(3,1,13))
    y=rng.normal(size=(3,1,21))
    for axis in ("X","Y"):
        directions=axial_trace_xy_basis(x,y,axis,bases,problem.scales)
        assert directions.shape[1]>0
        relative=np.max(np.abs(cm@directions))/max(1.,np.max(np.abs(directions)))
        assert relative<1.e-9
        if axis=="Y":
            for column in (0,directions.shape[1]-1):
                field=directions[:,column].reshape(3,13,21)
                for c in range(3):
                    for angle in range(21):
                        part=field[c,:,angle]
                        assert np.linalg.norm(part-bases[c]@(bases[c].T@part))<1.e-9


def test_kl_initial_sector_respects_frozen_angular_factors():
    case=kl_case()
    prepared,density,_=_modal_spec(case)
    op=_SeparatedModalProblem(prepared,density,2)
    rng=np.random.default_rng(2026)
    y=rng.normal(size=(3,2,21))
    trial=axial_trace_harmonic_x_basis(
        y,3,op.axial_trace_bases,op.scales)
    assert trial.shape[0]==op.nm
    assert np.max(np.abs(op.cm@trial),initial=0.)<1.e-9


def test_mvi_kl_dispatch_can_build_coupled_x_y_bases():
    case=kl_case()
    prepared,density,_=_modal_spec(case)
    op=_SeparatedModalProblem(prepared,density,1)
    rng=np.random.default_rng(123)
    x=rng.normal(size=(3,1,13))
    y=rng.normal(size=(3,1,21))
    for axis in ("X","Y"):
        bm,_=op.projected_bases(x,y,axis)
        assert bm.shape[1]>0
        assert np.max(np.abs(op.cm@bm),initial=0.)<1.e-9
