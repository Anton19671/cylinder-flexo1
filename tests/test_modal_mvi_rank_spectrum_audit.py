"""Non-solver checks of MVI modal rank and spectrum selection.

No cylinder equations are assembled and no eigenproblem is solved here.
"""
import copy

import numpy as np
import pytest

from cylinder_modal_reference import (
    _component_separation_ranks, _use_harmonic_mvi_spectrum,
)


def _case():
    return {"method": "MVI", "scale_model": "MACRO",
            "mvi_grid": {"nx": 13, "ns": 13,
                         "operator_scheme": "SBP42_FOURIER_ODD"},
            "boundary_conditions": {"variational": {
                "mechanical": {
                    "left": {"support": "CLAMPED", "higher": "NATURAL"},
                    "right": {"support": "CLAMPED", "higher": "NATURAL"}},
                "electrical": {}}},
            "modal": {"n_modes": 8, "density_kg_m3": 7850.0}}


def test_auto_keeps_classical_unrestricted_mvi():
    assert not _use_harmonic_mvi_spectrum(_case(), 8)
    assert not _use_harmonic_mvi_spectrum(_case(), 9)


def test_explicit_unrestricted_never_forces_fourier_modes():
    case = _case()
    case["modal"]["mvi_spectrum_method"] = "UNRESTRICTED"
    assert not _use_harmonic_mvi_spectrum(case, 8)


def test_legacy_fourier_flag_is_not_a_second_solver():
    # A saved old CASE can retain this flag, but the dispatcher does not
    # call the Fourier-family solver on that basis.
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/"numerics"/
            "cylinder_modal_reference.py").read_text(encoding="utf-8")
    routing=source.split("def solve_modal_case(",1)[1].split(
        "def _component_separation_ranks(",1)[0]
    assert "harmonic_scan=False" in routing
    assert "harmonic_scan=_use_harmonic_mvi_spectrum" not in routing


def test_fourier_family_selection_requires_physical_symmetry():
    case = _case()
    case["modal"]["mvi_spectrum_method"] = "FOURIER_FAMILIES"
    assert _use_harmonic_mvi_spectrum(case, 8)
    other = copy.deepcopy(case)
    other["scale_model"] = "MICRO"
    assert not _use_harmonic_mvi_spectrum(other, 8)
    other = copy.deepcopy(case)
    other["boundary_conditions"]["variational"]["electrical"] = {
        "left": {"support": "FIXED"}}
    assert not _use_harmonic_mvi_spectrum(other, 8)


def test_effective_rank_can_be_less_than_selected_mvi_rank():
    x = np.linspace(-1, 1, 9)
    theta = 2.0 * np.pi * np.arange(13) / 13
    fields = np.zeros((2, 2, 9, 13), dtype=float)
    fields[0, 0] = np.outer(x**2, np.cos(3 * theta))
    fields[1, 0] = (np.outer(x**2, np.cos(3 * theta)) +
                    np.outer(x**3, np.sin(4 * theta)))
    assert _component_separation_ranks(fields) == [[1, 0], [2, 0]]


def test_effective_rank_rejects_bad_shapes():
    with pytest.raises(ValueError, match="MODAL_RANK_AUDIT_INVALID_FIELDS"):
        _component_separation_ranks(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="MODAL_RANK_AUDIT_INVALID_FIELDS"):
        _component_separation_ranks(np.full((1, 1, 2, 3), np.nan))



@pytest.mark.parametrize("nx,ns,nc,rank,allowed",[
    (13,13,15,1,True), (13,13,15,8,True),
    (17,17,15,1,True), (21,63,15,1,True),
    (21,63,15,8,False),
])
def test_unrestricted_mvi_grid_gate_before_dense_assembly(nx,ns,nc,rank,allowed):
    from cylinder_modal_reference import _check_unrestricted_modal_mvi_bounds
    if allowed:
        _check_unrestricted_modal_mvi_bounds(nx,ns,nc,rank)
    else:
        with pytest.raises(ValueError,match="MODAL_MVI_UNRESTRICTED_DIRECTION_LIMIT"):
            _check_unrestricted_modal_mvi_bounds(nx,ns,nc,rank)



def test_rank_capacity_accounts_for_electric_modes():
    from cylinder_modal_reference import _modal_rank_capacity_audit
    axial=np.linspace(0.,1.,9)
    circum=np.linspace(0.,2.*np.pi,13,endpoint=False)
    mechanics=np.zeros((1,1,9,13))
    mechanics[0,0]=np.outer(axial,np.cos(circum))
    electric=np.zeros((1,1,9,13))
    electric[0,0]=(np.outer(axial,np.cos(circum))+
                   np.outer(axial**2,np.sin(circum)))
    report=_modal_rank_capacity_audit(mechanics,electric,2)
    assert report["mechanical_ranks"]==[[1]]
    assert report["electrical_ranks"]==[[2]]
    assert report["mechanical_max"]==[1]
    assert report["all_fields_max"]==[2]
    assert report["rank_capacity_unused"]==[False]
    no_electric=_modal_rank_capacity_audit(mechanics,None,2)
    assert no_electric["rank_capacity_unused"]==[True]
