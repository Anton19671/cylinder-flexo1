"""Modal operator normalization must agree across dense and sparse paths."""
import copy

import pytest

from cylinder_modal_reference import _effective_modal_case


@pytest.mark.parametrize(
    "method,key,original,expected",
    [
        ("MVI", "mvi_grid", "SBP21_PERIODIC_ODD", "SBP42_FOURIER_ODD"),
        ("FDM", "fdm_grid", "SBP21_PERIODIC_ODD_CSR", "SBP42_FOURIER_ODD_CSR"),
        ("FDM", "fdm_grid", "SBP42_FOURIER_ODD_CSR", "SBP42_FOURIER_ODD_CSR"),
    ],
)
def test_modal_normalizes_legacy_operators_without_mutating_case(method, key, original, expected):
    case = {"method": method, key: {"nx": 13, "ns": 13, "operator_scheme": original}}
    saved = copy.deepcopy(case)
    assert _effective_modal_case(case)[key]["operator_scheme"] == expected
    assert case == saved


@pytest.mark.parametrize("scheme", ["SBP21_PERIODIC_ODD", "SBP21_PERIODIC_ODD_CSR"])
def test_modal_rejects_grid_too_small_for_automatic_upgrade(scheme):
    key = "fdm_grid" if scheme.endswith("_CSR") else "mvi_grid"
    case = {"method": "FDM" if key == "fdm_grid" else "MVI",
            key: {"nx": 8, "ns": 9, "operator_scheme": scheme}}
    with pytest.raises(ValueError, match="nx >= 9"):
        _effective_modal_case(case)

def test_saved_modal_legacy_operator_and_its_solver_operator_agree():
    from cylinder_reference_method_switch import selected_case
    original = {
        "solver_backend": "VARIATIONAL_REFERENCE",
        "calculation_kind": "MODAL", "scale_model": "MACRO",
        "method": "MVI", "rank": 3, "fdm_backend": "DENSE_REFERENCE",
        "mvi_grid": {"nx": 13, "ns": 13, "operator_scheme": "SBP21_PERIODIC_ODD"},
        "reference_mvi": {"max_sweeps": 45},
        "modal": {"density_kg_m3": 7850, "n_modes": 8},
    }
    copy_before=copy.deepcopy(original)
    switched=selected_case(original, "FDM")
    assert switched["fdm_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD_CSR"
    assert switched["modal"] == original["modal"]
    assert "rank" not in switched and "reference_mvi" not in switched
    back=selected_case(switched, "MVI", 3)
    assert back["mvi_grid"]["operator_scheme"] == "SBP42_FOURIER_ODD"
    assert back["modal"] == original["modal"]
    assert original == copy_before
