"""The eight-rank CI audit must reject incorrect modal shapes or spectra."""
from copy import deepcopy

from modal_mvi_spectrum_audit import check_spectrum_acceptance


def _valid_report():
    names=["MKR_FDM"] + [f"MVI_{r}" for r in range(1,9)]
    comparison={
        "by_frequency_index": [
            {"mode":i+1,"difference_percent":0.00000001}
            for i in range(8)],
        "max_offdiag_mass_mac":1.e-20,
        "degenerate_pair_subspaces": {
            key:{"principal_mass_mac":[1.,.999999999999],
                 "subspace_verified":True}
            for key in ("1-2","3-4","5-6","7-8")},
    }
    return {
        "runs": {name:{"status":"SOLVED"} for name in names},
        "comparisons": {name:deepcopy(comparison) for name in names[1:]},
        "fdm_max_offdiag_mass_mac":1.e-20,
        "stiffness_operator_probe":{"relative_norm_difference":1.e-10},
    }


def test_all_eight_ranks_and_doublet_subspaces_accept():
    assert check_spectrum_acceptance(_valid_report())["status"]=="PASS"


def test_missing_rank_is_failure_even_if_all_others_solved():
    report=_valid_report()
    report["runs"]["MVI_2"]["status"]="FAILED"
    assert "MVI_2:NOT_SOLVED" in check_spectrum_acceptance(report)["issues"]


def test_missing_comparison_is_failure_even_if_solver_completed():
    report=_valid_report()
    del report["comparisons"]["MVI_4"]
    assert "MVI_4:COMPARISON_MISSING" in check_spectrum_acceptance(report)["issues"]


def test_first_four_modes_match_but_last_doublet_missing_fails():
    report=_valid_report()
    del report["comparisons"]["MVI_1"]["degenerate_pair_subspaces"]["7-8"]
    assert "MVI_1:DOUBLET_SUBSPACE_MISSING" in check_spectrum_acceptance(report)["issues"]


def test_same_eigenfrequencies_do_not_mask_wrong_doublet_shapes():
    report=_valid_report()
    report["comparisons"]["MVI_3"]["degenerate_pair_subspaces"]["5-6"][
        "principal_mass_mac"]=[1.,.7]
    assert ("MVI_3:DOUBLET_SUBSPACE_FAILED_5-6"
            in check_spectrum_acceptance(report)["issues"])


def test_frequency_mismatch_fails_despite_correct_modes():
    report=_valid_report()
    report["comparisons"]["MVI_8"]["by_frequency_index"][7][
        "difference_percent"]=.25
    assert ("MVI_8:FREQUENCY_MISMATCH_MODE_8"
            in check_spectrum_acceptance(report)["issues"])


def test_numerical_operator_mismatch_is_not_ignored():
    report=_valid_report()
    report["stiffness_operator_probe"]["relative_norm_difference"]=.001
    assert "K_OPERATOR_NOT_VERIFIED" in check_spectrum_acceptance(report)["issues"]


def test_mass_orthogonality_must_hold_for_each_rank():
    report=_valid_report()
    report["comparisons"]["MVI_7"]["max_offdiag_mass_mac"]=.002
    assert ("MVI_7:MASS_ORTHOGONALITY_NOT_VERIFIED"
            in check_spectrum_acceptance(report)["issues"])
