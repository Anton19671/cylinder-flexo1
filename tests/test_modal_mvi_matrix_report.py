"""Unit checks of the 56-case CI artifact gate; no shell calculation."""
import copy
import json

import pytest
from ci.modal_mvi_matrix_report import (
    THEORIES, RANKS, summarize, verify_case, write_report,
)


def case(model="KL", rank=1):
    freqs = [100., 100., 200., 200., 300., 300., 400., 400.]
    return {
        "model": model, "rank": rank, "grid": [13, 13],
        "boundary": "C-C", "completed": True,
        "runs": {
            "FDM": {"frequencies_hz": freqs},
            "MVI_" + str(rank): {"frequencies_hz": list(freqs)},
        },
        "comparison": {
            "status": "PASS", "issues": [],
            "frequency_verified": True, "all_mode_subspaces_verified": True,
            "frequency_rtol": 2.e-5, "frequency_atol_hz": 1.e-4,
            "relative_frequency_errors": [0.] * 8,
            "mode_subspaces": [
                {"modes": [j, j+1], "shape_verified": True,
                 "principal_mass_mac": [1., 1.],
                 "minimum_mass_mac": 1.}
                for j in (1, 3, 5, 7)
            ],
        },
    }


def test_all_56_cases_are_enumerated():
    assert len(THEORIES) == 7 and len(RANKS) == 8
    assert len({(t, r) for t in THEORIES for r in RANKS}) == 56


def test_pass_real_frequency_and_shape_compares():
    row = verify_case("FSDT", 3, case("FSDT", 3))
    assert row["status"] == "PASS"
    assert row["frequency_modes_checked"] == 8
    assert row["minimum_principal_mass_mac"] == 1.


def test_corrupted_spectrum_is_not_accepted():
    report = case()
    report["runs"]["MVI_1"]["frequencies_hz"][6] = 410.
    with pytest.raises(ValueError, match="frequency mismatch|unsorted"):
        verify_case("KL", 1, report)


def test_corrupted_or_duplicate_modes_are_not_accepted():
    report = case()
    report["comparison"]["mode_subspaces"][-1]["modes"] = [7, 7]
    with pytest.raises(ValueError, match="missing or repeated"):
        verify_case("KL", 1, report)
    report = case()
    report["comparison"]["mode_subspaces"][-1]["minimum_mass_mac"] = .90
    with pytest.raises(ValueError, match="mass-weighted"):
        verify_case("KL", 1, report)


def test_stale_case_name_and_incomplete_flag_are_not_pass():
    for field, val in (("model", "FSDT"), ("completed", False),
                       ("rank", 2)):
        report = case()
        report[field] = val
        with pytest.raises(ValueError, match="identity or completed"):
            verify_case("KL", 1, report)


def test_missing_and_corrupt_artifacts_appear_in_report(tmp_path):
    p = tmp_path / "modal_KL_rank1_audit.json"
    p.write_text("{broken", encoding="utf-8")
    rows = summarize(tmp_path)
    assert len(rows) == 56
    assert rows[0]["status"] == "FAIL"
    assert rows[1]["status"] == "MISSING"
    verdict = write_report(rows, tmp_path / "output")
    assert verdict == "INCOMPLETE_OR_FAILED"
    assert (tmp_path / "output" / "matrix-56.csv").is_file()


def test_complete_matrix_uses_independent_per_case_files(tmp_path):
    for theory in THEORIES:
        for rank in RANKS:
            p = tmp_path / (
                "modal_" + theory + "_rank" + str(rank) + "_audit.json")
            p.write_text(json.dumps(case(theory, rank)), encoding="utf-8")
    rows = summarize(tmp_path)
    assert all(row["status"] == "PASS" for row in rows)
    assert write_report(rows, tmp_path / "report") == "PASS"
