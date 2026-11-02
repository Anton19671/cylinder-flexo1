"""Modal logging: progress must be readable without modifying eigenvalue work."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"interface"))
from cylinder_modal_journal import format_modal_event, should_display_modal_event


def test_column_events_are_never_in_human_journal():
    for name in ("K_COLUMN_START","K_COLUMN_DONE","M_COLUMN_START",
                 "M_COLUMN_DONE","MVI_INITIAL_X_K_COLUMN_START",
                 "MVI_CANDIDATE_INITIAL_STATE","MVI_HALFSTEP_HARMONIC_AUDIT"):
        assert format_modal_event({"stage":name,"frequency_hz":4465396}) is None
        assert not should_display_modal_event({"stage":name})


def test_attempt_count_and_harmonic_seed_visible():
    s=format_modal_event({"stage":"MVI_CANDIDATE_START","mode":1,
                          "n_modes":12,"rank":1,"attempt":5,"attempts":24,
                          "seed":{"kind":"harmonic_partner","n":7,"phase":"sin"},
                          "elapsed_s":8.29})
    assert "форма 1/12" in s and "попытка 5/24" in s
    assert "n=7" in s and "sin" in s and "8.3 с" in s


def test_converged_candidate_not_presented_as_accepted_mode():
    s=format_modal_event({"stage":"MVI_CANDIDATE_CONVERGED",
                          "mode":1,"n_modes":12,"rank":1,"attempt":1,
                          "attempts":24,"frequency_hz":574.88447})
    assert "кандидат" in s and "не принятая" in s
    assert "574.884470 Гц" in s


def test_cycle_has_frequency_and_shape_change():
    s=format_modal_event({"stage":"MVI_SWEEP_DONE","mode":1,"n_modes":12,
                          "rank":1,"attempt":1,"attempts":24,"sweep":4,
                          "max_sweeps":30,"frequency_hz":574.88447,
                          "shape_change_relative":1.e-7})
    assert "цикл Y→X 4/30" in s
    assert "попытка 1/24" in s
    assert "Δформы=1.00e-07" in s


def test_only_final_mode_is_accepted():
    s=format_modal_event({"stage":"MVI_MODE_DONE","mode":12,
                          "n_modes":12,"rank":1,"frequency_hz":843.00958})
    assert "СОБСТВЕННАЯ ФОРМА ПРИНЯТА" in s and "843.009580" in s


def test_rejected_attempt_contains_short_reason():
    s=format_modal_event({"stage":"MVI_CANDIDATE_REJECTED","mode":1,
                          "n_modes":12,"rank":1,"attempt":3,"attempts":24,
                          "reason":"MODAL_MVI_INVALID_TEST"})
    assert "отклонена" in s and "MODAL_MVI_INVALID_TEST" in s


def test_physical_fdm_modes_have_distinct_method_label():
    s=format_modal_event({"stage":"FDM_MODE_DONE","mode":1,
                          "n_modes":12,"frequency_hz":574.88447})
    assert "МКР/FDM" in s and "ПРИНЯТА" in s
