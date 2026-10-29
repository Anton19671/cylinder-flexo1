"""V-8/V-9: reference runs never count rejected iterates as accepted."""
import json
from pathlib import Path
from cylinder_reference_progress_adapter import (
    normalize_reference_event, accepted_reference_steps)


def test_reference_events_are_readable():
    v=normalize_reference_event({"event":"reference_dynamic_sweep","step":2,
                                 "iteration":4,"residual_scaled":.001,
                                 "load_factor":.5,"seq":12})
    assert v["step_index"]==2 and v["iteration"]==4
    assert v["residual_value"]==.001
    assert normalize_reference_event({"event":"newton_iteration","step":2}) is None


def test_only_durable_accepted_reference_steps_count(tmp_path):
    rows=[{"index":1,"converged":True,"state_file":"REFERENCE_STEP_0001.npz"},
          {"index":2,"converged":False,"state_file":"REFERENCE_FAILED_ITERATE.npz"}]
    (tmp_path/"REFERENCE_STEP_0001.npz").write_bytes(b"test-state")
    (tmp_path/"REFERENCE_FAILED_ITERATE.npz").write_bytes(b"test-rejected")
    (tmp_path/"FINAL_RUN_SUMMARY.json").write_text(
        json.dumps({"calculation_kind":"STATIC","steps":rows}),encoding="utf-8")
    found=accepted_reference_steps(tmp_path)
    assert len(found)==1 and found[0]["index"]==1


def test_missing_state_never_counts_as_accepted(tmp_path):
    (tmp_path/"FINAL_RUN_SUMMARY.json").write_text(json.dumps({
        "calculation_kind":"DYNAMIC",
        "steps":[{"index":1,"converged":True,
                  "state_file":"REFERENCE_DYNAMIC_STEP_0001.npz"}]}),encoding="utf-8")
    assert accepted_reference_steps(tmp_path)==[]
