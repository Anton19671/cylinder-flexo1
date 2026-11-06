"""Output durability regression for separate MKR/MVI runners; no physical calculations."""
from pathlib import Path

import numpy as np
import pytest

from independent_mkr.run_models import _save_npz_atomic as save_mkr
from independent_mvi.run_models import _save_npz_atomic as save_mvi


@pytest.mark.parametrize("save", [save_mkr, save_mvi])
def test_npz_checkpoint_is_readable_and_no_temporary_file_remains(tmp_path, save):
    final = tmp_path / "step_0001.npz"
    sample = np.arange(18, dtype=float).reshape(3, 6)
    save(final, mechanical=sample, time_s=0.002)
    with np.load(final) as saved:
        np.testing.assert_array_equal(saved["mechanical"], sample)
        assert float(saved["time_s"]) == 0.002
    assert not (tmp_path / "step_0001.tmp.npz").exists()


@pytest.mark.parametrize("save", [save_mkr, save_mvi])
def test_npz_failed_compression_never_publishes_incomplete_checkpoint(tmp_path, monkeypatch):
    module = __import__(save.__module__, fromlist=["np"])
    final = tmp_path / "step_0002.npz"

    def simulate_write_interruption(stream, **arrays):
        stream.write(b"INCOMPLETE")
        raise OSError("simulated storage interruption")

    monkeypatch.setattr(module.np, "savez_compressed", simulate_write_interruption)
    with pytest.raises(OSError, match="simulated storage interruption"):
        save(final, mechanical=np.zeros((2, 2)))
    assert not final.exists()
    assert not (tmp_path / "step_0002.tmp.npz").exists()


@pytest.mark.parametrize("save", [save_mkr, save_mvi])
def test_existing_checkpoint_remains_unchanged_after_failed_new_write(tmp_path, monkeypatch):
    module = __import__(save.__module__, fromlist=["np"])
    final = tmp_path / "step_0003.npz"
    save(final, mechanical=np.array([3.0]))

    def fail(stream, **arrays):
        stream.write(b"BROKEN")
        raise OSError("disk full")

    monkeypatch.setattr(module.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        save(final, mechanical=np.array([4.0]))
    with np.load(final) as saved:
        np.testing.assert_array_equal(saved["mechanical"], [3.0])
    assert not (tmp_path / "step_0003.tmp.npz").exists()
