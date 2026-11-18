"""Native coupled GUI regressions with dummy widgets, no Tk or shell solves."""
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

import cylinder_native_research_gui as gui


def window(tmp_path):
    win=SimpleNamespace(process=None,case_path=tmp_path/"case.json",
                        results_dir=tmp_path/"result",seen_steps=set(),history=[],
                        expected_steps=1,progress={},read_queue=Queue())
    win._log=lambda *args:None
    win._require_idle=lambda:gui.NativeResearchWindow._require_idle(win)
    win._draw_graph=lambda:None
    win._radial_index=lambda theory:2
    win.editor=SimpleNamespace(delete=lambda *args:None,insert=lambda *args:None)
    win._show_output=lambda case,base:setattr(win,"results_dir",Path(case["output_dir"]))
    return win


def test_folder_picker_selects_new_child_and_preserves_existing_results(tmp_path,monkeypatch):
    win=window(tmp_path)
    existing=tmp_path/"case_results"
    existing.mkdir()
    marker=existing/"keep.txt"
    marker.write_text("original")
    case={"output_dir":"old"}
    win._edited=lambda:dict(case)
    monkeypatch.setattr(gui.filedialog,"askdirectory",lambda **kw:str(tmp_path))
    gui.NativeResearchWindow.choose_results(win)
    assert win.results_dir==tmp_path/"case_results_001"
    assert not win.results_dir.exists()
    assert marker.read_text()=="original"


@pytest.mark.parametrize("method",("open_case","choose_results"))
def test_case_and_output_are_locked_while_process_runs(tmp_path,monkeypatch,method):
    win=window(tmp_path)
    win.process=SimpleNamespace(poll=lambda:None)
    errors=[]
    monkeypatch.setattr(gui.messagebox,"showerror",lambda *a,**k:errors.append(a))
    def forbidden(**kwargs):
        raise AssertionError("file picker must not open during a run")
    monkeypatch.setattr(gui.filedialog,"askdirectory",forbidden)
    monkeypatch.setattr(gui.filedialog,"askopenfilename",forbidden)
    getattr(gui.NativeResearchWindow,method)(win)
    assert len(errors)==1
    assert win.results_dir==tmp_path/"result"


def test_single_step_progress_excludes_initial_state(tmp_path):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    q=np.zeros((3,9,5))
    summary=win.results_dir/"RUN_SUMMARY.json"
    summary.write_text(json.dumps({"accepted_steps":0}))
    np.savez(win.results_dir/"STEP_0000.npz",mechanical=q,time_s=0.)
    gui.NativeResearchWindow._scan_history(win)
    assert win.progress["value"]==0.
    np.savez(win.results_dir/"STEP_0001.npz",mechanical=q,time_s=.1)
    summary.write_text(json.dumps({"accepted_steps":1}))
    gui.NativeResearchWindow._scan_history(win)
    assert win.progress["value"]==100.


def test_native_online_history_waits_for_summary_commit_after_step_file(tmp_path):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    summary=win.results_dir/"RUN_SUMMARY.json"
    summary.write_text(json.dumps({"accepted_steps":0}))
    q=np.zeros((3,9,5))
    np.savez(win.results_dir/"STEP_0001.npz",mechanical=q,time_s=.1)
    gui.NativeResearchWindow._scan_history(win)
    assert win.history==[]
    assert win.progress.get("value",0.)==0.
    summary.write_text(json.dumps({"accepted_steps":1}))
    gui.NativeResearchWindow._scan_history(win)
    assert win.history==[(.1,0.)]
    assert win.progress["value"]==100.


@pytest.mark.parametrize("summary_text",(None,"{",'{"accepted_steps": true}',
    '{"accepted_steps": -1}','{"accepted_steps": 1.0}'))
def test_native_history_does_not_accept_steps_without_valid_commit_marker(
        tmp_path,summary_text):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    if summary_text is not None:
        (win.results_dir/"RUN_SUMMARY.json").write_text(summary_text)
    np.savez(win.results_dir/"STEP_0001.npz",
             mechanical=np.zeros((3,9,5)),time_s=.1)
    gui.NativeResearchWindow._scan_history(win)
    assert win.history==[]


def test_native_history_reads_large_step_numbers_in_numeric_order(tmp_path):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.expected_steps=10000
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    (win.results_dir/"RUN_SUMMARY.json").write_text(
        json.dumps({"accepted_steps":10000}))
    for k in (10000,9999):
        np.savez(win.results_dir/f"STEP_{k:04d}.npz",
                 mechanical=np.zeros((3,9,5)),time_s=float(k))
    gui.NativeResearchWindow._scan_history(win)
    assert [t for t,w in win.history]==[9999.,10000.]
    assert win.progress["value"]==100.


@pytest.mark.parametrize("moment",(float("nan"),float("inf"),-1.))
def test_native_history_skips_invalid_physical_time(tmp_path,moment):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    (win.results_dir/"RUN_SUMMARY.json").write_text(
        json.dumps({"accepted_steps":1}))
    np.savez(win.results_dir/"STEP_0001.npz",
             mechanical=np.zeros((3,9,5)),time_s=moment)
    gui.NativeResearchWindow._scan_history(win)
    assert win.history==[]


def test_native_static_plot_waits_until_accepted_counter_is_committed(tmp_path):
    win=window(tmp_path)
    win.results_dir.mkdir()
    win.case_path.write_text(json.dumps({"physics":{"theory":"KL"}}))
    summary=win.results_dir/"RUN_SUMMARY.json"
    summary.write_text(json.dumps({"accepted_steps":0}))
    np.savez(win.results_dir/"STATIC_ACCEPTED.npz",mechanical=np.zeros((3,9,5)))
    gui.NativeResearchWindow._scan_history(win)
    assert win.history==[]
    summary.write_text(json.dumps({"accepted_steps":1}))
    gui.NativeResearchWindow._scan_history(win)
    assert len(win.history)==1


def test_reader_waits_for_exit_before_publishing_completion(tmp_path):
    win=window(tmp_path)
    calls=[]
    process=SimpleNamespace(stdout=iter(["hello\n"]),wait=lambda:calls.append("wait"))
    gui.NativeResearchWindow._reader(win,process)
    assert calls==["wait"]
    assert win.read_queue.get()=="hello\n"
    assert win.read_queue.get()==("finished",process)


def test_stop_is_persisted_without_discarding_committed_step(tmp_path):
    win=window(tmp_path)
    win.results_dir.mkdir()
    path=win.results_dir/"RUN_SUMMARY.json"
    path.write_text(json.dumps({"status":"IN_PROGRESS","accepted_steps":2,
                                "requested_steps":5,"time_s":.2}))
    step=win.results_dir/"STEP_0002.npz"
    step.write_bytes(b"preserved")
    gui.NativeResearchWindow._record_stopped(win)
    saved=json.loads(path.read_text())
    assert saved["status"]=="STOPPED_BY_USER"
    assert saved["accepted_steps"]==2 and saved["time_s"]==.2
    assert step.read_bytes()==b"preserved"
    assert not list(win.results_dir.glob("*.pending"))


def test_kill_only_targets_still_running_child(tmp_path):
    calls=[]
    win=window(tmp_path)
    live=SimpleNamespace(poll=lambda:None,kill=lambda:calls.append("kill"))
    done=SimpleNamespace(poll=lambda:0,kill=lambda:calls.append("wrong"))
    gui.NativeResearchWindow._kill_if_running(win,live)
    gui.NativeResearchWindow._kill_if_running(win,done)
    assert calls==["kill"]
