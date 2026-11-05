"""Headless lifecycle tests of each method's GUI, without physical solves.

Use dummy GUI attributes and a finished child process; no Tk root is created.
"""
from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

import independent_mkr.gui as mkr
import independent_mvi.gui as mvi


class Widget:
    def __init__(self):
        self.state={}
    def configure(self,**kwargs):
        self.state.update(kwargs)


@pytest.mark.parametrize("method,ui,summary_name,solver_name",[
    ("MKR",mkr,"MKR_OWN_RUN_SUMMARY.json","MKR_STRONG_FINITE_DIFFERENCES"),
    ("MVI",mvi,"MVI_OWN_RUN_SUMMARY.json","MVI_XY_VARIATIONAL_ITERATIONS"),
])
def test_stop_status_persisted_in_valid_json_after_child_exit(
    tmp_path,method,ui,summary_name,solver_name
):
    root=tmp_path/method
    root.mkdir()
    path=root/summary_name
    original={
        "method":solver_name,"theory":"KL","integrator":"NEWMARK",
        "status":"IN_PROGRESS","accepted_steps":1,"requested_steps":3,
        "dt_s":.002,"time_s":.002
    }
    path.write_text(json.dumps(original),encoding="utf-8")
    observed=[]
    class DummyWindow:
        proc=SimpleNamespace(returncode=-15,poll=lambda: -15)
        saved_output=root
        pause_signal=None
        _stop_requested=True
        _close_after_stop=False
        run_button=Widget()
        stop_button=Widget()
        pause_button=Widget()
        progress_text=SimpleNamespace(set=lambda _value:None)
        lines=Queue()
        def _note(self,value):
            observed.append(str(value))
        def winfo_exists(self):
            return False
        def after(self,*args):
            raise AssertionError("UNEXPECTED_BACKGROUND_TASK")
        def destroy(self):
            raise AssertionError("UNEXPECTED_GUI_SHUTDOWN")
    target=DummyWindow()
    ui.IndependentMKRWindow._poll(target) if method=="MKR" else (
        ui.IndependentMVIWindow._poll(target))
    result=json.loads(path.read_text(encoding="utf-8"))
    assert result["status"]=="STOPPED_BY_USER"
    assert result["accepted_steps"]==1
    assert target.proc is None
    assert target.stop_button.state["state"]=="disabled"
    assert target.run_button.state["state"]=="normal"
    assert any("STOPPED_BY_USER" in line for line in observed)


@pytest.mark.parametrize("ui", (mkr,mvi))
def test_gui_has_case_browser_numerical_history_and_safe_browse(ui):
    import ast
    source=Path(ui.__file__).read_text(encoding="utf-8")
    tree=ast.parse(source)
    name="IndependentMKRWindow" if ui is mkr else "IndependentMVIWindow"
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)
             and x.name==name)
    functions={x.name for x in cls.body if isinstance(x,ast.FunctionDef)}
    assert {"import_reference_case","load_own_case","view_results",
            "toggle_pause","stop","run","_close","_poll"}<=functions
    assert 'text="Загрузить CASE"' in source
    assert 'text="Просмотр результатов"' in source
    assert "csv.DictReader" in source
    assert "load_case(case_file)" in source
    assert 'report["status"]="STOPPED_BY_USER"' in source
