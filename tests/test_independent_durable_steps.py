"""Real runner I/O and failure semantics, using FAKE solvers only.

No physical shell operator is instantiated, no physical calculations occur.
Test verifies both independent output channels, partial history, PAUSE file
contract and rejection of invalid states even when mocked solver returns.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import independent_mkr.run_models as mkr_runner
import independent_mvi.run_models as mvi_runner

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config"/"verification"/"independent_all_models"


def _case(tmp_path,method,steps=2):
    original=CASES/f"{method}_CUF_TE1_NEWMARK.json"
    data=json.loads(original.read_text(encoding="utf-8"))
    data["time"]["steps"]=steps
    output=tmp_path/method/"RUNS"/"output"
    output.parent.mkdir(parents=True)
    data["output_dir"]=str(output)
    path=tmp_path/method/"CASES"/"case.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data),encoding="utf-8")
    return path,output


@pytest.mark.parametrize("method",("MKR","MVI"))
@pytest.mark.parametrize("nonfinite", (False,True))
def test_runner_persists_accepted_history_and_fails_closed_without_physics(
    tmp_path,monkeypatch,method,nonfinite
):
    module=mkr_runner if method=="MKR" else mvi_runner
    path,output=_case(tmp_path,method)
    active=[]
    class SyntheticSolver:
        def __init__(self,physics,*args):
            self.n=6
            self.physics=physics
        def step_newmark(self,q,v,a,t,dt_s,**kwargs):
            k=len(active)+1
            active.append(k)
            new=np.array(q,copy=True)
            new[4,5,3]=float(k)*1.e-6
            if nonfinite and k==2:
                new[4,5,3]=np.nan
            final=(new,v.copy(),a.copy())
            if method=="MVI":
                rank=1
                nx,ns=self.physics.nx,self.physics.ns
                X=np.zeros((self.n,rank,nx))
                Y=np.zeros((self.n,rank,ns))
                return final,(X,Y),((k,"Y"),(k,"X"))
            return final
    if method=="MKR":
        monkeypatch.setattr(module,"IndependentStrongModel",SyntheticSolver)
    else:
        monkeypatch.setattr(module,"IndependentVariationalModel",SyntheticSolver)
    if nonfinite:
        with pytest.raises(RuntimeError,match="NONFINITE"):
            module.run(path,validate_only=False)
    else:
        report=module.run(path,validate_only=False)
        assert report["status"]=="CONVERGED"
    status=json.loads((output/f"{method}_OWN_RUN_SUMMARY.json").read_text())
    assert status["accepted_steps"]==(1 if nonfinite else 2)
    assert status["status"]==("FAILED" if nonfinite else "CONVERGED")
    if nonfinite:
        assert status["failed_step"]==2
        assert status["error_type"]=="RuntimeError"
    with (output/f"{method}_OWN_PHYSICAL_HISTORY.csv").open(
        newline="",encoding="utf-8"
    ) as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==(1 if nonfinite else 2)
    assert all(row["stage"]=="END" for row in rows)
    actual=list(output.glob("MKR_STEP_*.npz" if method=="MKR" else
                            "MVI_R1_STEP_*.npz"))
    expected=(1 if nonfinite else 2)*(1 if method=="MKR" else 2)
    assert len(actual)==expected
    if nonfinite:
        assert not list(output.glob("MKR_STEP_0002_*.npz"))
        assert not list(output.glob("MVI_R*_STEP_0002_*.npz"))


@pytest.mark.parametrize("method",("MKR","MVI"))
def test_case_validation_never_constructs_solver(tmp_path,monkeypatch,method):
    module=mkr_runner if method=="MKR" else mvi_runner
    path,output=_case(tmp_path,method)
    def forbidden(*args,**kwargs):
        raise AssertionError("VALIDATION_TRIED_TO_COMPUTE")
    if method=="MKR":
        monkeypatch.setattr(module,"IndependentStrongModel",forbidden)
    else:
        monkeypatch.setattr(module,"IndependentVariationalModel",forbidden)
    result=module.run(path,validate_only=True)
    assert result["validated_only"] is True
    assert not output.exists()


@pytest.mark.parametrize("method",("mkr","mvi"))
def test_gui_pause_is_explicit_signal_not_subprocess_os_suspend(method):
    source=(ROOT/f"independent_{method}"/"gui.py").read_text(
        encoding="utf-8")
    runner=(ROOT/f"independent_{method}"/"run_models.py").read_text(
        encoding="utf-8")
    assert "def toggle_pause(self)" in source
    assert 'self.pause_signal=self.saved_path.with_suffix(".pause")' in source
    assert "self.pause_signal.unlink(missing_ok=True)" in source
    assert 'pause_signal=Path(path).with_suffix(".pause")' in runner
    assert 'while pause_signal.is_file():' in runner
    assert 'status="PAUSED"' not in runner
    assert 'summary["status"]="PAUSED"' in runner


@pytest.mark.parametrize("method", ("MKR", "MVI"))
def test_interrupted_npz_write_rolls_back_entire_uncommitted_step(
    tmp_path, monkeypatch, method
):
    """Simulate write failure on step two; no physical solve is constructed."""
    module=mkr_runner if method=="MKR" else mvi_runner
    path,output=_case(tmp_path,method)
    count=[]
    class SyntheticSolver:
        def __init__(self,physics,*args):
            self.n=6
            self.physics=physics
        def step_newmark(self,q,v,a,t,dt_s,**kwargs):
            count.append(1)
            state=(q.copy(),v.copy(),a.copy())
            if method=="MVI":
                X=np.zeros((self.n,1,self.physics.nx))
                Y=np.zeros((self.n,1,self.physics.ns))
                return state,(X,Y),()
            return state
    monkeypatch.setattr(module,
        "IndependentStrongModel" if method=="MKR" else "IndependentVariationalModel",
        SyntheticSolver)
    original=module._save_npz_atomic
    def fail_on_second_step(destination,**arrays):
        if "STEP_0002_" in str(destination):
            raise OSError("SIMULATED_STEP_WRITE_FAILURE")
        return original(destination,**arrays)
    monkeypatch.setattr(module,"_save_npz_atomic",fail_on_second_step)
    with pytest.raises(OSError,match="SIMULATED_STEP_WRITE_FAILURE"):
        module.run(path,validate_only=False)
    report=json.loads((output/f"{method}_OWN_RUN_SUMMARY.json").read_text())
    assert report["status"]=="FAILED"
    assert report["accepted_steps"]==1
    assert report["failed_step"]==2
    with (output/f"{method}_OWN_PHYSICAL_HISTORY.csv").open(
        encoding="utf-8",newline=""
    ) as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==1
    assert rows[0]["step"]=="1"
    assert not list(output.glob("*STEP_0002_*.npz"))
    assert list(output.glob("*STEP_0001_*.npz"))


@pytest.mark.parametrize("method", ("MKR", "MVI"))
@pytest.mark.parametrize("scheme", ("BATHE", "GAUSS_IRK4"))
def test_multistage_write_failure_discards_all_uncommitted_stages(
    tmp_path, monkeypatch, method, scheme
):
    """Bathe and Gauss stages are rolled back as a unit, with fake solver only."""
    module=mkr_runner if method=="MKR" else mvi_runner
    path,output=_case(tmp_path,method)
    document=json.loads(path.read_text(encoding="utf-8"))
    document["time"]["integrator"]=scheme
    path.write_text(json.dumps(document),encoding="utf-8")

    class SyntheticSolver:
        def __init__(self,physics,*args):
            self.n=6
            self.physics=physics
        def _factors(self):
            X=np.zeros((self.n,1,self.physics.nx))
            Y=np.zeros((self.n,1,self.physics.ns))
            return (X,Y)
        def step_bathe(self,q,v,a,t,dt_s,**kwargs):
            state=(q.copy(),v.copy(),a.copy())
            if method=="MVI":
                return state,state,self._factors(),()
            return state,state
        def step_gauss_irk4(self,q,v,a,t,dt_s,**kwargs):
            state=(q.copy(),v.copy(),a.copy())
            if method=="MVI":
                return (state,state),state,self._factors(),()
            return (state,state),state

    monkeypatch.setattr(module,
        "IndependentStrongModel" if method=="MKR" else "IndependentVariationalModel",
        SyntheticSolver)
    original=module._save_npz_atomic
    last_stage="END"
    def fail_last_stage(destination,**arrays):
        if "STEP_0002_END.npz" in str(destination):
            raise OSError("SIMULATED_MULTISTAGE_WRITE_FAILURE")
        return original(destination,**arrays)
    monkeypatch.setattr(module,"_save_npz_atomic",fail_last_stage)
    with pytest.raises(OSError,match="SIMULATED_MULTISTAGE_WRITE_FAILURE"):
        module.run(path,validate_only=False)
    report=json.loads((output/f"{method}_OWN_RUN_SUMMARY.json").read_text())
    assert report["status"]=="FAILED"
    assert report["accepted_steps"]==1
    with (output/f"{method}_OWN_PHYSICAL_HISTORY.csv").open(
        newline="",encoding="utf-8"
    ) as stream:
        rows=list(csv.DictReader(stream))
    assert rows
    assert all(row["step"]=="1" for row in rows)
    assert len(rows)==(2 if scheme=="BATHE" else 3)
    assert not list(output.glob("*STEP_0002_*.npz"))
    assert len(list(output.glob("*STEP_0001_*.npz")))==(
        (2 if scheme=="BATHE" else 3)+(1 if method=="MVI" else 0))
