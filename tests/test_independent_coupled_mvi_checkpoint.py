"""Crash/restart integration of independent coupled MVI, without shell solves.

Mocked time drivers produce valid fixed-rank states. These tests exercise
persistent inputs, summaries, NPZ/CSV checkpoints and reentry exclusively.
"""
from __future__ import annotations

import csv
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_paths import activate
activate()
from independent_mvi import run_coupled_research as native
from independent_mvi.coupled_initial import MVIAccelerationResult
from independent_mvi.coupled_newmark import CoupledMVIDynamicState
from independent_mvi.coupled_checkpoint import input_manifest


def build(tmp_path,*,steps=3):
    nx,ns=9,5
    x=np.zeros((3,1,nx))
    x[:,:,1:-1]=1.e-6
    y=np.ones((3,1,ns))
    np.savez_compressed(
        tmp_path/"initial.npz",x_profiles=x,y_profiles=y,
        velocity=np.zeros((3,nx,ns)))
    np.savez_compressed(
        tmp_path/"loads.npz",mechanical_load=np.zeros((3,nx,ns)))
    document=dict(
        schema=native.SCHEMA,research_opt_in=True,
        physics=dict(theory="KL",geometry_model="LINEAR",
                     material=dict(young_pa=1.e9,poisson=.3,ell_m=1.e-6,
                                   mu1_c_per_m=0.,mu2_c_per_m=0.,
                                   kappa_f_per_m=0.,gradient_model="G3D"),
                     thickness_m=.1,density_kg_m3=2500.,eta_s=0.,
                     mass_damping_rate_per_s=0.,n_gauss=2,
                     potential_order=None),
        grid=dict(nx=nx,ns=ns,length_m=1.,radius_m=2.),
        boundary=dict(mechanical="IMMOBILE_PARALLEL_ENDS",electric="OFF"),
        solver=dict(kind="DYNAMIC",integrator="NEWMARK",dt_s=.001,
                    steps=steps,max_sweeps=4,
                    max_acceleration_coordinates=1024,
                    max_halfstep_coordinates=512,
                    field_scales=[1.]*3,energy_scale_j=1.),
        rank=1,initial_state_npz="initial.npz",loads_npz="loads.npz",
        output_dir="output")
    path=tmp_path/"case.json"
    path.write_text(json.dumps(document),encoding="utf-8")
    return path


def mock_drivers(monkeypatch,*,fail_at=None):
    events=[]
    def initial(config,factors,velocity,**kw):
        events.append(("initial",0))
        state=CoupledMVIDynamicState(
            0.,factors,velocity,np.zeros_like(velocity))
        return MVIAccelerationResult(
            state,"MVI_INITIAL_EQUILIBRIUM_CLOSED",None,None)
    def advance(self,state,dt_s,**kwargs):
        step=round(state.time_s/dt_s)+1
        events.append(("advance",step))
        if step==fail_at:
            raise RuntimeError("SYNTHETIC_INTERRUPTION")
        future=CoupledMVIDynamicState(
            state.time_s+dt_s,state.factors,state.velocity,
            state.acceleration)
        return SimpleNamespace(state=future,status="MOCK_ACCEPTED")
    monkeypatch.setattr(native,"close_mvi_instantaneous_state",initial)
    monkeypatch.setattr(native.NativeCoupledMVINewmark,"advance",advance)
    monkeypatch.setattr(native,"_dynamic_green_audit",lambda *a,**k:{})
    return events


def summary(tmp_path):
    return json.loads((tmp_path/"output"/"RUN_SUMMARY.json").read_text())


def history(tmp_path):
    with (tmp_path/"output"/"PHYSICAL_RADIAL_HISTORY.csv").open(
            newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_resume_continues_from_last_accepted_step_without_reinitialization(
        tmp_path,monkeypatch):
    path=build(tmp_path)
    events=mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError,match="SYNTHETIC_INTERRUPTION"):
        native.run(path,validate_only=False)
    assert events==[("initial",0),("advance",1),("advance",2)]
    assert summary(tmp_path)["accepted_steps"]==1
    assert len(history(tmp_path))==2
    assert (tmp_path/"output"/"STEP_0001.npz").exists()
    resumed_events=mock_drivers(monkeypatch)
    done=native.run(path,validate_only=False,resume=True)
    assert resumed_events==[("advance",2),("advance",3)]
    assert done["accepted_steps"]==3
    assert done["status"]=="EXPERIMENTAL_COMPLETED_UNCERTIFIED"
    assert len(history(tmp_path))==4
    assert [int(row["step"]) for row in history(tmp_path)]==[0,1,2,3]
    assert (tmp_path/"output"/"STEP_0003.npz").exists()


def test_changed_input_never_corrupts_prior_summary(tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    before=(tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()
    with np.load(tmp_path/"loads.npz") as source:
        changed=np.array(source["mechanical_load"])
    changed[0,4,0]=1.
    np.savez_compressed(tmp_path/"loads.npz",mechanical_load=changed)
    with pytest.raises(ValueError,match="CHECKPOINT_INPUT_CHANGED"):
        native.run(path,validate_only=False,resume=True)
    assert (tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()==before
    assert (tmp_path/"output"/"STEP_0001.npz").exists()


def test_corrupt_committed_state_rejected_without_summary_change(
        tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    original=(tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()
    file=tmp_path/"output"/"STEP_0001.npz"
    with np.load(file,allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in archive.files}
    data["mechanical"][2,4,0]+=1.
    np.savez_compressed(file,**data)
    with pytest.raises(ValueError,match="CHECKPOINT_STATE_FINGERPRINT_MISMATCH"):
        native.run(path,validate_only=False,resume=True)
    assert (tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()==original


@pytest.mark.parametrize("part",("velocity","acceleration"))
def test_tampered_dynamics_cannot_be_resumed(tmp_path,monkeypatch,part):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    original=(tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()
    file=tmp_path/"output"/"STEP_0001.npz"
    with np.load(file,allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in archive.files}
    data[part][0,4,0]+=1.e-7
    np.savez_compressed(file,**data)
    with pytest.raises(ValueError,match="CHECKPOINT_STATE_FINGERPRINT_MISMATCH"):
        native.run(path,validate_only=False,resume=True)
    assert (tmp_path/"output"/"RUN_SUMMARY.json").read_bytes()==original


def test_orphan_future_npz_and_csv_are_trimmed_at_restart(
        tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    folder=tmp_path/"output"
    # The orphan is beyond the committed marker and must not count.
    shutil.copyfile(folder/"STEP_0001.npz",folder/"STEP_0002.npz")
    with (folder/"PHYSICAL_RADIAL_HISTORY.csv").open(
            "a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(history(tmp_path)[0]))
        writer.writerow(history(tmp_path)[-1])
    events=mock_drivers(monkeypatch)
    result=native.run(path,validate_only=False,resume=True)
    assert result["accepted_steps"]==3
    assert events==[("advance",2),("advance",3)]
    assert len(history(tmp_path))==4
    with np.load(folder/"STEP_0002.npz") as archive:
        assert float(archive["time_s"])==pytest.approx(.002)


def test_changed_committed_history_rejected_without_overwriting(
        tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    out=tmp_path/"output"
    old=(out/"RUN_SUMMARY.json").read_bytes()
    history_path=out/"PHYSICAL_RADIAL_HISTORY.csv"
    with history_path.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        cols=tuple(reader.fieldnames)
        rows=list(reader)
    rows[1]["middle_w_m"]="999."
    with history_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError,match="CHECKPOINT_HISTORY_FIELD_MISMATCH"):
        native.run(path,validate_only=False,resume=True)
    assert (out/"RUN_SUMMARY.json").read_bytes()==old
    assert (out/"STEP_0001.npz").exists()


def test_existing_output_without_resume_is_rejected(tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError):
        native.run(path,validate_only=False)
    with pytest.raises(ValueError,match="OUTPUT_ALREADY_EXISTS"):
        native.run(path,validate_only=False)


def test_long_transient_case_validates_without_solving(tmp_path):
    path=build(tmp_path,steps=3000)
    report=native.run(path,validate_only=True)
    assert report["steps"]==3000
    assert report["physical_steps_computed"]==0
    assert not (tmp_path/"output").exists()


def test_manifest_records_exact_original_bytes(tmp_path):
    path=build(tmp_path)
    stamps=input_manifest(path,tmp_path/"initial.npz",tmp_path/"loads.npz")
    assert set(stamps)=={
        "case_sha256","initial_npz_sha256","loads_npz_sha256"}
    assert all(len(x)==64 for x in stamps.values())


def test_long_time_grid_accepts_only_roundoff_not_missing_step():
    from independent_mvi.coupled_checkpoint import _time_close
    dt=.002
    time=0.
    for k in range(1,100001):
        time+=dt
    reference=100000*dt
    assert _time_close(time,reference,dt)
    assert not _time_close(reference+dt,reference,dt)
    assert not _time_close(reference+.02*dt,reference,dt)
    assert not _time_close(float("nan"),reference,dt)
