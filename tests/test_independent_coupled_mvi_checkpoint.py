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
from independent_mvi.coupled_static import CoupledMVISweepResult
from independent_mvi.coupled_checkpoint import input_manifest
from independent_mvi.coupled_checkpoint import _check_stored_field


@pytest.mark.parametrize("amplitude",(1.e-200,1.e200))
def test_checkpoint_comparison_retains_extreme_finite_field_scales(amplitude):
    expected=np.full((3,4),amplitude)
    _check_stored_field(expected.copy(),expected,"MECHANICAL")
    changed=expected.copy()
    changed[0,0]*=2.
    with pytest.raises(ValueError,match="CHECKPOINT_MECHANICAL_MISMATCH"):
        _check_stored_field(changed,expected,"MECHANICAL")


def test_checkpoint_empty_and_nonfinite_fields():
    empty=np.zeros((0,9,5))
    _check_stored_field(empty,empty,"ELECTRIC")
    with pytest.raises(ValueError,match="SHAPE_OR_FINITE"):
        _check_stored_field(np.zeros(1),np.array([np.nan]),"MECHANICAL")


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


def test_stop_is_polled_inside_dynamic_step_and_never_commits_partial_state(
        tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch)
    polls=[]
    def control(output):
        polls.append(output)
        if len(polls)==2:
            raise RuntimeError("SYNTHETIC_USER_STOP")
    def advance(self,state,dt_s,**kwargs):
        assert callable(kwargs["cancelled"])
        kwargs["cancelled"]()
        raise AssertionError("Stop must interrupt the active timestep")
    monkeypatch.setattr(native,"cooperate",control)
    monkeypatch.setattr(native.NativeCoupledMVINewmark,"advance",advance)
    with pytest.raises(RuntimeError,match="SYNTHETIC_USER_STOP"):
        native.run(path,validate_only=False)
    assert len(polls)==2
    assert summary(tmp_path)["accepted_steps"]==0
    assert (tmp_path/"output"/"STEP_0000.npz").is_file()
    assert not (tmp_path/"output"/"STEP_0001.npz").exists()


def test_changed_implementation_never_resumes_or_overwrites_checkpoint(tmp_path,monkeypatch):
    path=build(tmp_path)
    mock_drivers(monkeypatch,fail_at=2)
    with pytest.raises(RuntimeError,match="SYNTHETIC_INTERRUPTION"):
        native.run(path,validate_only=False)
    file=tmp_path/"output"/"RUN_SUMMARY.json"
    before=file.read_bytes()
    assert len(summary(tmp_path)["solver_implementation_sha256"])==64
    monkeypatch.setattr(native,"solver_implementation_fingerprint",lambda:"0"*64)
    mock_drivers(monkeypatch)
    with pytest.raises(ValueError,match="IMPLEMENTATION_CHANGED_OR_UNRECORDED"):
        native.run(path,validate_only=False,resume=True)
    assert file.read_bytes()==before
    assert (tmp_path/"output"/"STEP_0001.npz").exists()
    assert not (tmp_path/"output"/"STEP_0002.npz").exists()


def test_source_fingerprint_tracks_bytes_and_relative_names(tmp_path):
    from independent_mvi.coupled_checkpoint import solver_implementation_fingerprint
    folder=tmp_path/"mechanics"
    folder.mkdir()
    source=folder/"local.py"
    source.write_text("VALUE=1\n")
    one=solver_implementation_fingerprint(tmp_path)
    assert solver_implementation_fingerprint(tmp_path)==one
    source.write_text("VALUE=2\n")
    two=solver_implementation_fingerprint(tmp_path)
    assert two!=one
    source.rename(folder/"other.py")
    assert solver_implementation_fingerprint(tmp_path)!=two


@pytest.mark.parametrize("accepted",[True,False])
def test_dynamic_nodal_norms_and_failure_history_survive_runner(
        tmp_path,monkeypatch,accepted):
    path=build(tmp_path,steps=1)
    mock_drivers(monkeypatch)
    def advance(self,state,dt_s,**kw):
        future=CoupledMVIDynamicState(
            state.time_s+dt_s,state.factors,state.velocity,state.acceleration)
        solve=CoupledMVISweepResult(
            state.factors,accepted,
            "MVI_FIXED_RANK_FULL_NODAL_STATIONARY" if accepted else
            "MVI_FIXED_RANK_INSUFFICIENT_FULL_NODAL_WORK",
            3,1.e-10,2.e-10,3.e-10,
            (dict(sweep=3,full_nodal=4.e-10 if accepted else .003,
                  condition=float("inf")),),
            4.e-10 if accepted else .003)
        return SimpleNamespace(
            state=future if accepted else None,solver=solve,
            attempted_time_s=future.time_s,status="MVI_NEWMARK_"+solve.status)
    monkeypatch.setattr(native.NativeCoupledMVINewmark,"advance",advance)
    if accepted:
        report=native.run(path,validate_only=False)
        assert report["full_nodal_equilibrium_satisfied"] is True
        assert report["full_nodal_norm"]==pytest.approx(4.e-10)
        assert report["solution_quality"]=="FULL_NODAL_EQUILIBRIUM"
        assert report["projected_norm_x"]==pytest.approx(2.e-10)
    else:
        with pytest.raises(RuntimeError,match="INSUFFICIENT_FULL_NODAL_WORK"):
            native.run(path,validate_only=False)
        report=summary(tmp_path)
        assert report["accepted_steps"]==0
        assert report["failed_time_step"]["solves"][0]["full_nodal_norm"]==.003
        assert not (tmp_path/"output"/"STEP_0001.npz").exists()
    file=tmp_path/"output"/"MVI_STEP_0001_DIAGNOSTICS.json"
    record=json.loads(file.read_text())
    assert record["state_accepted"] is accepted
    assert record["requested_rank"]==1
    assert record["solves"][0]["history"][0]["condition"] is None
    if not accepted:
        # A successful resume must not leave the rejected attempt in the
        # current summary when the retried step succeeds.
        mock_drivers(monkeypatch)
        resumed=native.run(path,validate_only=False,resume=True)
        assert resumed["accepted_steps"]==1
        assert "failed_time_step" not in resumed


def test_unmeasured_or_projected_dynamic_work_is_never_marked_full():
    solve=dict(status="CONVERGED_PROJECTED",converged=True,
               full_nodal_norm=.003,projected_norm_x=1.e-10,
               projected_norm_y=1.e-10)
    diagnostic=dict(status="ACCEPTED",solves=[solve],joint_nodal_norm=None)
    report=native._dynamic_solution_quality(diagnostic,1.e-8)
    assert report["solution_status"]=="CONVERGED_PROJECTED"
    assert report["full_nodal_equilibrium_satisfied"] is False
    assert report["solution_quality"]=="PROJECTED_EQUILIBRIUM"
    solve.update(status="FULL",full_nodal_norm=1.e-10)
    diagnostic["status"]="MVI_GAUSS_CONVERGED_PROJECTED"
    report=native._dynamic_solution_quality(diagnostic,1.e-8)
    assert report["full_nodal_equilibrium_satisfied"] is False
    assert report["solution_status"]=="CONVERGED_PROJECTED"
    diagnostic["status"]="ACCEPTED"
    solve.update(status="FULL",full_nodal_norm=None)
    report=native._dynamic_solution_quality(diagnostic,1.e-8)
    assert report["full_nodal_equilibrium_satisfied"] is None
    assert report["solution_quality"]=="UNREPORTED_EQUILIBRIUM"


def test_nonfinite_failed_sweep_evidence_is_json_safe():
    result=SimpleNamespace(
        status="MVI_HALFSTEP_TANGENT_NOT_SOLVED",converged=False,
        sweeps=0,field_change=float("inf"),projected_norm_x=float("inf"),
        projected_norm_y=float("inf"),full_nodal_norm=float("inf"),
        history=({"reduced_halfstep_history":[{"condition":float("inf")} ]},))
    evidence=native._sweep_diagnostics(result)
    json.dumps(evidence,allow_nan=False)
    assert evidence["full_nodal_norm"] is None
    assert evidence["status"]==result.status
    assert evidence["history"][0]["reduced_halfstep_history"][0]["condition"] is None


def test_dynamic_rank_quality_is_monotone_across_accepted_steps():
    """One projected step cannot be erased by later full-rank stationarity."""
    summary=dict(accepted_steps=0)
    projected=dict(solution_status="CONVERGED_PROJECTED",
                   solution_quality="PROJECTED_EQUILIBRIUM",
                   full_nodal_norm=.003,projected_norm_x=1.e-12,
                   projected_norm_y=2.e-12)
    full=dict(solution_status="MVI_NEWMARK_CONVERGED",
              solution_quality="FULL_NODAL_EQUILIBRIUM",
              full_nodal_norm=1.e-13,projected_norm_x=1.e-14,
              projected_norm_y=1.e-14)
    native._accumulate_dynamic_quality(summary,projected)
    summary["accepted_steps"]=1
    native._accumulate_dynamic_quality(summary,full)
    assert summary["dynamic_quality_step_counts"]==dict(
        FULL_NODAL_EQUILIBRIUM=1,PROJECTED_EQUILIBRIUM=1,
        UNREPORTED_EQUILIBRIUM=0)
    assert summary["solution_status"]=="CONVERGED_PROJECTED"
    assert summary["solution_quality"]=="PROJECTED_EQUILIBRIUM"
    assert summary["last_step_solution_quality"]=="FULL_NODAL_EQUILIBRIUM"
    assert summary["full_nodal_equilibrium_satisfied"] is False
    assert summary["full_nodal_norm"]==pytest.approx(.003)


def test_dynamic_quality_resume_requires_history_evidence():
    # An older checkpoint may have accepted steps but no quality ledger.
    summary=dict(accepted_steps=2)
    native._accumulate_dynamic_quality(
        summary,dict(solution_status="MVI_NEWMARK_CONVERGED",
                     solution_quality="FULL_NODAL_EQUILIBRIUM",
                     full_nodal_norm=1.e-13))
    assert summary["dynamic_quality_step_counts"]==dict(
        FULL_NODAL_EQUILIBRIUM=1,PROJECTED_EQUILIBRIUM=0,
        UNREPORTED_EQUILIBRIUM=2)
    assert summary["full_nodal_equilibrium_satisfied"] is None
    summary["accepted_steps"]=3
    native._accumulate_dynamic_quality(summary,{})
    assert summary["solution_quality"]=="UNREPORTED_EQUILIBRIUM"
    assert summary["full_nodal_equilibrium_satisfied"] is None
    assert summary["dynamic_quality_step_counts"]["UNREPORTED_EQUILIBRIUM"]==3


def test_dynamic_projected_and_unknown_states_never_claim_full_balance():
    summary=dict(accepted_steps=0)
    native._accumulate_dynamic_quality(summary,dict(
        solution_status="CONVERGED_PROJECTED",
        solution_quality="PROJECTED_EQUILIBRIUM",
        full_nodal_norm=.1))
    summary["accepted_steps"]=1
    native._accumulate_dynamic_quality(summary,{})
    assert summary["solution_status"]=="CONVERGED_PROJECTED"
    assert summary["solution_quality"]=="PROJECTED_WITH_UNREPORTED_STEPS"
    assert summary["full_nodal_equilibrium_satisfied"] is False
    with pytest.raises(ValueError,match="CHECKPOINT_INCONSISTENT"):
        native._accumulate_dynamic_quality(summary,{})
