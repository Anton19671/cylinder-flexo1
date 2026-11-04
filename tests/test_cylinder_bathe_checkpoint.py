"""Filesystem-only restart contract tests: no actual shell solve is launched."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_bathe_checkpoint import load_checked_bathe_checkpoint


def _prepare(tmp_path, method="FDM", rank=None):
    out=tmp_path/"output"
    out.mkdir()
    q=np.zeros((3,5,5))
    v=np.zeros_like(q)
    a=np.zeros_like(q)
    case={"dynamics":{"bathe_resume_accepted":True},
          "theory":"TEST","method":method,"load_definition":{"q":100000}}
    (out/"REFERENCE_CASE.json").write_text(json.dumps(case),encoding="utf-8")
    cfg=SimpleNamespace(n_steps=4,dt_s=.002,t_start_s=0.)
    dyn=SimpleNamespace(settings=cfg,density_kg_m3=7800.,
                        time_law=SimpleNamespace(value=lambda t:float(np.sin(2*np.pi*25*t))))
    prepared=SimpleNamespace(
        method=method,fdm_backend="MATRIX_FREE_GMRES" if method=="FDM" else "DENSE_REFERENCE",
        problem=SimpleNamespace(functional=SimpleNamespace(theory="CUF_TE4")),
        mvi_settings=SimpleNamespace(rank=rank))
    trace=(np.zeros((1,q.size)),np.zeros(1))
    summary=dict(solver_backend="VARIATIONAL_REFERENCE",
                 calculation_kind="DYNAMIC",integrator="BATHE",method=method,
                 theory="CUF_TE4",geometry_model="LINEAR",fdm_backend=prepared.fdm_backend,
                 rank=rank,requested_steps=4,debug_cold_start_each_full_step=False,
                 time_step_s=.002,start_time_s=0.,density_kg_m3=7800.,
                 accepted_steps=0,steps=[],converged=False,status="NOT_STARTED")
    case["geometry_model"]="LINEAR"
    (out/"REFERENCE_CASE.json").write_text(json.dumps(case),encoding="utf-8")
    np.savez_compressed(out/"REFERENCE_DYNAMIC_INITIAL.npz",
                        time_s=np.array(0.),mechanical=q,velocity=v,acceleration=a)
    (out/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary),encoding="utf-8")
    return out,case,prepared,dyn,q,v,a,trace,summary


def _save_accepted(out, dyn, summary, q, rank=None):
    tgamma=(2-np.sqrt(2))*.002
    row=dict(index=1,time_s=.002,time_gamma_s=tgamma,
             load_factor=dyn.time_law.value(.002),
             status="CONVERGED_FULL" if rank else "CONVERGED",
             converged=True,first_substep_file="BATHE_SUBSTEP_0001_GAMMA.npz",
             state_file="REFERENCE_DYNAMIC_STEP_0001.npz")
    summary.update(accepted_steps=1,steps=[row],status="IN_PROGRESS",converged=False)
    (out/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary),encoding="utf-8")
    extra={}
    if rank is not None:
        x=np.zeros((3,rank,5));y=np.zeros_like(x)
        # Orthogonal, normalized inactive coordinates; zero represented field.
        for c in range(3):
            for r in range(rank):
                x[c,r,r]=1.
                y[c,r,r]=1.
        # A nonzero state with all active factors, no invented full-nodal lift.
        q=np.einsum("crx,crs->cxs",x,y)
        extra={"x_profiles":x,"y_profiles":y,"state_scales":np.ones_like(q)}
    for file,t in (("BATHE_SUBSTEP_0001_GAMMA.npz",tgamma),
                   ("REFERENCE_DYNAMIC_STEP_0001.npz",.002)):
        np.savez_compressed(out/file,time_s=np.array(t),mechanical=q,
                            velocity=np.zeros_like(q),acceleration=np.zeros_like(q),
                            load_factor=np.array(dyn.time_law.value(t)),**extra)
    return row


def _load(parts):
    out,case,prepared,dyn,q,v,a,trace,_=parts
    return load_checked_bathe_checkpoint(
        out,case,prepared,dyn,q0=q,p0=None,v0=v,a0=a,
        initial_trace=trace,debug_restart=False)


def test_zero_steps_can_restart_without_invented_states(tmp_path):
    parts=_prepare(tmp_path)
    state,summary,history,previous,next_step=_load(parts)
    assert next_step==1
    assert not history
    assert previous is None
    np.testing.assert_array_equal(state[0],parts[4])


def test_restart_one_fdm_step_and_quarantine_unaccepted_stage(tmp_path):
    parts=_prepare(tmp_path)
    out,_,_,dyn,q,_,_,_,summary=parts
    _save_accepted(out,dyn,summary,q)
    np.savez_compressed(out/"BATHE_SUBSTEP_0002_GAMMA.npz",garbage=np.array(1.))
    state,summary,rows,previous,next_step=_load(parts)
    assert next_step==2 and len(rows)==1 and previous is None
    assert state[0].shape==q.shape
    assert not (out/"BATHE_SUBSTEP_0002_GAMMA.npz").exists()
    assert (out/"UNACCEPTED_INTERRUPTED_STAGE_FILES"/"BATHE_SUBSTEP_0002_GAMMA.npz").exists()


def test_resume_flag_can_be_enabled_on_exact_original_case(tmp_path):
    parts=_prepare(tmp_path)
    out=parts[0]
    stored=json.loads((out/"REFERENCE_CASE.json").read_text(encoding="utf-8"))
    stored["dynamics"].pop("bathe_resume_accepted")
    (out/"REFERENCE_CASE.json").write_text(json.dumps(stored),encoding="utf-8")
    state,_,history,_,start=_load(parts)
    assert start==1 and not history
    np.testing.assert_array_equal(state[0],parts[4])


def test_changed_case_is_rejected(tmp_path):
    parts=_prepare(tmp_path)
    parts[1]["load_definition"]["q"]=200000
    with pytest.raises(ValueError,match="PHYSICAL_CASE_CHANGED"):
        _load(parts)


def test_missing_accepted_state_rejected(tmp_path):
    parts=_prepare(tmp_path)
    out,_,_,dyn,q,_,_,_,summary=parts
    _save_accepted(out,dyn,summary,q)
    (out/"REFERENCE_DYNAMIC_STEP_0001.npz").unlink()
    with pytest.raises(ValueError,match="ACCEPTED_STATE_MISSING"):
        _load(parts)


def test_bad_time_and_load_rejected(tmp_path):
    parts=_prepare(tmp_path)
    out,_,_,dyn,q,_,_,_,summary=parts
    _save_accepted(out,dyn,summary,q)
    with np.load(out/"REFERENCE_DYNAMIC_STEP_0001.npz") as raw:
        values={key:np.array(raw[key]) for key in raw.files}
    values["load_factor"]=np.array(.8)
    np.savez_compressed(out/"REFERENCE_DYNAMIC_STEP_0001.npz",**values)
    with pytest.raises(ValueError,match="LOAD_FACTOR_MISMATCH"):
        _load(parts)


@pytest.mark.parametrize("rank", [1,3])
def test_rank_fixed_continuation_from_accepted_saved_mvi_factors(tmp_path,rank):
    parts=_prepare(tmp_path,method="MVI",rank=rank)
    out,_,_,dyn,q,_,_,_,summary=parts
    _save_accepted(out,dyn,summary,q,rank=rank)
    state,summary,rows,previous,next_step=_load(parts)
    assert next_step==2 and previous is not None
    assert previous.mvi_certificate["accepted"]
    assert previous.mvi_certificate["fixed_rank"]==rank
    np.testing.assert_allclose(
        np.einsum("crx,crs->cxs",previous.x_profiles,previous.y_profiles)*
        previous.scales,previous.mechanical)


def test_corrupted_mvi_factor_rejected(tmp_path):
    parts=_prepare(tmp_path,method="MVI",rank=1)
    out,_,_,dyn,q,_,_,_,summary=parts
    _save_accepted(out,dyn,summary,q,rank=1)
    path=out/"REFERENCE_DYNAMIC_STEP_0001.npz"
    with np.load(path) as raw:
        values={key:np.array(raw[key]) for key in raw.files}
    values["y_profiles"]*=2
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match="MVI_RECONSTRUCTION"):
        _load(parts)
