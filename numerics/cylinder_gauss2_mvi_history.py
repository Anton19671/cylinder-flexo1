"""Fail-closed multi-step physical Gauss IRK4 + fixed-rank X/Y MVI history.

Experimental small-grid opt-in driver only. Newmark CASE/GUI is unaffected.
Every accepted step is independently checked against Gauss collocation and
rank-R endpoint representation before it is published as a durable NPZ.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

from cylinder_gauss2_rk import GAUSS2_A, GAUSS2_B, GAUSS2_C, Gauss2Settings, Gauss2State
from cylinder_gauss2_mvi import Gauss2MVISettings, Gauss2MVIResult, _profile_field
from cylinder_gauss2_variational_mvi import solve_gauss2_variational_mvi_step


@dataclass(frozen=True)
class Gauss2MVIHistoryResult:
    output_dir: Path
    final: Gauss2State
    accepted_steps: int
    requested_steps: int
    summary: dict


def _bounded_field_error(observed, expected, label, *, relative=2.e-9,
                         roundoff_reference=None):
    """Compare mechanical/electric components in their own physical scale."""
    got = np.asarray(observed, float)
    target = np.asarray(expected, float)
    if (got.shape != target.shape or not np.isfinite(got).all() or
            not np.isfinite(target).all()):
        raise ValueError('GAUSS2_HISTORY_'+label+'_INVALID_FIELD')
    if not got.size:
        raise ValueError('GAUSS2_HISTORY_'+label+'_EMPTY_FIELD')
    # Never use max(1,...) for micro-scale displacements / electric signals.
    # In a componentwise 3D field, the last two axes form the surface grid.
    axes = (-2, -1) if got.ndim >= 3 else tuple(range(got.ndim))
    size = np.maximum(np.max(abs(got), axis=axes, keepdims=True),
                      np.max(abs(target), axis=axes, keepdims=True))
    # Collocation can cancel large same-component terms down to zero.
    # Permit only their predictable arithmetic roundoff, not a global
    # unit-sized absolute tolerance that would erase weak physical modes.
    if roundoff_reference is None:
        arithmetic_scale = size
    else:
        reference=np.asarray(roundoff_reference,float)
        if (reference.shape != target.shape or not np.isfinite(reference).all()):
            raise ValueError('GAUSS2_HISTORY_'+label+'_INVALID_REFERENCE')
        arithmetic_scale=np.max(abs(reference),axis=axes,keepdims=True)
    tolerance = (relative * size + 128*np.finfo(float).eps *
                 np.maximum(arithmetic_scale, np.finfo(float).tiny))
    difference = abs(got-target)
    if np.any(difference > tolerance):
        raise ValueError('GAUSS2_HISTORY_'+label+'_KINEMATICS_INCONSISTENT')
    return float(np.max(difference, initial=0))


def verify_gauss2_mvi_step(previous, result, settings, *, rank):
    """Audit actual stored stage/end fields without altering physical state.

    Stage positions and velocities must follow two-stage Gauss collocation.
    The full nodal velocity/acceleration may have rank greater than R.
    The mechanical/electrical endpoint must still be represented by rank-R
    profiles, with no additive field or post-step SVD truncation.
    """
    if (not isinstance(previous, Gauss2State) or
            not isinstance(result, Gauss2MVIResult) or
            not isinstance(settings, Gauss2Settings)):
        raise TypeError('Gauss2State, Gauss2MVIResult and settings required')
    if (result.converged is not True or
            result.method != 'EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI' or
            not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 8):
        raise ValueError('GAUSS2_HISTORY_NOT_ACCEPTED_MVI_STEP')
    if len(result.stages) != 2:
        raise ValueError('GAUSS2_HISTORY_INVALID_STAGE_COUNT')
    dt = settings.dt_s
    end = result.final
    if not isinstance(end, Gauss2State):
        raise ValueError('GAUSS2_HISTORY_INVALID_FINAL_STATE')
    time_tol = 32*np.finfo(float).eps * max(1., abs(previous.time_s), abs(end.time_s))
    if (abs(end.time_s-(previous.time_s+dt)) > time_tol or
            end.time_s <= previous.time_s):
        raise ValueError('GAUSS2_HISTORY_TIME_GRID_CHANGED')
    q0, v0 = previous.mechanical, previous.velocity
    if q0.shape != end.mechanical.shape or v0.shape != end.velocity.shape:
        raise ValueError('GAUSS2_HISTORY_MECHANICAL_SHAPE_CHANGED')
    if (previous.electrical is None) != (end.electrical is None):
        raise ValueError('GAUSS2_HISTORY_ELECTRIC_SWITCH_CHANGED')
    if previous.electrical is not None and previous.electrical.shape != end.electrical.shape:
        raise ValueError('GAUSS2_HISTORY_ELECTRIC_SHAPE_CHANGED')

    stages = result.stages
    for i, stage in enumerate(stages):
        expected = previous.time_s+dt*GAUSS2_C[i]
        if not math.isfinite(stage.time_s) or abs(stage.time_s-expected) > time_tol:
            raise ValueError('GAUSS2_HISTORY_INCORRECT_STAGE_TIME')
        if (previous.electrical is None) != (stage.electrical is None):
            raise ValueError('GAUSS2_HISTORY_ELECTRIC_STAGE_SWITCH_CHANGED')
        if stage.electrical is not None and stage.electrical.shape != end.electrical.shape:
            raise ValueError('GAUSS2_HISTORY_ELECTRIC_STAGE_SHAPE_CHANGED')

    acceleration=np.stack([np.asarray(s.acceleration,float) for s in stages])
    qstage=np.stack([np.asarray(s.mechanical,float) for s in stages])
    vstage=np.stack([np.asarray(s.velocity,float) for s in stages])
    expected_v=v0[None,...]+dt*np.einsum('ij,j...->i...',GAUSS2_A,acceleration)
    expected_q=(q0[None,...]+dt*GAUSS2_C[:,None,None,None]*v0[None,...]
                +dt*dt*np.einsum('ij,j...->i...',GAUSS2_A@GAUSS2_A,acceleration))
    stage_q_terms=(abs(q0)[None,...]+dt*GAUSS2_C[:,None,None,None]*abs(v0)[None,...]
                   +dt*dt*np.einsum('ij,j...->i...',abs(GAUSS2_A@GAUSS2_A),abs(acceleration)))
    stage_v_terms=abs(v0)[None,...]+dt*np.einsum('ij,j...->i...',abs(GAUSS2_A),abs(acceleration))
    qerr=_bounded_field_error(qstage,expected_q,'STAGE_DISPLACEMENT',
                              roundoff_reference=stage_q_terms)
    verr=_bounded_field_error(vstage,expected_v,'STAGE_VELOCITY',
                              roundoff_reference=stage_v_terms)
    expected_end_q=q0+dt*np.einsum('i,i...->...',GAUSS2_B,vstage)
    expected_end_v=v0+dt*np.einsum('i,i...->...',GAUSS2_B,acceleration)
    qe=_bounded_field_error(end.mechanical,expected_end_q,'END_DISPLACEMENT',
                            roundoff_reference=abs(q0)+dt*np.einsum(
                                'i,i...->...',GAUSS2_B,abs(vstage)))
    ve=_bounded_field_error(end.velocity,expected_end_v,'END_VELOCITY',
                            roundoff_reference=abs(v0)+dt*np.einsum(
                                'i,i...->...',GAUSS2_B,abs(acceleration)))

    combined = (end.mechanical if end.electrical is None else
                np.concatenate((end.mechanical,end.electrical)))
    scale=np.asarray(result.scales,float)
    x=np.asarray(result.x_profiles,float)
    y=np.asarray(result.y_profiles,float)
    if (scale.shape != combined.shape or
            x.shape != (combined.shape[0],rank,combined.shape[1]) or
            y.shape != (combined.shape[0],rank,combined.shape[2]) or
            not np.isfinite(scale).all() or np.any(scale<=0) or
            not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('GAUSS2_HISTORY_PROFILE_SHAPES_OR_SCALES_CHANGED')
    reconstructed=scale*_profile_field(x,y)
    # Uncancelled X/Y terms determine the only justified roundoff floor.
    uncancelled=scale*np.einsum('crx,crs->cxs',abs(x),abs(y))
    rankerr=_bounded_field_error(reconstructed,combined,'FIXED_RANK_ENDPOINT',
                                 roundoff_reference=uncancelled)
    return {'stage_displacement_error':qerr,'stage_velocity_error':verr,
            'end_displacement_error':qe,'end_velocity_error':ve,
            'profile_reconstruction_error':rankerr}


def _atomic_json(path, payload):
    tmp=path.with_name(path.name+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_npz(path, arrays):
    tmp=path.with_name(path.name+'.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def _state_arrays(state):
    arrays={'time_s':np.array(state.time_s),
            'mechanical':state.mechanical.copy(),'velocity':state.velocity.copy()}
    if state.electrical is not None:
        arrays['electrical']=state.electrical.copy()
    return arrays


def run_gauss2_variational_mvi_history(problem, initial, settings, *,
        mvi_settings, n_steps, density_kg_m3, loads_at, output_dir,
        newton_settings=None, mass_damping_rate_per_s=0., eta_s=0.,
        on_accepted_step=None):
    """Opt-in physical IRK4 MVI history; only complete steps get STEP_XXXX.

    On a failed step the last accepted state remains on disk, the summary
    becomes FAILED and the failure is raised. Neither rank nor method is
    silently changed. This driver does not route through Newmark or GUI.
    """
    if (getattr(problem,'kind',None)!='DYNAMIC' or
            not isinstance(initial,Gauss2State) or
            not isinstance(settings,Gauss2Settings) or
            not isinstance(mvi_settings,Gauss2MVISettings)):
        raise TypeError('physical DYNAMIC problem, state and IRK4 MVI settings required')
    if type(n_steps) is not int or not 1 <= n_steps <= 10000:
        raise ValueError('n_steps must be an integer from 1 to 10000')
    if not callable(loads_at):
        raise TypeError('time-dependent physical loads_at(t) required')
    if on_accepted_step is not None and not callable(on_accepted_step):
        raise TypeError('on_accepted_step must be a callback or None')
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('GAUSS2_OUTPUT_DIR_NOT_EMPTY')
    out.mkdir(parents=True,exist_ok=True)
    summary_path=out/'GAUSS2_MVI_HISTORY.json'
    summary={'method':'EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI',
             'time_method':'GAUSS_LEGENDRE_2_STAGE_IRK4',
             'spatial_method':'MVI_XY',
             'rank':mvi_settings.rank,
             'experimental':True,'physical_validation_complete':False,
             'n_steps_requested':n_steps,'n_steps_accepted':0,
             't_start_s':initial.time_s,'dt_s':settings.dt_s,
             'status':'RUNNING','steps':[],
             'endpoint_policy':'NO_RANK_CHANGE_OR_PROJECTION',
             'moving_supports_supported':False}
    _atomic_npz(out/'GAUSS2_MVI_INITIAL.npz',_state_arrays(initial))
    _atomic_json(summary_path,summary)
    previous=None
    current=initial
    for idx in range(1,n_steps+1):
        try:
            result=solve_gauss2_variational_mvi_step(
                problem,current,settings,mvi_settings=mvi_settings,
                density_kg_m3=density_kg_m3,loads_at=loads_at,
                newton_settings=newton_settings,
                mass_damping_rate_per_s=mass_damping_rate_per_s,
                eta_s=eta_s,previous=previous)
            audit=verify_gauss2_mvi_step(current,result,settings,rank=mvi_settings.rank)
            fname=f'GAUSS2_MVI_STEP_{idx:04d}.npz'
            arrays=_state_arrays(result.final)
            arrays.update({'stage_time_s':np.array([s.time_s for s in result.stages]),
                           'stage_mechanical':np.stack([s.mechanical for s in result.stages]),
                           'stage_velocity':np.stack([s.velocity for s in result.stages]),
                           'stage_acceleration':np.stack([s.acceleration for s in result.stages]),
                           'x_profiles':result.x_profiles.copy(),
                           'y_profiles':result.y_profiles.copy(),
                           'state_scales':result.scales.copy()})
            if result.final.electrical is not None:
                arrays['stage_electrical']=np.stack([s.electrical for s in result.stages])
            _atomic_npz(out/fname,arrays)
            summary['steps'].append({'index':idx,'time_s':result.final.time_s,
                                     'state_file':fname,'rank':mvi_settings.rank,
                                     'sweeps':result.sweeps,'iterations':result.iterations,
                                     'stage_kinematics':audit})
            summary['n_steps_accepted']=idx
            _atomic_json(summary_path,summary)
            previous,current=result,result.final
            if on_accepted_step is not None:
                on_accepted_step(idx,result)
        except Exception as exc:
            summary['status']='FAILED'
            summary['failed_step']=idx
            summary['failure_type']=type(exc).__name__
            summary['failure_detail']=str(exc)
            _atomic_json(summary_path,summary)
            raise RuntimeError(f'GAUSS2_MVI_HISTORY_STEP_{idx}_FAILED: {exc}') from exc
    summary['status']='CONVERGED'
    _atomic_json(summary_path,summary)
    return Gauss2MVIHistoryResult(out,current,n_steps,n_steps,summary)


__all__=('Gauss2MVIHistoryResult','verify_gauss2_mvi_step',
         'run_gauss2_variational_mvi_history')
