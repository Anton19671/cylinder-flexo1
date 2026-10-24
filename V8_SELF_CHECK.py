#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import final_load_builder as lb
import final_te_basis_provider as tb
import cylinder_final_stage11a_dispatcher as disp


def run_checks(verbose=True):
    rep={"status":"PASS"}
    basis=tb.get_basis('TE1')
    base={"type":"UNIFORM_RADIAL_FACE","q":1.25,"face":"OUTER","load_factors":[1.0]}
    loads,fac,meta,rad=lb.build_generated_loads(base,theory='CUF',L=100.0,R=50.0,h=1.0,nx=7,ns=8,stencil=7,basis=basis)
    assert meta['load_kinematics']=='DEAD_REFERENCE'
    assert meta['load_direction_update']=='FIXED_IN_REFERENCE_CONFIGURATION'
    rep['generated_default']=meta['load_kinematics']

    explicit=dict(base,kinematics='DEAD_REFERENCE')
    loads2,_,meta2,_=lb.build_generated_loads(explicit,theory='CUF',L=100.0,R=50.0,h=1.0,nx=7,ns=8,stencil=7,basis=basis)
    assert np.array_equal(loads,loads2) and meta2['load_kinematics']=='DEAD_REFERENCE'
    rep['explicit_dead_same_array']=True

    bad=dict(base,kinematics='FOLLOWER_CURRENT_NORMAL')
    try:
        lb.build_generated_loads(bad,theory='CUF',L=100.0,R=50.0,h=1.0,nx=7,ns=8,stencil=7,basis=basis)
    except lb.LoadDefinitionError as exc:
        assert 'follower' in str(exc).lower() and 'not implemented' in str(exc).lower()
        rep['follower_rejected']=str(exc)
    else:
        raise AssertionError('V8 failed: follower load was accepted without follower mechanics/tangent')

    # External generalized-load arrays are also explicitly treated as fixed
    # reference loads by the unified dispatcher.
    with tempfile.TemporaryDirectory(prefix='v8_dead_') as td:
        td=Path(td); lp=td/'loads.npz'
        np.savez_compressed(lp,loads=np.zeros((1,3,7,7)),load_factors=np.array([1.0]))
        cfg={
            'theory':'KL','method':'FDM','mode':'LINEAR_ELASTIC','units':{'system':'N_MM_MPA'},
            'geometry':{'L':100.0,'R':50.0,'h':1.0},'material':{'E0':70000.0,'nu0':0.3},
            'fdm_grid':{'nx':7,'ns':7,'stencil':7},'mvi_grid':{'nx':7,'ns':7,'stencil':7},
            'quadrature':{'type':'GAUSS','n_gauss':3},
            'boundary_conditions':{'left':'CLAMPED','right':'CLAMPED','advanced':None},
            'load_definition':{'type':'EXTERNAL_NPZ','npz':str(lp),'kinematics':'DEAD_REFERENCE'},
            'loads_npz':str(lp),'initial_state':{'source':None,'npz':None},'initial_GB':{'source':None,'npz':None},
            'inner_mvi':{'projected_tolerance':None,'increment_tolerance':None,'tangent_fd_step':None,'max_halfstep_iterations':None,'max_xy_cycles':None},
            'newton':{'residual_tolerance_relative':None,'increment_tolerance':None,'fd_step':None,'max_iterations':None},
            'birger':{'epsilon_tol':None,'omega':None,'G_ref':None,'sigma_ref':None,'tol_G':None,'tol_sigma':None,'tol_u':None,'max_outer_iterations':None,'epsilon_i_limit':None,'delta_J_limit':None},
            'postprocess':{},'output_dir':str(td/'out')
        }
        cp=td/'case.json'; cp.write_text(json.dumps(cfg),encoding='utf-8')
        c=disp._common(cfg)
        _,_,lm,_,_=disp._load_series_for_case(cfg,cp,c,basis=None)
        assert lm['load_kinematics']=='DEAD_REFERENCE'
        rep['external_npz']=lm['load_kinematics']
        cfg['load_definition']['kinematics']='FOLLOWER_CURRENT_NORMAL'
        try:
            disp._load_series_for_case(cfg,cp,c,basis=None)
        except disp.FinalCaseError as exc:
            assert 'follower' in str(exc).lower()
            rep['external_follower_rejected']=True
        else:
            raise AssertionError('V8 failed: external follower declaration accepted')

    if verbose:
        print('V8_SELF_CHECK: PASS')
        print('  generated load semantics: DEAD_REFERENCE')
        print('  explicit DEAD_REFERENCE leaves generalized load unchanged')
        print('  FOLLOWER_CURRENT_NORMAL: REJECTED (not falsely implemented)')
        print('  EXTERNAL_NPZ semantics: DEAD_REFERENCE')
        print('V8_REPORT_JSON='+json.dumps(rep,ensure_ascii=False,sort_keys=True))
    return rep

if __name__=='__main__':
    run_checks(True)
