#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final release integration regression for the cylindrical-shell solver.

Software/integration verification only. Published benchmark validation remains
an external activity and is not embedded into the solver.
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
import numpy as np

import FINAL_SELF_CHECK as base
import FINAL_POSTPROCESS as post
import Z1_SELF_CHECK as z1
import Z2_SELF_CHECK as z2
import Z4_SELF_CHECK as z4
import BC_HINGE_VARIANTS_SELF_CHECK as bc_hinges
import KL_FDM_BC_ENFORCEMENT_SELF_CHECK as kl_bc_enforcement
from cylinder_final_stage11a_dispatcher import (
    run_final_case, run_mvi_rank_series, validate_final_case,
)

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'RELEASE_REGRESSION_WORK'
RESULTS=ROOT/'FINAL_RELEASE_REGRESSION_RESULTS.json'


def _fresh():
    shutil.rmtree(WORK,ignore_errors=True)
    WORK.mkdir(parents=True)


def _write_case(d,cfg):
    d.mkdir(parents=True,exist_ok=True)
    p=d/'case.json'; p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    v=validate_final_case(p)
    if not v.get('valid',False):
        raise AssertionError(f'case validation failed: {d.name}')
    return p


def _fdm_te_linear(model):
    d=WORK/f'FDM_CUF_{model}_LINEAR'
    cfg=base._case_config('CUF_TE1','LINEAR_ELASTIC')
    cfg['theory']='CUF'; cfg['cuf_model']=model; cfg['output_dir']='OUT'
    p=_write_case(d,cfg)
    r=run_final_case(p)
    if not r.get('converged',False) or r.get('accepted_steps')!=1:
        raise AssertionError(f'FDM CUF {model}: not converged')
    st=r['steps'][0]
    ri=float(st.get('residual_inf',np.inf))
    if not np.isfinite(ri) or ri>1e-8:
        raise AssertionError(f'FDM CUF {model}: residual {ri}')
    with np.load(d/'OUT'/'STEP_0001_STATE.npz',allow_pickle=False) as zz:
        q=np.asarray(zz['q'],float)
    m=int(model[2:])+1
    if q.shape!=(3,m,7,7) or not np.all(np.isfinite(q)):
        raise AssertionError(f'FDM CUF {model}: wrong/nonfinite state shape {q.shape}')
    w=float(np.max(np.abs(q[2,0])))
    if w<=0: raise AssertionError(f'FDM CUF {model}: zero w under nonzero load')
    return {'status':r['status'],'residual_inf':ri,'w0_abs_max':w,'state_shape':list(q.shape),'out':str(d/'OUT')}


def _fdm_kl_gl_birger_smoke():
    """Stage9F route: KL + Green-Lagrange + Birger + strong FDM."""
    d=WORK/'FDM_KL_GL_BIRGER'
    cfg=base._case_config('KL','NONLINEAR_ELASTIC')
    E0=70000.0; nu0=0.3; G0=E0/(2.0*(1.0+nu0))
    cfg['mode']='NONLINEAR_BIRGER'
    cfg['geometry_model']='GREEN_LAGRANGE'
    cfg['material_model']='BIRGER'
    cfg['quadrature']={'type':'GAUSS','n_gauss':7}
    cfg['load_definition']['q']=1e-4
    cfg['initial_GB']={'source':'ELASTIC_G0','npz':None}
    cfg['newton']={
        'residual_tolerance_relative':1e-9,'increment_tolerance':1e-12,
        'fd_step':1e-8,'max_iterations':8,
    }
    cfg['birger']={
        'epsilon_tol':1e-12,'omega':1.0,'G_ref':G0,'sigma_ref':1.0,
        'tol_G':1e-9,'tol_sigma':1e-9,'tol_u':1e-9,'max_outer_iterations':3,
        'epsilon_i_limit':1.0,'delta_J_limit':1.0,'initial_slope_rtol':0.05,
        'curve':{'mode':'FORMULA','formula':'slope*eps','parameters':{'slope':3.0*G0}},
    }
    cfg['output_dir']='OUT'
    p=_write_case(d,cfg)
    r=run_final_case(p)
    if not r.get('converged',False) or r.get('accepted_steps')!=1:
        raise AssertionError(f'FDM KL GL Birger: not converged ({r.get("status")})')
    st=r['steps'][0]
    if st.get('status')!='CONVERGED_BIRGER':
        raise AssertionError(f'FDM KL GL Birger: unexpected step status {st.get("status")}')
    with np.load(d/'OUT'/'STEP_0001_STATE.npz',allow_pickle=False) as zz:
        U=np.asarray(zz['U'],float); GB=np.asarray(zz['GB'],float)
    if U.shape!=(3,7,7) or GB.shape!=(7,7,7):
        raise AssertionError(f'FDM KL GL Birger: wrong state shapes {U.shape}, {GB.shape}')
    return {
        'status':r['status'],'step_status':st.get('status'),
        'residual_relative':float(st.get('residual_relative',np.inf)),
        'outer_iterations':int(st.get('outer_iterations',0)),
        'w_abs_max':float(np.max(np.abs(U[2]))),
    }


def _mvi_seed(theory,model='TE1'):
    nx=ns=7; rank=1
    x=np.linspace(0.0,1.0,nx); s=np.arange(ns,dtype=float)
    # quartic x-shape satisfies value and first-derivative zero at both ends;
    # therefore it is valid for KL clamped seed and also for CUF clamped seed.
    shape=x*x*(1.0-x)*(1.0-x)
    if theory=='CUF':
        m=int(model[2:])+1
        X=np.empty((3,m,rank,nx),float); Y=np.empty((3,m,rank,ns),float)
        for a in range(3):
            for t in range(m):
                X[a,t,0]=1e-5*(1.0+0.1*a+0.05*t)*shape
                Y[a,t,0]=1.0+0.05*np.cos(2*np.pi*s/ns+0.2*a+0.1*t)
    else:
        X=np.empty((3,rank,nx),float); Y=np.empty((3,rank,ns),float)
        for a in range(3):
            X[a,0]=1e-5*(1.0+0.1*a)*shape
            Y[a,0]=1.0+0.05*np.cos(2*np.pi*s/ns+0.2*a)
    return X,Y


def _mvi_case(theory,mode,model='TE1'):
    nonlinear=mode.startswith('NONLINEAR')
    elastic_mode='NONLINEAR_ELASTIC' if nonlinear else 'LINEAR_ELASTIC'
    legacy='CUF_TE1' if theory=='CUF' else 'KL'
    cfg=base._case_config(legacy,elastic_mode)
    if theory=='CUF':
        cfg['theory']='CUF'; cfg['cuf_model']=model
    cfg['method']='MVI'; cfg['mode']=mode; cfg['rank']=1
    cfg['load_definition']['q']=1e-7 if nonlinear else 1e-5
    cfg['inner_mvi']={
        'projected_tolerance':1e-7,
        'increment_tolerance':1e-6,
        'tangent_fd_step':1e-7,
        'max_halfstep_iterations':30,
        'max_xy_cycles':30,
    }
    if 'BIRGER' in mode:
        E0=70000.0; nu0=0.3; G0=E0/(2.0*(1.0+nu0))
        cfg['birger']={
            'epsilon_tol':1e-12,'omega':1.0,'G_ref':G0,'sigma_ref':1.0,
            'tol_G':1e-8,'tol_sigma':1e-7,'tol_u':1e-6,'max_outer_iterations':2,
            'epsilon_i_limit':1.0,'delta_J_limit':1.0,'initial_slope_rtol':0.05,
            'curve':{'mode':'FORMULA','formula':'slope*eps','parameters':{'slope':3.0*G0}},
        }
    return cfg


def _mvi_smoke(theory,mode,model='TE1',auto_profiles=False,rank=1):
    tag=f'{theory}_{model}_{mode}' if theory=='CUF' else f'KL_{mode}'
    d=WORK/f'MVI_{tag}'; d.mkdir(parents=True,exist_ok=True)
    cfg=_mvi_case(theory,mode,model)
    cfg['rank']=int(rank)
    if auto_profiles:
        cfg['initial_profiles_npz']=None
    else:
        X,Y=_mvi_seed(theory,model)
        np.savez_compressed(d/'profiles.npz',X=X,Y=Y)
        cfg['initial_profiles_npz']='profiles.npz'
    cfg['output_dir']='OUT'
    p=_write_case(d,cfg)
    r=run_final_case(p)
    if not r.get('converged',False) or r.get('accepted_steps')!=1:
        raise AssertionError(f'MVI {tag}: not converged ({r.get("status")})')
    expected_source='AUTO' if auto_profiles else 'EXTERNAL_NPZ'
    got_source=(r.get('initial_profiles') or {}).get('source')
    if got_source!=expected_source:
        raise AssertionError(f'MVI {tag}: profile source {got_source}, expected {expected_source}')
    if auto_profiles and not (d/'OUT'/'INPUT'/'AUTO_INITIAL_PROFILES_MVI.npz').exists():
        raise AssertionError(f'MVI {tag}: automatic profile file was not saved')
    return {'status':r['status'],'accepted_steps':r['accepted_steps'],'rank':int(rank),'profiles':got_source}


def run(verbose=True):
    t0=time.perf_counter(); _fresh()
    # Dedicated optional-fix guards.
    z1.run_checks(False); z2.run_checks(False); z4.run_checks(False); bc_hinges.run(); kl_bc_enforcement.run(False)

    fdm_te={m:_fdm_te_linear(m) for m in ('TE1','TE2','TE3','TE4')}
    fdm_kl_gl_birger=_fdm_kl_gl_birger_smoke()

    # Four physical branches through the MVI dispatcher for both main theories.
    modes=('LINEAR_ELASTIC','NONLINEAR_ELASTIC','LINEAR_BIRGER','NONLINEAR_BIRGER')
    mvi_four={
        'CUF_TE1':{mode:_mvi_smoke('CUF',mode,'TE1',auto_profiles=(mode=='LINEAR_ELASTIC')) for mode in modes},
        'KL':{
            mode:_mvi_smoke(
                'KL',mode,
                auto_profiles=(mode=='LINEAR_ELASTIC'),
                rank=(2 if mode=='LINEAR_ELASTIC' else 1),
            )
            for mode in modes
        },
    }
    # Higher-order CUF interface smoke: explicit seeds, no hidden seed generator.
    mvi_higher={m:_mvi_smoke('CUF','LINEAR_ELASTIC',m) for m in ('TE2','TE3','TE4')}

    # The GUI's MVI-1...8 action uses the rank-series dispatcher.  Exercise
    # its automatic bank path without paying for all eight numerical solves.
    series_case=WORK/'MVI_KL_LINEAR_ELASTIC'/'case.json'
    series=run_mvi_rank_series(series_case,ranks=(1,))
    if not series.get('all_converged',False):
        raise AssertionError('automatic MVI rank-series profile bank did not converge')
    auto_bank=WORK/'MVI_KL_LINEAR_ELASTIC'/'OUT'/'INPUT'/'AUTO_INITIAL_PROFILE_BANK_MVI_1_8.npz'
    if not auto_bank.exists():
        raise AssertionError('automatic MVI rank-series profile bank was not saved')

    # Postprocess a real current output and assert V8 report semantics.
    post_out=Path(fdm_te['TE1']['out'])
    pr=post.process_output(post_out,make_zip=False,fast=True)
    rep=json.loads(Path(pr['report']).read_text(encoding='utf-8'))
    if rep.get('load_kinematics')!='DEAD_REFERENCE':
        raise AssertionError('postprocess did not persist DEAD_REFERENCE load semantics')
    if pr.get('derived_error') not in (None,''):
        raise AssertionError(f'postprocess derived_error: {pr.get("derived_error")}')
    disp_dir=post_out/'GRAPHS'/'DISPLACEMENTS'
    if not list(disp_dir.glob('*FINAL_W_PROFILES.png')):
        raise AssertionError('engineering FINAL_W_PROFILES.png was not created')
    if not list(disp_dir.glob('*FINAL_W_PROFILES_TRUE_SCALE.png')):
        raise AssertionError('true-scale FINAL_W_PROFILES_TRUE_SCALE.png was not created')

    # GUI module import smoke only; no display/window required.
    import FINAL_GUI  # noqa: F401

    result={
        'status':'PASS',
        'elapsed_s':time.perf_counter()-t0,
        'fdm_cuf_te1_te4_linear':fdm_te,
        'fdm_kl_green_lagrange_birger':fdm_kl_gl_birger,
        'mvi_four_modes':mvi_four,
        'mvi_higher_cuf_linear':mvi_higher,
        'mvi_rank_series_auto':{
            'status':'PASS','ranks':[1],'all_converged':True,
            'profile_bank':str(auto_bank),
        },
        'postprocess':{
            'states':pr['states'],'derived_error':pr['derived_error'],
            'load_kinematics':rep.get('load_kinematics'),
            'numeric_zip_exists':Path(pr['numeric_zip']).exists() if pr['numeric_zip'] else False,
        },
        'gui_import':'PASS',
        'kl_hinge_variants':'PASS',
        'kl_fdm_bc_enforcement':'PASS',
        'notes':[
            'Software/integration regression; not a substitute for external published benchmark validation.',
            'MVI automatic profiles are exercised for CUF-TE1 and KL; expert external NPZ profiles are exercised by the remaining MVI cases.',
            'Follower/current-normal load remains intentionally unsupported; verified load is DEAD_REFERENCE.',
        ],
    }
    RESULTS.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    if verbose:
        print('FINAL_RELEASE_REGRESSION: PASS')
        print('  FDM CUF TE1-TE4 linear: PASS')
        print('  FDM KL Green-Lagrange + Birger Stage9F: PASS')
        print('  MVI CUF-TE1 four modes: PASS')
        print('  MVI KL four modes: PASS')
        print('  MVI CUF TE2-TE4 linear: PASS')
        print('  automatic MVI initial profiles CUF/KL: PASS')
        print('  automatic MVI-1...8 profile bank path: PASS')
        print('  postprocess/report DEAD_REFERENCE: PASS')
        print('  GUI import: PASS')
        print('  KL S1/S2/S3/S4 GUI/postprocess semantics: PASS')
        print('  KL FDM CLAMPED vs S3 numerical BC enforcement: PASS')
        print(f"  elapsed: {result['elapsed_s']:.3f} s")
    return result

if __name__=='__main__':
    run(True)
