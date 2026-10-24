#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KL Linear/Nonlinear x Elastic/Birger dispatcher control run."""
from __future__ import annotations
import json
import shutil
import numpy as np
import FINAL_SELF_CHECK as f
from cylinder_final_stage11a_dispatcher import run_final_case


def _run_birger(mode,elastic_mode):
    E0=70000.0; nu0=0.3; G0=E0/(2.0*(1.0+nu0))
    d=f.WORK/f"KL_{mode}"
    d.mkdir(parents=True,exist_ok=True)
    cfg=f._case_config("KL",elastic_mode)
    cfg["mode"]=mode
    cfg["initial_GB"]={"source":"ELASTIC_G0","npz":None}
    cfg["birger"]={
        "epsilon_tol":1e-12,"omega":1.0,"G_ref":G0,"sigma_ref":1.0,
        "tol_G":1e-10,"tol_sigma":1e-8,"tol_u":1e-8,"max_outer_iterations":2,
        "epsilon_i_limit":1.0,"delta_J_limit":1.0,"initial_slope_rtol":0.05,
        "curve":{"mode":"FORMULA","formula":"slope*eps","parameters":{"slope":3.0*G0}},
    }
    p=d/'case.json'; p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    result=run_final_case(p)
    if not result.get('converged',False) or result.get('accepted_steps')!=1:
        raise AssertionError(f"KL four-mode {mode}: did not converge: {result.get('status')}")
    with np.load(d/'OUT'/'STEP_0001_STATE.npz',allow_pickle=False) as z:
        U=np.asarray(z['U'],float)
    wmax=float(np.max(np.abs(U[2])))
    f._assert_close(f"KL {mode} elastic-curve wmax",wmax,f.REFERENCE[("KL",elastic_mode)])
    return mode,wmax,int(result['steps'][0].get('outer_iterations',-1))


def run_checks(verbose=True):
    if f.WORK.exists():
        shutil.rmtree(f.WORK)
    f.WORK.mkdir(parents=True)
    elastic=[]
    for mode in ('LINEAR_ELASTIC','NONLINEAR_ELASTIC'):
        w,r,it=f._run_nonzero_fdm('KL',mode)
        elastic.append((mode,w,r,it))
    birger=[
        _run_birger('LINEAR_BIRGER','LINEAR_ELASTIC'),
        _run_birger('NONLINEAR_BIRGER','NONLINEAR_ELASTIC'),
    ]
    if verbose:
        print('KL_FOUR_MODE_SELF_CHECK: PASS')
        for mode,w,r,it in elastic:
            print(f'  {mode}: wmax={w:.17g}, residual={r:.3e}, iterations={it}')
        for mode,w,outer in birger:
            print(f'  {mode}: wmax={w:.17g}, outer={outer}')
    return {'status':'PASS','elastic':elastic,'birger':birger}

if __name__=='__main__':
    run_checks(True)
