#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks for V1 and V11a load-generation fixes.

V1: LINE_RADIAL_GENERATOR must have a mesh-independent physical band width;
    the old one-node q=p_line/ds representation is forbidden.
V11a: PATCH fronts are C1 cos^2 transitions over the FD-stencil half-width.

V11.2 is now implemented and tested separately in V11_2_SELF_CHECK.py.
This file remains the generator-focused V1/V11a regression.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import numpy as np

import final_load_builder as lb
from cylinder_final_stage11a_dispatcher import run_final_case

BASE=Path(__file__).resolve().parent
WORK=BASE/'V1_V11_SELF_CHECK_WORK'


def _line_spec(**kw):
    s=dict(
        type='LINE_RADIAL_GENERATOR', p_line=100.0, band_width=800.0,
        c0=0.0, circumferential_coordinate='THETA', x1=0.0, x2=1000.0,
        load_factors=[1.0],
    )
    s.update(kw); return s


def _patch_spec(**kw):
    s=dict(
        type='PATCH_RADIAL_FACE', q=0.5, x1=200.0, x2=800.0,
        circumferential_coordinate='THETA', c1=0.5, c2=1.5,
        full_circumference=False, load_factors=[1.0],
    )
    s.update(kw); return s


def _test_v1_line_band():
    errs=[]
    peak=100.0/800.0
    for ns in (11,21,41,81):
        _,_,meta,q=lb.build_generated_loads(
            _line_spec(),theory='KL',L=1000.0,R=500.0,h=5.0,
            nx=21,ns=ns,stencil=7,
        )
        if not np.isclose(meta['band_peak_pressure_nominal'],peak,rtol=0,atol=1e-15):
            raise AssertionError('V1 nominal band pressure changed with grid')
        if not np.isclose(float(np.max(q)),peak,rtol=2e-14,atol=1e-15):
            raise AssertionError(f'V1 discrete band peak changed at ns={ns}: {np.max(q)}')
        if meta['band_width_physical']!=800.0:
            raise AssertionError('V1 physical band width changed with grid')
        if meta['line_discrete_intensity_rel_error']>0.01:
            raise AssertionError(f"V1 line intensity error too large at ns={ns}: {meta['line_discrete_intensity_rel_error']}")
        errs.append(float(meta['line_discrete_intensity_rel_error']))
    if errs[-1]>=errs[0]:
        raise AssertionError(f'V1 line discretization did not improve from ns=11 to 81: {errs}')

    bad=_line_spec(band_width=400.0)
    try:
        lb.build_generated_loads(bad,theory='KL',L=1000.0,R=500.0,h=5.0,nx=21,ns=11,stencil=7)
    except lb.LoadDefinitionError as exc:
        if 'increase ns' not in str(exc):
            raise AssertionError(f'V1 mesh guard lacks actionable ns message: {exc}')
    else:
        raise AssertionError('V1 accepted a band narrower than two ds')

    missing=_line_spec(); missing.pop('band_width')
    try:
        lb.build_generated_loads(missing,theory='KL',L=1000.0,R=500.0,h=5.0,nx=21,ns=21,stencil=7)
    except lb.LoadDefinitionError:
        pass
    else:
        raise AssertionError('V1 accepted LINE without physical band_width')
    return errs


def _test_v11_patch_smoothing():
    areas=[]; rel=[]
    for ns in (11,21,41,81):
        _,_,meta,q=lb.build_generated_loads(
            _patch_spec(),theory='KL',L=1000.0,R=500.0,h=5.0,
            nx=41,ns=ns,stencil=7,
        )
        areas.append(float(meta['patch_nominal_area']))
        rel.append(float(meta['patch_discrete_resultant_rel_error']))
        if meta['front_smoothing']!='COS2_C1' or meta['load_front_surface_nodes']<=0:
            raise AssertionError(f'V11a no smooth front detected at ns={ns}')
        if np.max(q)>0.5*(1.0+1e-13) or np.min(q)<-1e-15:
            raise AssertionError('V11a smoothed PATCH exceeded requested pressure range')
        if rel[-1]>0.01:
            raise AssertionError(f'V11a discrete PATCH resultant error too large at ns={ns}: {rel[-1]}')
    if not np.allclose(areas,areas[0],rtol=0,atol=1e-12):
        raise AssertionError(f'V11a nominal physical PATCH area changed with grid: {areas}')

    # THETA and physical arc-length S definitions of the same patch must agree.
    _,_,_,qt=lb.build_generated_loads(
        _patch_spec(),theory='KL',L=1000,R=500,h=5,nx=41,ns=41,stencil=7
    )
    ss=_patch_spec(circumferential_coordinate='S',c1=250.0,c2=750.0)
    _,_,_,qs=lb.build_generated_loads(
        ss,theory='KL',L=1000,R=500,h=5,nx=41,ns=41,stencil=7
    )
    if not np.allclose(qt,qs,rtol=0,atol=2e-15):
        raise AssertionError('V11a THETA/S equivalent PATCH definitions disagree')

    # Full-circumference PATCH must not acquire an artificial circumferential front.
    sf=_patch_spec(full_circumference=True)
    sf.pop('c1'); sf.pop('c2')
    _,_,meta,q=lb.build_generated_loads(sf,theory='KL',L=1000,R=500,h=5,nx=41,ns=41,stencil=7)
    if meta['edge_width_c_actual']!=0.0:
        raise AssertionError('V11a full circumference PATCH has circumferential smoothing')
    if not np.allclose(q,q[0:1,:],rtol=0,atol=1e-15):
        raise AssertionError('V11a full circumference PATCH is not circumferentially uniform')
    return areas,rel


def _end_to_end_case(load_definition,name):
    d=WORK/name; d.mkdir(parents=True,exist_ok=True)
    cfg={
        'theory':'KL','method':'FDM','mode':'LINEAR_ELASTIC',
        'geometry':{'L':100.0,'R':50.0,'h':1.0},
        'material':{'E0':70000.0,'nu0':0.3},
        'units':{'system':'N_MM_MPA'},
        'fdm_grid':{'nx':11,'ns':11,'stencil':7},
        'mvi_grid':{'nx':11,'ns':11,'stencil':7},
        'quadrature':{'type':'GAUSS','n_gauss':3},
        'boundary_conditions':{'left':'CLAMPED','right':'CLAMPED','advanced':None},
        'load_definition':load_definition,
        'linear':{'residual_tolerance_relative':1.0e-10},
        'output_dir':'OUT',
    }
    p=d/'case.json'; p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    r=run_final_case(p)
    if not r.get('converged',False) or r.get('accepted_steps')!=1:
        raise AssertionError(f'{name} end-to-end case failed: {r.get("status")}')
    with np.load(d/'OUT'/'STEP_0001_STATE.npz',allow_pickle=False) as z:
        U=np.asarray(z['U'],float)
    wmax=float(np.max(np.abs(U[2])))
    if not np.isfinite(wmax) or wmax<=0.0:
        raise AssertionError(f'{name} end-to-end case produced invalid displacement')
    return wmax


def run_checks(verbose=True):
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True)
    try:
        line_errs=_test_v1_line_band()
        areas,patch_errs=_test_v11_patch_smoothing()
        wp=_end_to_end_case({
            'type':'PATCH_RADIAL_FACE','q':0.01,'x1':20.0,'x2':80.0,
            'circumferential_coordinate':'THETA','c1':0.5,'c2':2.5,
            'full_circumference':False,'load_factors':[1.0],
        },'PATCH')
        wl=_end_to_end_case({
            'type':'LINE_RADIAL_GENERATOR','p_line':1.0,'band_width':80.0,
            'c0':1.0,'circumferential_coordinate':'THETA','x1':20.0,'x2':80.0,
            'load_factors':[1.0],
        },'LINE')
        out={
            'status':'PASS',
            'V1_line_intensity_rel_error_ns_11_21_41_81':line_errs,
            'V11_patch_nominal_area_ns_11_21_41_81':areas,
            'V11_patch_resultant_rel_error_ns_11_21_41_81':patch_errs,
            'end_to_end_KL_patch_wmax':wp,
            'end_to_end_KL_line_band_wmax':wl,
            'V11_2_residual_front_split':'SEE_V11_2_SELF_CHECK_PASS',
        }
        if verbose:
            print('V1_V11_SELF_CHECK: PASS')
            print('  V1 finite physical line band + ns guard: PASS')
            print('  V11a C1 cos^2 PATCH front + fixed physical area: PASS')
            print('  THETA/S equivalent PATCH: PASS')
            print('  KL dispatcher PATCH/LINE end-to-end: PASS')
            print('  V11.2 residual front split: covered by V11_2_SELF_CHECK')
        return out
    finally:
        shutil.rmtree(WORK,ignore_errors=True)


if __name__=='__main__':
    r=run_checks(verbose=True)
    (BASE/'V1_V11_VERIFICATION_RESULTS.json').write_text(
        json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8'
    )
