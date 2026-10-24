#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 regression: relative Newton residual criterion.

Checks the audit repair:
- scale is fixed once before Newton: max(||load||inf, ||R_initial||inf, 1),
- convergence uses residual_tolerance_relative * scale,
- equivalent N-mm-MPa and SI descriptions take the same Newton iteration count
  and give the same physical displacement for CUF/KL elastic and frozen-GB,
- old CASE key newton.residual_tolerance is rejected rather than silently reused.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import final_residual_scaling as rs
import final_load_builder as lb
import final_te_basis_provider as teb
import cylinder_te1_mvi_core as core
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage2b as s2
import cylinder_cuf_stage7c_general_boundary as cbc
import cylinder_kl_stage8b1_boundary as kbc
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as ce
import cylinder_kl_stage9c_strong_fdm_vk_elastic as ke
import cylinder_cuf_stage10d_strong_fdm_full_gl_birger as cb
import cylinder_kl_stage9d2_strong_fdm_vk_birger as kb
import cylinder_final_stage11a_dispatcher as disp

BASE=Path(__file__).resolve().parent


def _problem(length_scale: float):
    sc=float(length_scale)
    L=100.0*sc; R=50.0*sc; h=1.0*sc
    E0=70000.0 if sc==1.0 else 70.0e9
    nu0=0.3
    pressure=1.0 if sc==1.0 else 1.0e6  # 1 MPa
    grid=core.CylinderGrid(L,R,7,7)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    quad=s2.gauss_thickness_quadrature(h,3)
    basis=teb.get_basis('TE1')
    bc_c=cbc.GeneralCylinderEndBC.clamped_clamped(basis.m)
    bc_k=kbc.KLCylinderEndBC.clamped_clamped()
    spec={"type":"UNIFORM_RADIAL_FACE","q":pressure,"face":"OUTER","load_factors":[1.0]}
    loadc=lb.build_generated_loads(spec,theory='CUF',L=L,R=R,h=h,nx=7,ns=7,stencil=7,basis=basis)[0][0]
    spec_kl={k:v for k,v in spec.items() if k!='face'}
    loadk=lb.build_generated_loads(spec_kl,theory='KL',L=L,R=R,h=h,nx=7,ns=7,stencil=7)[0][0]
    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)
    GB=np.full((quad.z.size,7,7),G0,float)
    return locals()


def _wmax_cuf(q): return float(np.max(np.abs(np.asarray(q)[2,0])))
def _wmax_kl(U): return float(np.max(np.abs(np.asarray(U)[2])))


def _relative_difference(a,b):
    return abs(float(a)-float(b))/max(abs(float(a)),abs(float(b)),1e-300)


def _run_pair(sc: float):
    p=_problem(sc)
    # Make increment non-limiting here: this check isolates the V2 residual
    # criterion. Increment scaling is a separate question and is not V2.
    cs=ce.CUFStrongFDMNewtonSettings(1e-9,1e9,1e-8,10)
    ks=ke.KLStrongFDMNewtonSettings(1e-9,1e9,1e-8,10)
    zc=np.zeros_like(p['loadc']); zk=np.zeros_like(p['loadk'])
    c_el=ce.solve_full_gl_elastic_strong_fdm(
        zc,p['ops'],p['basis'],p['bc_c'],E0=p['E0'],nu0=p['nu0'],
        quadrature=p['quad'],generalized_load=p['loadc'],settings=cs)
    k_el=ke.solve_vk_elastic_strong_fdm(
        zk,p['ops'],p['bc_k'],E=p['E0'],nu=p['nu0'],h=p['h'],
        generalized_load=p['loadk'],settings=ks)
    c_bg=cb.solve_frozen_gb_full_gl_strong_fdm(
        zc,p['ops'],p['basis'],p['bc_c'],E0=p['E0'],K0=p['K0'],GB=p['GB'],
        quadrature=p['quad'],generalized_load=p['loadc'],newton_settings=cs)
    k_bg=kb.solve_frozen_gb_vk_strong_fdm(
        zk,p['ops'],p['bc_k'],K0=p['K0'],GB=p['GB'],quadrature=p['quad'],
        generalized_load=p['loadk'],newton_settings=ks)
    for name,r in [('CUF_ELASTIC',c_el),('KL_ELASTIC',k_el),('CUF_FROZEN_GB',c_bg),('KL_FROZEN_GB',k_bg)]:
        if not r.converged:
            raise AssertionError(f'{name} did not converge: {r.status}')
        if not (r.residual_relative <= 1e-9):
            raise AssertionError(f'{name}: relative residual {r.residual_relative:.3e} > 1e-9')
        if not np.isfinite(r.residual_scale) or r.residual_scale<=0:
            raise AssertionError(f'{name}: invalid residual scale')
    return {
        'CUF_ELASTIC':(c_el,_wmax_cuf(c_el.q)),
        'KL_ELASTIC':(k_el,_wmax_kl(k_el.U)),
        'CUF_FROZEN_GB':(c_bg,_wmax_cuf(c_bg.state)),
        'KL_FROZEN_GB':(k_bg,_wmax_kl(k_bg.state)),
    }


def run_checks(verbose=True):
    # Direct formula guard.
    scale=rs.residual_scale(np.array([[-2.0,0.5]]),np.array([1.0,-3.0]))
    if scale!=3.0 or not np.isclose(rs.threshold(1e-9,scale),3e-9,rtol=0.0,atol=1e-24):
        raise AssertionError('V2 residual scaling formula mismatch')

    mm=_run_pair(1.0)
    si=_run_pair(1e-3)
    rows={}
    for name in mm:
        rm,wm=mm[name]; rslt,ws=si[name]
        if rm.iterations!=rslt.iterations:
            raise AssertionError(f'{name}: Newton iteration mismatch mm/MPa={rm.iterations}, SI={rslt.iterations}')
        # Convert SI displacement back to mm.
        wsi_mm=ws/1e-3
        wrel=_relative_difference(wm,wsi_mm)
        if wrel>2e-10:
            raise AssertionError(f'{name}: physical displacement mismatch {wrel:.3e}')
        # For this 1 MPa equivalent test, residual scales must transform by 1e6.
        sratio=rslt.residual_scale/rm.residual_scale
        if abs(sratio/1e6-1.0)>2e-12:
            raise AssertionError(f'{name}: residual scale ratio {sratio:.12g} != 1e6')
        rows[name]={
            'iterations':rm.iterations,
            'w_mm':wm,
            'w_si_as_mm':wsi_mm,
            'w_rel_difference':wrel,
            'residual_relative_mm':rm.residual_relative,
            'residual_relative_si':rslt.residual_relative,
            'residual_scale_mm':rm.residual_scale,
            'residual_scale_si':rslt.residual_scale,
        }

    # Strict CASE migration guard from the audit.
    legacy={'newton':{'residual_tolerance':1e-6,'increment_tolerance':1e-9,'fd_step':1e-8,'max_iterations':4}}
    try:
        disp._newton_settings(legacy,'CUF')
    except disp.FinalCaseError as exc:
        if 'residual_tolerance_relative' not in str(exc):
            raise AssertionError('legacy tolerance rejected with unclear message') from exc
    else:
        raise AssertionError('legacy absolute residual_tolerance was silently accepted')

    modern={'newton':{'residual_tolerance_relative':1e-9,'increment_tolerance':1e-9,'fd_step':1e-8,'max_iterations':4}}
    st=disp._newton_settings(modern,'CUF')
    if st.residual_tolerance_relative!=1e-9:
        raise AssertionError('modern relative residual tolerance not propagated')

    rep={'status':'PASS','criterion':'||R||inf <= tol_rel * max(||load||inf, ||R_initial||inf, 1)','recommended_default':1e-9,'cases':rows,'legacy_case_key_rejected':True}
    if verbose:
        print('V2_SELF_CHECK: PASS')
        for name,row in rows.items():
            print(f"  {name}: iterations={row['iterations']}, w(mm)={row['w_mm']:.12g}, relR mm/SI={row['residual_relative_mm']:.3e}/{row['residual_relative_si']:.3e}")
        print('  legacy newton.residual_tolerance: REJECTED (required migration)')
    return rep


if __name__=='__main__':
    rep=run_checks(True)
    (BASE/'V2_VERIFICATION_RESULTS.json').write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8')
    print('V2_REPORT_JSON='+json.dumps(rep,sort_keys=True))
