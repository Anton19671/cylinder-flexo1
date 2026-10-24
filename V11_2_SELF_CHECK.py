#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V11.2 regression: split strong residual into interior/boundary/PATCH-front.

Checks:
- region masks are disjoint and exhaustive for CUF and KL assembled rows;
- a resolved PATCH may exclude only its smoothed load-front rows from the
  Newton stopping norm;
- an under-resolved PATCH whose entire applied load lies on the front falls
  back automatically to the complete residual norm (prevents q=0 acceptance);
- CUF/KL nonlinear elastic and frozen-GB paths expose the split metrics;
- final dispatcher carries the generated PATCH front mask into nonlinear FDM.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np

import final_load_builder as lb
import final_residual_regions as rr
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
from cylinder_final_stage11a_dispatcher import run_final_case

BASE=Path(__file__).resolve().parent
WORK=BASE/'V11_2_SELF_CHECK_WORK'


def _setup(nx=21,ns=21):
    L=100.0; R=50.0; h=1.0; E0=70000.0; nu0=0.3
    basis=teb.get_basis('TE1')
    grid=core.CylinderGrid(L,R,nx,ns)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    quad=s2.gauss_thickness_quadrature(h,3)
    bc_c=cbc.GeneralCylinderEndBC.clamped_clamped(basis.m)
    bc_k=kbc.KLCylinderEndBC.clamped_clamped()
    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)
    GB=np.full((quad.z.size,ns,nx),G0,float)
    return locals()


def _spec(resolved=True):
    if resolved:
        return {
            'type':'PATCH_RADIAL_FACE','q':0.1,'face':'OUTER',
            'x1':15.0,'x2':85.0,'circumferential_coordinate':'THETA',
            'c1':0.3,'c2':3.0,'full_circumference':False,'load_factors':[1.0],
        }
    return {
        'type':'PATCH_RADIAL_FACE','q':0.1,'face':'OUTER',
        'x1':20.0,'x2':80.0,'circumferential_coordinate':'THETA',
        'c1':0.5,'c2':2.5,'full_circumference':False,'load_factors':[1.0],
    }


def _partition_guard(m):
    if np.any(m.interior&m.boundary) or np.any(m.interior&m.load_front) or np.any(m.boundary&m.load_front):
        raise AssertionError('V11.2 masks overlap')
    if not np.all(m.interior|m.boundary|m.load_front):
        raise AssertionError('V11.2 masks do not cover all rows')


def _direct_checks():
    p=_setup(21,21); spec=_spec(True)
    cc=lb.build_generated_loads(spec,theory='CUF',L=p['L'],R=p['R'],h=p['h'],nx=21,ns=21,stencil=7,basis=p['basis'],return_masks=True)
    spec_kl={k:v for k,v in spec.items() if k!='face'}
    kk=lb.build_generated_loads(spec_kl,theory='KL',L=p['L'],R=p['R'],h=p['h'],nx=21,ns=21,stencil=7,return_masks=True)
    fc=np.asarray(cc[4]['front_surface_mask'],bool); fk=np.asarray(kk[4]['front_surface_mask'],bool)
    if not np.array_equal(fc,fk) or not np.any(fc):
        raise AssertionError('generated PATCH front mask missing/inconsistent')

    mc=rr.resolve_front_exclusion(rr.cuf_masks(p['ops'],p['basis'],fc),cc[0][0])
    mk=rr.resolve_front_exclusion(rr.kl_masks(p['ops'],fk),kk[0][0])
    _partition_guard(mc); _partition_guard(mk)
    if not mc.exclude_load_front_from_convergence or not mk.exclude_load_front_from_convergence:
        raise AssertionError('resolved PATCH did not enable front split')

    # Synthetic semantics: a huge front residual must be reported but must not
    # dominate stopping when ordinary rows are already below tolerance.
    for name,m in [('CUF',mc),('KL',mk)]:
        r=np.zeros(m.size,float)
        r[m.interior]=1e-12; r[m.boundary]=2e-12; r[m.load_front]=1e-2
        rm=rr.metrics(r,m,1.0)
        if rm.residual_inf!=1e-2 or rm.residual_convergence_inf>2.0001e-12:
            raise AssertionError(f'{name}: load-front split semantics failed')
        if not rr.converged(rm,1e-9):
            raise AssertionError(f'{name}: front incorrectly blocked convergence')

    cs=ce.CUFStrongFDMNewtonSettings(1e-9,1e9,1e-8,10)
    ks=ke.KLStrongFDMNewtonSettings(1e-9,1e9,1e-8,10)
    zc=np.zeros((3,p['basis'].m,21,21)); zk=np.zeros((3,21,21))
    cel=ce.solve_full_gl_elastic_strong_fdm(
        zc,p['ops'],p['basis'],p['bc_c'],E0=p['E0'],nu0=p['nu0'],quadrature=p['quad'],
        generalized_load=cc[0][0],settings=cs,load_front_surface_mask=fc)
    kel=ke.solve_vk_elastic_strong_fdm(
        zk,p['ops'],p['bc_k'],E=p['E0'],nu=p['nu0'],h=p['h'],generalized_load=kk[0][0],
        settings=ks,load_front_surface_mask=fk)
    cbg=cb.solve_frozen_gb_full_gl_strong_fdm(
        zc,p['ops'],p['basis'],p['bc_c'],E0=p['E0'],K0=p['K0'],GB=p['GB'],quadrature=p['quad'],
        generalized_load=cc[0][0],newton_settings=cs,load_front_surface_mask=fc)
    kbg=kb.solve_frozen_gb_vk_strong_fdm(
        zk,p['ops'],p['bc_k'],K0=p['K0'],GB=p['GB'],quadrature=p['quad'],
        generalized_load=kk[0][0],newton_settings=ks,load_front_surface_mask=fk)
    rows={}
    for name,r in [('CUF_ELASTIC',cel),('KL_ELASTIC',kel),('CUF_FROZEN_GB',cbg),('KL_FROZEN_GB',kbg)]:
        if not r.converged:
            raise AssertionError(f'{name} failed: {r.status}')
        if r.residual_convergence_relative>1e-9:
            raise AssertionError(f'{name}: convergence residual above tolerance')
        if not np.isclose(r.residual_convergence_inf,max(r.residual_inf_interior,r.residual_inf_boundary),rtol=0,atol=1e-30):
            raise AssertionError(f'{name}: convergence norm is not max(interior,boundary)')
        rows[name]={
            'iterations':int(r.iterations),'residual_all':float(r.residual_inf),
            'residual_interior':float(r.residual_inf_interior),'residual_boundary':float(r.residual_inf_boundary),
            'residual_load_front':float(r.residual_inf_load_front),'residual_convergence':float(r.residual_convergence_inf),
        }

    # Under-resolved PATCH safety fallback: all nonzero load rows lie on the
    # front on this coarse grid. Front diagnostics stay active, but convergence
    # must revert to the complete norm so the zero state cannot be accepted.
    q=_setup(11,11); tiny=_spec(False)
    tiny_kl={k:v for k,v in tiny.items() if k!='face'}
    kt=lb.build_generated_loads(tiny_kl,theory='KL',L=q['L'],R=q['R'],h=q['h'],nx=11,ns=11,stencil=7,return_masks=True)
    mt=rr.resolve_front_exclusion(rr.kl_masks(q['ops'],kt[4]['front_surface_mask']),kt[0][0])
    if mt.exclude_load_front_from_convergence:
        raise AssertionError('under-resolved PATCH unsafe front exclusion was enabled')
    r0=np.zeros(mt.size,float); r0[mt.load_front]=1.0
    rm0=rr.metrics(r0,mt,1.0)
    if rr.converged(rm0,1e-9) or rm0.residual_convergence_inf!=1.0:
        raise AssertionError('under-resolved PATCH could falsely accept the zero state')

    return rows, int(np.count_nonzero(fc)), int(np.count_nonzero(kt[4]['front_surface_mask']))


def _dispatcher_check():
    d=WORK/'dispatcher'; d.mkdir(parents=True,exist_ok=True)
    cfg={
        'theory':'KL','method':'FDM','mode':'NONLINEAR_ELASTIC',
        'geometry':{'L':100.0,'R':50.0,'h':1.0},
        'material':{'E0':70000.0,'nu0':0.3},
        'units':{'system':'N_MM_MPA'},
        'fdm_grid':{'nx':21,'ns':21,'stencil':7},
        'mvi_grid':{'nx':21,'ns':21,'stencil':7},
        'quadrature':{'type':'GAUSS','n_gauss':3},
        'boundary_conditions':{'left':'CLAMPED','right':'CLAMPED','advanced':None},
        'load_definition':{k:v for k,v in _spec(True).items() if k!='face'},
        'initial_state':{'source':'ZERO','npz':None},
        'newton':{'residual_tolerance_relative':1e-9,'increment_tolerance':1e9,'fd_step':1e-8,'max_iterations':10},
        'output_dir':'OUT',
    }
    cp=d/'case.json'; cp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    summary=run_final_case(cp)
    if not summary.get('converged',False):
        raise AssertionError(f"dispatcher nonlinear PATCH failed: {summary.get('status')}")
    step=json.loads((d/'OUT'/'STEP_0001.json').read_text(encoding='utf-8'))
    need=('residual_inf_interior','residual_inf_boundary','residual_inf_load_front','residual_convergence_inf','residual_convergence_relative')
    if any(k not in step for k in need):
        raise AssertionError(f'dispatcher did not serialize V11.2 metrics: {step.keys()}')
    with np.load(d/'OUT'/'GENERATED_LOADS.npz',allow_pickle=False) as z:
        if 'load_front_surface_mask' not in z.files or not np.any(z['load_front_surface_mask']):
            raise AssertionError('dispatcher did not persist generated load-front mask')
    return {k:step[k] for k in need}



def run_quick_checks(verbose=False):
    """Fast structural guard used by FINAL_SELF_CHECK.

    The standalone run_checks() remains the full nonlinear/dispatcher regression.
    This quick variant avoids repeating those expensive solves inside the umbrella
    FINAL_SELF_CHECK while still guarding the V11.2 partition and safety rules.
    """
    p=_setup(11,11)
    # Build a synthetic, deliberately well-resolved surface front mask so the
    # row-partition logic itself is checked without another nonlinear solve.
    front=np.zeros((11,11),bool)
    front[3:8,4]=True; front[3:8,6]=True
    front[3,4:7]=True; front[7,4:7]=True
    mc=rr.cuf_masks(p['ops'],p['basis'],front)
    mk=rr.kl_masks(p['ops'],front)
    _partition_guard(mc); _partition_guard(mk)
    # Explicit semantics: front is diagnostic only when exclusion is enabled.
    for name,m in [('CUF',mc),('KL',mk)]:
        r=np.zeros(m.size,float)
        r[m.interior]=1e-12; r[m.boundary]=2e-12; r[m.load_front]=1e-2
        rm=rr.metrics(r,m,1.0)
        if rm.residual_inf != 1e-2 or rm.residual_convergence_inf > 2.0001e-12:
            raise AssertionError(f'{name}: quick V11.2 split semantics failed')
    # Safety fallback must re-include the front when all applied load lives there.
    q=np.zeros(mk.size,float); q[mk.load_front]=1.0
    safe=rr.resolve_front_exclusion(mk,q)
    if safe.exclude_load_front_from_convergence:
        raise AssertionError('V11.2 quick guard: unsafe front exclusion enabled')
    if verbose: print('V11_2_QUICK_SELF_CHECK: PASS')
    return {'status':'PASS','mode':'QUICK_STRUCTURAL_GUARD'}

def run_checks(verbose=True):
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True)
    try:
        rows,nfront,nfront_coarse=_direct_checks()
        disp=_dispatcher_check()
        rep={
            'status':'PASS','rule':'Newton stop uses max(interior,boundary); PATCH front is diagnostic only when a non-front loaded strong row exists',
            'resolved_patch_front_surface_nodes':nfront,'coarse_patch_front_surface_nodes':nfront_coarse,
            'direct_cases':rows,'dispatcher_step':disp,
            'under_resolved_patch_fallback':'FULL_RESIDUAL',
        }
        if verbose:
            print('V11_2_SELF_CHECK: PASS')
            print(f'  resolved PATCH front nodes: {nfront}; under-resolved fallback front nodes: {nfront_coarse}')
            for name,row in rows.items():
                print(f"  {name}: it={row['iterations']}, all={row['residual_all']:.3e}, conv={row['residual_convergence']:.3e}, front={row['residual_load_front']:.3e}")
            print('  under-resolved PATCH: front exclusion safely disabled')
            print('  dispatcher PATCH mask propagation + STEP diagnostics: PASS')
        return rep
    finally:
        shutil.rmtree(WORK,ignore_errors=True)


if __name__=='__main__':
    rep=run_checks(True)
    (BASE/'V11_2_VERIFICATION_RESULTS.json').write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8')
    print('V11_2_REPORT_JSON='+json.dumps(rep,sort_keys=True))
