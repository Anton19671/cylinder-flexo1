#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification for Stage9F: KL + Green-Lagrange + Birger + strong FDM."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np

import V3_SELF_CHECK as v3
import cylinder_te1_stage5a_birger as b5
import cylinder_kl_stage9e_strong_fdm_gl_elastic as gle
import cylinder_kl_stage9f_strong_fdm_gl_birger as glb
import cylinder_final_stage11a_dispatcher as dispatch

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'KL_GL_BIRGER_FDM_SELF_CHECK_WORK'

def _rel_flat(a,b):
    aa=np.asarray(a,float).reshape(-1); bb=np.asarray(b,float).reshape(-1)
    return float(np.linalg.norm(aa-bb,np.inf)/max(np.linalg.norm(bb,np.inf),1e-300))

def _linear_settings(G0,E0):
    return b5.BirgerSettings(
        epsilon_tol=1e-14,omega=1.0,G_ref=G0,sigma_ref=E0,
        tol_G=1e-11,tol_sigma=1e-11,tol_u=1e-11,
        max_outer_iterations=4,epsilon_i_limit=1.0,delta_J_limit=1.0,
        initial_slope_rtol=0.05,
    )

def run_checks(verbose=True):
    p=v3._problem(1.0)
    ops,U,bc=p['ops'],p['U'],p['bc_k']
    E0,nu0,h=p['E0'],p['nu0'],p['h']
    G0,K0,quad,GB=p['G0'],p['K0'],p['quad'],p['GB']
    load0=p['loadk']

    # 1) Exact frozen-GB GL tangent against independent 5-point derivative.
    rng=np.random.default_rng(20260830)
    d=rng.normal(size=U.shape); d/=np.linalg.norm(d)
    y=glb.analytic_tangent_action_frozen_gb(U,d,ops,bc,K0=K0,GB=GB,quadrature=quad)
    def res(A):
        return glb.frozen_birger_strong_residual_gl(
            A,ops,bc,K0=K0,GB=GB,quadrature=quad,generalized_load=load0
        )[0]
    delta=1e-5
    ref=(-res(U+2*delta*d)+8*res(U+delta*d)-8*res(U-delta*d)+res(U-2*delta*d))/(12*delta)
    tangent_rel=_rel_flat(y,ref)
    if tangent_rel>3e-7:
        raise AssertionError(f'Stage9F directional tangent error {tangent_rel:.3e}')

    Ja,_=glb.analytic_jacobian_frozen_gb(U,ops,bc,K0=K0,GB=GB,quadrature=quad,generalized_load=load0)
    Jf,_=glb.finite_difference_jacobian_frozen_gb(U,ops,bc,K0=K0,GB=GB,quadrature=quad,generalized_load=load0,fd_step=1e-5)
    jac_rel=_rel_flat(Ja,Jf)
    if jac_rel>3e-7:
        raise AssertionError(f'Stage9F analytic/FD Jacobian error {jac_rel:.3e}')

    # 2) Elastic-limit equivalence. Frozen Birger with Phi=3G0*eps must equal
    #    the GL Hooke solver on the same thickness quadrature.
    load=np.zeros_like(U); load[2]=0.5
    newton=gle.KLStrongFDMNewtonSettings(1e-9,1e-11,1e-8,12)
    elastic=gle.solve_gl_elastic_strong_fdm(
        np.zeros_like(U),ops,bc,E=E0,nu=nu0,h=h,
        generalized_load=load,settings=newton,
        resultants_method='QUADRATURE',quadrature=quad,
    )
    if not elastic.converged:
        raise AssertionError(elastic.status)
    def elastic_phi(eps): return 3.0*G0*np.asarray(eps,float)
    GB0=np.full((len(quad.z),ops.grid.ns,ops.grid.nx),G0)
    birger,counter=glb.solve_gl_birger_strong_fdm(
        np.zeros_like(U),GB0,ops=ops,bc=bc,E0=E0,nu0=nu0,h=h,
        quadrature=quad,generalized_load=load,newton_settings=newton,
        material_diagram=elastic_phi,birger_settings=_linear_settings(G0,E0),
    )
    if not birger.converged:
        raise AssertionError(birger.status)
    elastic_limit_rel=_rel_flat(birger.state,elastic.U)
    if elastic_limit_rel>5e-11:
        raise AssertionError(f'elastic-limit state mismatch {elastic_limit_rel:.3e}')
    gb_elastic_rel=float(np.max(np.abs(birger.GB/G0-1.0)))
    if gb_elastic_rel>5e-12:
        raise AssertionError(f'elastic-limit GB mismatch {gb_elastic_rel:.3e}')

    # 3) A monotone bilinear intensity curve must actually update GB.
    loadp=np.zeros_like(U); loadp[2]=1e-4
    ey=1e-8; slope0=3.0*G0; slope1=0.25*slope0
    def plastic_phi(eps):
        e=np.asarray(eps,float)
        return np.where(e<=ey,slope0*e,slope0*ey+slope1*(e-ey))
    plastic_settings=b5.BirgerSettings(
        epsilon_tol=1e-14,omega=0.8,G_ref=G0,sigma_ref=1.0,
        tol_G=1e-8,tol_sigma=1e-8,tol_u=1e-8,max_outer_iterations=80,
        epsilon_i_limit=1e-3,delta_J_limit=1.0,initial_slope_rtol=0.05,
    )
    plastic,pcounter=glb.solve_gl_birger_strong_fdm(
        np.zeros_like(U),GB0,ops=ops,bc=bc,E0=E0,nu0=nu0,h=h,
        quadrature=quad,generalized_load=loadp,newton_settings=newton,
        material_diagram=plastic_phi,birger_settings=plastic_settings,
    )
    if not plastic.converged:
        raise AssertionError(plastic.status)
    gb_ratio_min=float(np.min(plastic.GB/G0)); gb_ratio_max=float(np.max(plastic.GB/G0))
    if not (gb_ratio_min<0.95 and gb_ratio_max<0.95):
        raise AssertionError(f'Birger field did not soften: {gb_ratio_min}, {gb_ratio_max}')

    # 4) Boundary-condition smoke.  Keep the left end clamped to remove
    #    rigid-mode ambiguity and exercise every standard/legacy right-end
    #    support through the same Stage9F strong boundary rows.
    support_names=('CLAMPED','SIMPLY_SUPPORTED','FREE','SYMMETRY','GUIDED',
                   'PINNED_S1','PINNED_S2','PINNED_S3','PINNED_S4')
    support_results={}
    support_load=np.zeros_like(U); support_load[2]=1e-4
    support_newton=gle.KLStrongFDMNewtonSettings(1e-8,1e-11,1e-8,10)
    support_birger=b5.BirgerSettings(
        epsilon_tol=1e-12,omega=1.0,G_ref=G0,sigma_ref=1.0,
        tol_G=1e-8,tol_sigma=1e-8,tol_u=1e-8,max_outer_iterations=3,
        epsilon_i_limit=1.0,delta_J_limit=1.0,initial_slope_rtol=0.05,
    )
    for right in support_names:
        bc_case=dispatch._make_kl_simple_bc(('CLAMPED',right),{})
        sr,_=glb.solve_gl_birger_strong_fdm(
            np.zeros_like(U),GB0,ops=ops,bc=bc_case,E0=E0,nu0=nu0,h=h,
            quadrature=quad,generalized_load=support_load,newton_settings=support_newton,
            material_diagram=elastic_phi,birger_settings=support_birger,
        )
        if not sr.converged:
            raise AssertionError(f'Stage9F support CLAMPED/{right}: {sr.status}')
        support_results[right]={
            'status':sr.status,'residual_relative':float(getattr(sr,'residual_relative',np.nan))
        }

    # 5) Unified dispatcher route and real FINAL_CASE execution.
    if dispatch._kl_child_mode({'geometry_model':'GREEN_LAGRANGE','material_model':'BIRGER','method':'FDM'})!='KL_GL_BIRGER':
        raise AssertionError('Stage9F route not enabled')
    shutil.rmtree(WORK,ignore_errors=True); WORK.mkdir(parents=True)
    cfg={
      'theory':'KL','method':'FDM','mode':'NONLINEAR_BIRGER',
      'geometry_model':'GREEN_LAGRANGE','material_model':'BIRGER',
      'units':{'system':'N_MM_MPA'},
      'geometry':{'L':100.0,'R':50.0,'h':1.0},
      'material':{'E0':E0,'nu0':nu0,'source':'USER'},
      'fdm_grid':{'nx':7,'ns':7,'stencil':7},'mvi_grid':{'nx':7,'ns':7,'stencil':7},
      'quadrature':{'type':'GAUSS','n_gauss':7},
      'boundary_conditions':{'left':'CLAMPED','right':'CLAMPED','advanced':None},
      'load_definition':{'type':'UNIFORM_RADIAL_FACE','q':1e-4,'load_factors':[1.0]},
      'initial_state':{'source':'ZERO','npz':None},
      'initial_GB':{'source':'ELASTIC_G0','npz':None},
      'newton':{'residual_tolerance_relative':1e-9,'increment_tolerance':1e-12,'fd_step':1e-8,'max_iterations':8},
      'birger':{
        'epsilon_tol':1e-12,'omega':1.0,'G_ref':G0,'sigma_ref':1.0,
        'tol_G':1e-9,'tol_sigma':1e-9,'tol_u':1e-9,'max_outer_iterations':3,
        'epsilon_i_limit':1.0,'delta_J_limit':1.0,'initial_slope_rtol':0.05,
        'curve':{'mode':'FORMULA','formula':'slope*eps','parameters':{'slope':3.0*G0}},
      },
      'step_predictor':{'mode':'ERROR'},
      'postprocess':{'deformation_scale':1.0,'gb_ratio_thresholds':[],'control_points':[]},
      'output_dir':'OUT',
    }
    case=WORK/'case.json'; case.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    check=dispatch.validate_final_case(case)
    if not check.get('valid',False):
        raise AssertionError('Stage9F FINAL_CASE validation failed')
    summary=dispatch.run_final_case(case)
    if not summary.get('converged',False) or summary.get('accepted_steps')!=1:
        raise AssertionError(f"Stage9F FINAL_CASE failed: {summary.get('status')}")
    state_file=WORK/'OUT'/'STEP_0001_STATE.npz'
    if not state_file.exists():
        raise AssertionError('Stage9F dispatcher did not save accepted state')

    report={
      'status':'PASS','tangent_rel_inf':tangent_rel,'jacobian_rel_inf':jac_rel,
      'elastic_limit_state_rel_inf':elastic_limit_rel,'elastic_limit_GB_rel':gb_elastic_rel,
      'elastic_limit_outer_iterations':int(birger.outer_iterations),
      'plastic_outer_iterations':int(plastic.outer_iterations),
      'plastic_GB_ratio_min':gb_ratio_min,'plastic_GB_ratio_max':gb_ratio_max,
      'support_smoke':support_results,
      'dispatcher_status':summary.get('status'),'dispatcher_residual_relative':summary['steps'][0].get('residual_relative'),
    }
    (ROOT/'KL_GL_BIRGER_FDM_SELF_CHECK_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if verbose:
        print('KL_GL_BIRGER_FDM_SELF_CHECK: PASS')
        print(f'  tangent rel inf          = {tangent_rel:.3e}')
        print(f'  analytic/FD J rel inf    = {jac_rel:.3e}')
        print(f'  elastic-limit state rel  = {elastic_limit_rel:.3e}')
        print(f'  plastic GB/G0            = {gb_ratio_min:.6f} .. {gb_ratio_max:.6f}')
        print(f'  supports tested          = {len(support_results)}')
        print(f'  dispatcher               = {summary.get("status")}')
    return report

if __name__=='__main__':
    run_checks(True)
