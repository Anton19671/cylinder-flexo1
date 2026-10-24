#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3c regression: exact analytic tangents for Birger frozen-GB inner FDM.

The verified Birger algorithm is an outer secant-modulus iteration.  GB is
strictly frozen during every inner solve.  Therefore V3c differentiates the
inner residual with respect to the displacement/CUF state at fixed GB; it does
NOT insert dGB/depsilon_i into Newton.

Checks:
- CUF and KL exact frozen-GB tangent actions vs independent 5-point derivatives;
- full analytic J vs retained scaled-central frozen-GB FD reference;
- CUF K2 exact analytic coloring vs dense analytic assembly on 15x15 TE1;
- nonlinear frozen-GB Newton state/iterations vs FD-reference Newton;
- production Birger inner Newton paths do not call FD Jacobians;
- genuinely nonuniform GB fields are used, so this is not an elastic-G0 alias.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

import V3_SELF_CHECK as v3
import V3B_SELF_CHECK as v3b
import final_v3_analytic as v3an
import cylinder_cuf_stage10d_strong_fdm_full_gl_birger as cuf
import cylinder_kl_stage9d2_strong_fdm_vk_birger as kl
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_te1_stage5a_birger as b5
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as ce
import cylinder_kl_stage9c_strong_fdm_vk_elastic as ke

BASE=Path(__file__).resolve().parent


def _d5(res,x,v,delta=1e-5):
    return (-res(x+2*delta*v)+8*res(x+delta*v)-8*res(x-delta*v)+res(x-2*delta*v))/(12*delta)


def _rel_inf(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.linalg.norm(a-b,np.inf)/max(np.linalg.norm(b,np.inf),1e-300))


def _assert_le(name,val,limit):
    if not np.isfinite(val) or val>limit:
        raise AssertionError(f"{name}: {val:.3e} > {limit:.3e}")


def _gb_field(quad,ns,nx,G0):
    js=np.arange(ns,dtype=float)[:,None]
    ix=np.arange(nx,dtype=float)[None,:]
    rows=[]
    for g in range(len(quad.z)):
        # Deliberately far from constant G0, but strictly positive and smooth.
        rows.append(G0*(0.58+0.045*g+0.07*np.sin(2*np.pi*js/ns)+0.05*np.cos(2*np.pi*ix/nx)))
    G=np.stack(rows,axis=0)
    if np.any(G<=0) or np.allclose(G,G0):
        raise AssertionError('bad V3c GB test field')
    return G


def _action_checks(report):
    p=v3._problem(1.0)
    ops=p['ops']; q=p['q']; U=p['U']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']
    E=p['E0']; nu=p['nu0']; quad=p['quad']; loadc=p['loadc']; loadk=p['loadk']
    G0=E/(2*(1+nu)); K0=E/(3*(1-2*nu))
    GB=_gb_field(quad,ops.grid.ns,ops.grid.nx,G0)
    rng=np.random.default_rng(2026081703)

    dc=rng.normal(size=q.shape); dc/=np.linalg.norm(dc)
    yc=cuf.analytic_tangent_action_frozen_gb(q,dc,ops,basis,bc_c,K0=K0,GB=GB,quadrature=quad)
    resc=lambda x:cuf.full_gl_frozen_gb_fullgrid_residual(x.reshape(q.shape),ops,basis,bc_c,K0=K0,GB=GB,quadrature=quad,generalized_load=loadc)[0]
    refc=_d5(resc,q.ravel(),dc.ravel())
    ec=_rel_inf(yc,refc); _assert_le('V3c CUF frozen-GB tangent action',ec,4e-7)

    dk=rng.normal(size=U.shape); dk/=np.linalg.norm(dk)
    yk=kl.analytic_tangent_action_frozen_gb(U,dk,ops,bc_k,K0=K0,GB=GB,quadrature=quad)
    resk=lambda x:kl.frozen_birger_strong_residual_vk(x.reshape(U.shape),ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk)[0]
    refk=_d5(resk,U.ravel(),dk.ravel())
    ek=_rel_inf(yk,refk); _assert_le('V3c KL frozen-GB tangent action',ek,4e-7)

    report['directional']={'CUF':ec,'KL':ek,'GB_min_over_G0':float(GB.min()/G0),'GB_max_over_G0':float(GB.max()/G0)}


def _matrix_checks(report):
    p=v3._problem(1.0)
    ops=p['ops']; q=p['q']; U=p['U']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']
    E=p['E0']; nu=p['nu0']; quad=p['quad']; loadc=p['loadc']; loadk=p['loadk']
    G0=E/(2*(1+nu)); K0=E/(3*(1-2*nu)); GB=_gb_field(quad,ops.grid.ns,ops.grid.nx,G0)
    eta=1e-5

    t=time.perf_counter(); Jca,_=cuf.analytic_jacobian_frozen_gb(q,ops,basis,bc_c,K0=K0,GB=GB,quadrature=quad,generalized_load=loadc); tca=time.perf_counter()-t
    t=time.perf_counter(); Jcf,_=cuf.finite_difference_jacobian_frozen_gb(q,ops,basis,bc_c,K0=K0,GB=GB,quadrature=quad,generalized_load=loadc,fd_step=eta); tcf=time.perf_counter()-t
    ec=_rel_inf(Jca,Jcf); _assert_le('V3c CUF J analytic/FD',ec,3e-7)

    t=time.perf_counter(); Jka,_=kl.analytic_jacobian_frozen_gb(U,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk); tka=time.perf_counter()-t
    t=time.perf_counter(); Jkf,_=kl.finite_difference_jacobian_frozen_gb(U,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk,fd_step=eta); tkf=time.perf_counter()-t
    ek=_rel_inf(Jka,Jkf); _assert_le('V3c KL J analytic/FD',ek,3e-7)

    report['matrix_vs_fd']={
        'CUF':{'rel_inf':ec,'analytic_s':tca,'fd_s':tcf,'analytic_stats':cuf.analytic_jacobian_frozen_gb.last_stats},
        'KL':{'rel_inf':ek,'analytic_s':tka,'fd_s':tkf,'analytic_stats':kl.analytic_jacobian_frozen_gb.last_stats},
    }


def _cuf_coloring(report):
    p=v3b._make_cuf('TE1',15)
    q=p['q']; ops=p['ops']; basis=p['basis']; bc=p['bc']; E=p['E']; nu=p['nu']; quad=p['quad']
    G0=E/(2*(1+nu)); K0=E/(3*(1-2*nu)); GB=_gb_field(quad,15,15,G0)
    action=cuf._prepare_exact_frozen_gb_tangent_action(q,ops,basis,bc,K0=K0,GB=GB,quadrature=quad)
    t=time.perf_counter(); Jc,stats=v3an.assemble_cuf_colored_exact_tangent(q,ops,action,dependency_layers=2); tc=time.perf_counter()-t
    t=time.perf_counter(); Jd,_=v3an.assemble_dense_exact_tangent(q,action); td=time.perf_counter()-t
    maxabs=float(np.max(np.abs(Jc-Jd)))
    if not np.array_equal(Jc,Jd):
        er=_rel_inf(Jc,Jd); _assert_le('V3c CUF colored/dense exact',er,2e-14)
    else: er=0.0
    report['cuf_15x15_coloring']={'max_abs':maxabs,'rel_inf':er,'colored_s':tc,'dense_s':td,'stats':stats.to_dict()}


def _newton_checks(report):
    p=v3._problem(1.0)
    ops=p['ops']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']; E=p['E0']; nu=p['nu0']; quad=p['quad']
    G0=E/(2*(1+nu)); K0=E/(3*(1-2*nu)); GB=_gb_field(quad,ops.grid.ns,ops.grid.nx,G0)
    loadc=np.zeros_like(p['q']); loadc[2,0]=1.0
    loadk=np.zeros_like(p['U']); loadk[2]=1.0
    q0=np.zeros_like(p['q']); U0=np.zeros_like(p['U'])
    cs=ce.CUFStrongFDMNewtonSettings(1e-8,1e-12,1e-8,8)
    ks=ke.KLStrongFDMNewtonSettings(1e-8,1e-12,1e-8,8)

    orig_fd_c=cuf.finite_difference_jacobian_frozen_gb
    def forbidden_c(*a,**kw): raise AssertionError('CUF Birger production called frozen-GB FD Jacobian')
    cuf.finite_difference_jacobian_frozen_gb=forbidden_c
    try:
        t=time.perf_counter(); ca=cuf.solve_frozen_gb_full_gl_strong_fdm(q0,ops,basis,bc_c,E0=E,K0=K0,GB=GB,quadrature=quad,generalized_load=loadc,newton_settings=cs); tca=time.perf_counter()-t
    finally: cuf.finite_difference_jacobian_frozen_gb=orig_fd_c
    if not ca.converged: raise AssertionError('V3c CUF analytic frozen-GB Newton did not converge')

    orig_fd_k=kl.finite_difference_jacobian_frozen_gb
    def forbidden_k(*a,**kw): raise AssertionError('KL Birger production called frozen-GB FD Jacobian')
    kl.finite_difference_jacobian_frozen_gb=forbidden_k
    try:
        t=time.perf_counter(); ka=kl.solve_frozen_gb_vk_strong_fdm(U0,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk,newton_settings=ks); tka=time.perf_counter()-t
    finally: kl.finite_difference_jacobian_frozen_gb=orig_fd_k
    if not ka.converged: raise AssertionError('V3c KL analytic frozen-GB Newton did not converge')

    orig_aj_c=cuf.analytic_jacobian_frozen_gb
    def fdref_c(q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,base_residual=None):
        return orig_fd_c(q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,generalized_load=generalized_load,fd_step=1e-5,base_residual=base_residual)
    cuf.analytic_jacobian_frozen_gb=fdref_c
    try:
        t=time.perf_counter(); cf=cuf.solve_frozen_gb_full_gl_strong_fdm(q0,ops,basis,bc_c,E0=E,K0=K0,GB=GB,quadrature=quad,generalized_load=loadc,newton_settings=cs); tcf=time.perf_counter()-t
    finally: cuf.analytic_jacobian_frozen_gb=orig_aj_c

    orig_aj_k=kl.analytic_jacobian_frozen_gb
    def fdref_k(U,ops,bc,*,K0,GB,quadrature,generalized_load,base_residual=None):
        return orig_fd_k(U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,generalized_load=generalized_load,fd_step=1e-5,base_residual=base_residual)
    kl.analytic_jacobian_frozen_gb=fdref_k
    try:
        t=time.perf_counter(); kf=kl.solve_frozen_gb_vk_strong_fdm(U0,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk,newton_settings=ks); tkf=time.perf_counter()-t
    finally: kl.analytic_jacobian_frozen_gb=orig_aj_k

    if not cf.converged or not kf.converged: raise AssertionError('V3c FD-reference Newton did not converge')
    ec=float(np.max(np.abs(ca.state-cf.state))/max(np.max(np.abs(cf.state)),1e-300)); _assert_le('V3c CUF Newton state',ec,3e-8)
    ek=float(np.max(np.abs(ka.state-kf.state))/max(np.max(np.abs(kf.state)),1e-300)); _assert_le('V3c KL Newton state',ek,3e-8)
    if ca.iterations!=cf.iterations: raise AssertionError('V3c CUF Newton iteration count changed')
    if ka.iterations!=kf.iterations: raise AssertionError('V3c KL Newton iteration count changed')
    report['newton']={
        'CUF':{'state_rel':ec,'analytic_iterations':ca.iterations,'fd_iterations':cf.iterations,'analytic_residual':ca.residual_inf,'fd_residual':cf.residual_inf,'analytic_s':tca,'fd_s':tcf},
        'KL':{'state_rel':ek,'analytic_iterations':ka.iterations,'fd_iterations':kf.iterations,'analytic_residual':ka.residual_inf,'fd_residual':kf.residual_inf,'analytic_s':tka,'fd_s':tkf},
    }


def _active_outer_cycle_checks(report):
    """Compare a genuinely active Birger outer cycle against FD-inner reference."""
    p=v3._problem(1.0)
    ops=p['ops']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']
    E=p['E0']; nu=p['nu0']; h=p['h']; quad=p['quad']
    G0=E/(2*(1+nu))
    ey=2.0e-4; slope0=3.0*G0; slope1=0.25*slope0
    def phi(eps):
        a=np.asarray(eps,float)
        return np.where(a<=ey,slope0*a,slope0*ey+slope1*(a-ey))
    settings=b5.BirgerSettings(
        epsilon_tol=1e-12,omega=1.0,G_ref=G0,sigma_ref=slope0*ey,
        tol_G=2e-6,tol_sigma=2e-6,tol_u=2e-6,max_outer_iterations=30,
        epsilon_i_limit=.2,delta_J_limit=1.0,initial_slope_rtol=.05,
    )
    GB0=np.full((len(quad.z),ops.grid.ns,ops.grid.nx),G0)
    loadc=np.zeros_like(p['q']); loadc[2,0]=1.0
    loadk=np.zeros_like(p['U']); loadk[2]=1.0
    cs=ce.CUFStrongFDMNewtonSettings(1e-8,1e-12,1e-5,10)
    ks=ke.KLStrongFDMNewtonSettings(1e-8,1e-12,1e-5,10)

    def rel(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        return float(np.max(np.abs(a-b))/max(np.max(np.abs(b)),1e-300))

    cargs=dict(
        initial_q=np.zeros_like(p['q']),initial_GB=GB0,ops=ops,basis=basis,bc=bc_c,
        E0=E,nu0=nu,quadrature=quad,generalized_load=loadc,newton_settings=cs,
        material_diagram=phi,birger_settings=settings,
    )
    t=time.perf_counter(); ca,_=cuf.solve_full_gl_birger_strong_fdm(**cargs); tca=time.perf_counter()-t
    orig_c=cuf.analytic_jacobian_frozen_gb
    def fd_c(q,ops,basis,bc,*,K0,GB,quadrature,generalized_load,base_residual=None):
        return cuf.finite_difference_jacobian_frozen_gb(
            q,ops,basis,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,fd_step=1e-5,base_residual=base_residual
        )
    cuf.analytic_jacobian_frozen_gb=fd_c
    try:
        t=time.perf_counter(); cf,_=cuf.solve_full_gl_birger_strong_fdm(**cargs); tcf=time.perf_counter()-t
    finally:
        cuf.analytic_jacobian_frozen_gb=orig_c
    if not ca.converged or not cf.converged: raise AssertionError('V3c active CUF Birger cycle did not converge')
    if ca.outer_iterations!=cf.outer_iterations: raise AssertionError('V3c active CUF outer iteration count changed')
    ecs=rel(ca.state,cf.state); ecg=rel(ca.GB,cf.GB); ece=rel(ca.material_state.epsilon_i,cf.material_state.epsilon_i)
    _assert_le('V3c active CUF state',ecs,3e-8); _assert_le('V3c active CUF GB',ecg,3e-8); _assert_le('V3c active CUF eps',ece,3e-8)
    if float(ca.GB.min()/G0)>0.9: raise AssertionError('V3c active CUF did not exercise Birger modulus reduction')

    kargs=dict(
        initial_U=np.zeros_like(p['U']),initial_GB=GB0,ops=ops,bc=bc_k,
        E0=E,nu0=nu,h=h,quadrature=quad,generalized_load=loadk,newton_settings=ks,
        material_diagram=phi,birger_settings=settings,
    )
    t=time.perf_counter(); ka,_=kl.solve_vk_birger_strong_fdm(**kargs); tka=time.perf_counter()-t
    orig_k=kl.analytic_jacobian_frozen_gb
    def fd_k(U,ops,bc,*,K0,GB,quadrature,generalized_load,base_residual=None):
        return kl.finite_difference_jacobian_frozen_gb(
            U,ops,bc,K0=K0,GB=GB,quadrature=quadrature,
            generalized_load=generalized_load,fd_step=1e-5,base_residual=base_residual
        )
    kl.analytic_jacobian_frozen_gb=fd_k
    try:
        t=time.perf_counter(); kf,_=kl.solve_vk_birger_strong_fdm(**kargs); tkf=time.perf_counter()-t
    finally:
        kl.analytic_jacobian_frozen_gb=orig_k
    if not ka.converged or not kf.converged: raise AssertionError('V3c active KL Birger cycle did not converge')
    if ka.outer_iterations!=kf.outer_iterations: raise AssertionError('V3c active KL outer iteration count changed')
    eks=rel(ka.state,kf.state); ekg=rel(ka.GB,kf.GB); eke=rel(ka.material_state.epsilon_i,kf.material_state.epsilon_i)
    _assert_le('V3c active KL state',eks,3e-8); _assert_le('V3c active KL GB',ekg,3e-8); _assert_le('V3c active KL eps',eke,3e-8)
    if float(ka.GB.min()/G0)>0.9: raise AssertionError('V3c active KL did not exercise Birger modulus reduction')

    report['active_outer_cycle']={
        'CUF':{'outer_iterations':ca.outer_iterations,'state_rel':ecs,'GB_rel':ecg,'epsilon_rel':ece,'GB_min_over_G0':float(ca.GB.min()/G0),'analytic_s':tca,'fd_s':tcf},
        'KL':{'outer_iterations':ka.outer_iterations,'state_rel':eks,'GB_rel':ekg,'epsilon_rel':eke,'GB_min_over_G0':float(ka.GB.min()/G0),'analytic_s':tka,'fd_s':tkf},
    }


def run_checks(verbose=True):
    report={'status':'RUNNING','principle':'INNER_DERIVATIVE_AT_FIXED_GB'}
    _action_checks(report)
    _matrix_checks(report)
    _cuf_coloring(report)
    _newton_checks(report)
    _active_outer_cycle_checks(report)
    report['status']='PASS'
    if verbose:
        print('V3C_SELF_CHECK: PASS')
        d=report['directional']; print(f"  frozen-GB Jv: CUF={d['CUF']:.3e}, KL={d['KL']:.3e}, GB/G0={d['GB_min_over_G0']:.3f}..{d['GB_max_over_G0']:.3f}")
        for name,row in report['matrix_vs_fd'].items():
            print(f"  {name} analytic/FD J rel={row['rel_inf']:.3e}, analytic={row['analytic_s']:.4f}s, FD={row['fd_s']:.4f}s")
        row=report['cuf_15x15_coloring']; st=row['stats']; print(f"  CUF 15x15 frozen-GB coloring: max_abs={row['max_abs']:.3e}, actions={st['tangent_action_evaluations']}/{st['dense_reference_action_evaluations']}")
        for name,row in report['newton'].items():
            print(f"  {name} frozen-GB Newton: state_rel={row['state_rel']:.3e}, iterations={row['analytic_iterations']}, analytic={row['analytic_s']:.4f}s, FD={row['fd_s']:.4f}s")
        for name,row in report['active_outer_cycle'].items():
            print(f"  {name} active Birger outer: outer={row['outer_iterations']}, state_rel={row['state_rel']:.3e}, GB/G0 min={row['GB_min_over_G0']:.3f}, analytic={row['analytic_s']:.3f}s, FD={row['fd_s']:.3f}s")
    return report


if __name__=='__main__':
    rep=run_checks(True)
    (BASE/'V3C_VERIFICATION_RESULTS.json').write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8')
    print('V3C_REPORT_JSON='+json.dumps(rep,sort_keys=True))
