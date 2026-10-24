#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3b regression: exact analytic elastic tangents for nonlinear FDM.

Checks:
- CUF Green-Lagrange exact dE/dq, dS/dq, dP/dq tangent action;
- KL von-Karman exact membrane/geometric/bending tangent action;
- exact tangent actions against an independent 5-point directional derivative;
- full analytic J against the retained V3 scaled-central FD reference;
- CUF K2-colored analytic assembly against one-action-per-column analytic reference;
- TE1/TE2/TE3 CUF tangent action coverage;
- KL ANALYTIC and QUADRATURE resultant paths;
- nonlinear Newton solutions with analytic J against FD-reference Newton solutions;
- production elastic Newton path does not call the FD Jacobian.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

import V3_SELF_CHECK as v3
import final_v3_analytic as v3an
import cylinder_te1_mvi_core as core
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage2b as s2
import final_te_basis_provider as teb
import cylinder_cuf_stage7c_general_boundary as cbc
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as cuf
import cylinder_kl_stage9c_strong_fdm_vk_elastic as kl

BASE=Path(__file__).resolve().parent


def _d5(res,x,v,delta=1e-5):
    return (-res(x+2*delta*v)+8*res(x+delta*v)-8*res(x-delta*v)+res(x-2*delta*v))/(12*delta)


def _rel_inf(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.linalg.norm(a-b,np.inf)/max(np.linalg.norm(b,np.inf),1e-300))


def _assert_le(name,val,limit):
    if not np.isfinite(val) or val>limit:
        raise AssertionError(f"{name}: {val:.3e} > {limit:.3e}")


def _make_cuf(model='TE1',n=7):
    L=100.0; R=50.0; h=1.0; E=70000.0; nu=0.3
    grid=core.CylinderGrid(L,R,n,n)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    quad=s2.gauss_thickness_quadrature(h,4)
    basis=teb.get_basis(model)
    bc=cbc.GeneralCylinderEndBC.clamped_clamped(basis.m)
    js,ix=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    ph=2*np.pi*js/n+.23*ix
    q=np.zeros((3,basis.m,n,n),float)
    q[0,0]=.012*np.sin(ph); q[1,0]=.009*np.cos(.8*ph); q[2,0]=.09*np.sin(.55*ph+.1)
    for t in range(1,basis.m):
        amp=1.5e-4/(10.0**(t-1))
        q[0,t]=amp*np.cos((.7+.05*t)*ph)
        q[1,t]=.8*amp*np.sin((.6+.04*t)*ph)
        q[2,t]=1.2*amp*np.cos((.5+.03*t)*ph)
    load=np.zeros_like(q)
    return locals()


def _action_checks(report):
    p=v3._problem(1.0)
    ops=p['ops']; q=p['q']; U=p['U']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']
    E=p['E0']; nu=p['nu0']; h=p['h']; quad=p['quad']; loadc=p['loadc']; loadk=p['loadk']
    rng=np.random.default_rng(20260817)

    dc=rng.normal(size=q.shape); dc/=np.linalg.norm(dc)
    yc=cuf.analytic_tangent_action(q,dc,ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad)
    resc=lambda x:cuf.full_gl_elastic_fullgrid_residual(x.reshape(q.shape),ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad,generalized_load=loadc)[0]
    refc=_d5(resc,q.ravel(),dc.ravel())
    ec=_rel_inf(yc,refc); _assert_le('V3b CUF tangent action',ec,3e-7)

    dk=rng.normal(size=U.shape); dk/=np.linalg.norm(dk)
    yk=kl.analytic_tangent_action(U,dk,ops,bc_k,E=E,nu=nu,h=h)
    resk=lambda x:kl.strong_collocation_residual_vk(x.reshape(U.shape),ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk)[0]
    refk=_d5(resk,U.ravel(),dk.ravel())
    ek=_rel_inf(yk,refk); _assert_le('V3b KL tangent action',ek,3e-7)

    # Same exact KL tangent must cover the quadrature-resultant residual path.
    ykq=kl.analytic_tangent_action(U,dk,ops,bc_k,E=E,nu=nu,h=h,resultants_method='QUADRATURE',quadrature=quad)
    reskq=lambda x:kl.strong_collocation_residual_vk(x.reshape(U.shape),ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk,resultants_method='QUADRATURE',quadrature=quad)[0]
    refkq=_d5(reskq,U.ravel(),dk.ravel())
    ekq=_rel_inf(ykq,refkq); _assert_le('V3b KL quadrature tangent action',ekq,3e-7)

    report['directional']={'CUF_TE1':ec,'KL_ANALYTIC':ek,'KL_QUADRATURE':ekq}


def _general_cuf_basis_checks(report):
    rng=np.random.default_rng(77)
    rows={}
    for model in ('TE1','TE2','TE3'):
        p=_make_cuf(model,7); q=p['q']; ops=p['ops']; basis=p['basis']; bc=p['bc']; E=p['E']; nu=p['nu']; quad=p['quad']; load=p['load']
        d=rng.normal(size=q.shape); d/=np.linalg.norm(d)
        got=cuf.analytic_tangent_action(q,d,ops,basis,bc,E0=E,nu0=nu,quadrature=quad)
        def res(x):
            return cuf.full_gl_elastic_fullgrid_residual(x.reshape(q.shape),ops,basis,bc,E0=E,nu0=nu,quadrature=quad,generalized_load=load)[0]
        ref=_d5(res,q.ravel(),d.ravel())
        er=_rel_inf(got,ref); _assert_le(f'V3b {model}',er,4e-7)
        rows[model]=er
    report['cuf_basis_directional']=rows


def _matrix_checks(report):
    p=v3._problem(1.0)
    ops=p['ops']; q=p['q']; U=p['U']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']
    E=p['E0']; nu=p['nu0']; h=p['h']; quad=p['quad']; loadc=p['loadc']; loadk=p['loadk']
    eta=1e-5

    t=time.perf_counter(); Jca,_=cuf.analytic_jacobian(q,ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad,generalized_load=loadc); tca=time.perf_counter()-t
    t=time.perf_counter(); Jcf,_=cuf.finite_difference_jacobian(q,ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad,generalized_load=loadc,fd_step=eta); tcf=time.perf_counter()-t
    erc=_rel_inf(Jca,Jcf); _assert_le('V3b CUF J analytic/FD',erc,2e-7)

    t=time.perf_counter(); Jka,_=kl.analytic_jacobian(U,ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk); tka=time.perf_counter()-t
    t=time.perf_counter(); Jkf,_=kl.finite_difference_jacobian(U,ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk,fd_step=eta); tkf=time.perf_counter()-t
    erk=_rel_inf(Jka,Jkf); _assert_le('V3b KL J analytic/FD',erk,2e-7)

    report['matrix_vs_fd']={
        'CUF':{'rel_inf':erc,'analytic_s':tca,'fd_s':tcf,'analytic_stats':cuf.analytic_jacobian.last_stats},
        'KL':{'rel_inf':erk,'analytic_s':tka,'fd_s':tkf,'analytic_stats':kl.analytic_jacobian.last_stats},
    }


def _cuf_coloring_exactness(report):
    p=_make_cuf('TE1',15); q=p['q']; ops=p['ops']; basis=p['basis']; bc=p['bc']; E=p['E']; nu=p['nu']; quad=p['quad']
    action=cuf._prepare_exact_elastic_tangent_action(q,ops,basis,bc,E0=E,nu0=nu,quadrature=quad)
    t=time.perf_counter(); Jc,stats=v3an.assemble_cuf_colored_exact_tangent(q,ops,action,dependency_layers=2); tc=time.perf_counter()-t
    t=time.perf_counter(); Jd,_=v3an.assemble_dense_exact_tangent(q,action); td=time.perf_counter()-t
    if not np.array_equal(Jc,Jd):
        er=_rel_inf(Jc,Jd); _assert_le('V3b CUF colored/dense analytic',er,2e-14)
    else:
        er=0.0
    report['cuf_15x15_coloring']={
        'max_abs':float(np.max(np.abs(Jc-Jd))), 'rel_inf':er,
        'colored_s':tc,'dense_s':td,'stats':stats.to_dict(),
    }


def _newton_comparison(report):
    p=v3._problem(1.0)
    ops=p['ops']; basis=p['basis']; bc_c=p['bc_c']; bc_k=p['bc_k']; E=p['E0']; nu=p['nu0']; h=p['h']; quad=p['quad']
    loadc=np.zeros_like(p['q']); loadc[2,0]=1.0
    loadk=np.zeros_like(p['U']); loadk[2]=1.0
    q0=np.zeros_like(p['q']); U0=np.zeros_like(p['U'])
    cs=cuf.CUFStrongFDMNewtonSettings(1e-8,1e-12,1e-8,8)
    ks=kl.KLStrongFDMNewtonSettings(1e-8,1e-12,1e-8,8)

    # Production guard: if elastic Newton touches FD, fail immediately.
    orig_fd_c=cuf.finite_difference_jacobian
    def forbidden_c(*a,**kw): raise AssertionError('CUF elastic production called FD Jacobian')
    cuf.finite_difference_jacobian=forbidden_c
    try:
        t=time.perf_counter(); ca=cuf.solve_full_gl_elastic_strong_fdm(q0,ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad,generalized_load=loadc,settings=cs); tca=time.perf_counter()-t
    finally:
        cuf.finite_difference_jacobian=orig_fd_c
    if not ca.converged: raise AssertionError('V3b CUF analytic Newton did not converge')

    orig_fd_k=kl.finite_difference_jacobian
    def forbidden_k(*a,**kw): raise AssertionError('KL elastic production called FD Jacobian')
    kl.finite_difference_jacobian=forbidden_k
    try:
        t=time.perf_counter(); ka=kl.solve_vk_elastic_strong_fdm(U0,ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk,settings=ks); tka=time.perf_counter()-t
    finally:
        kl.finite_difference_jacobian=orig_fd_k
    if not ka.converged: raise AssertionError('V3b KL analytic Newton did not converge')

    # Independent FD-reference Newton by temporarily replacing only the analytic-J hook.
    orig_aj_c=cuf.analytic_jacobian
    def fdref_c(q,ops,basis,bc,*,E0,nu0,quadrature,generalized_load,base_residual=None):
        return orig_fd_c(q,ops,basis,bc,E0=E0,nu0=nu0,quadrature=quadrature,generalized_load=generalized_load,fd_step=1e-5,base_residual=base_residual)
    cuf.analytic_jacobian=fdref_c
    try:
        t=time.perf_counter(); cf=cuf.solve_full_gl_elastic_strong_fdm(q0,ops,basis,bc_c,E0=E,nu0=nu,quadrature=quad,generalized_load=loadc,settings=cs); tcf=time.perf_counter()-t
    finally:
        cuf.analytic_jacobian=orig_aj_c

    orig_aj_k=kl.analytic_jacobian
    def fdref_k(U,ops,bc,*,E,nu,h,generalized_load,resultants_method='ANALYTIC',quadrature=None,base_residual=None):
        return orig_fd_k(U,ops,bc,E=E,nu=nu,h=h,generalized_load=generalized_load,fd_step=1e-5,resultants_method=resultants_method,quadrature=quadrature,base_residual=base_residual)
    kl.analytic_jacobian=fdref_k
    try:
        t=time.perf_counter(); kf=kl.solve_vk_elastic_strong_fdm(U0,ops,bc_k,E=E,nu=nu,h=h,generalized_load=loadk,settings=ks); tkf=time.perf_counter()-t
    finally:
        kl.analytic_jacobian=orig_aj_k

    if not cf.converged or not kf.converged: raise AssertionError('V3b FD-reference Newton did not converge')
    ec=float(np.max(np.abs(ca.q-cf.q))/max(np.max(np.abs(cf.q)),1e-300)); _assert_le('V3b CUF Newton state',ec,2e-8)
    ek=float(np.max(np.abs(ka.U-kf.U))/max(np.max(np.abs(kf.U)),1e-300)); _assert_le('V3b KL Newton state',ek,2e-8)
    if ca.iterations!=cf.iterations: raise AssertionError('V3b CUF Newton iteration count changed')
    if ka.iterations!=kf.iterations: raise AssertionError('V3b KL Newton iteration count changed')
    report['newton']={
        'CUF':{'state_rel':ec,'analytic_iterations':ca.iterations,'fd_iterations':cf.iterations,'analytic_residual':ca.residual_inf,'fd_residual':cf.residual_inf,'analytic_s':tca,'fd_s':tcf},
        'KL':{'state_rel':ek,'analytic_iterations':ka.iterations,'fd_iterations':kf.iterations,'analytic_residual':ka.residual_inf,'fd_residual':kf.residual_inf,'analytic_s':tka,'fd_s':tkf},
    }


def run_checks(verbose=True):
    report={'status':'RUNNING'}
    _action_checks(report)
    _general_cuf_basis_checks(report)
    _matrix_checks(report)
    _cuf_coloring_exactness(report)
    _newton_comparison(report)
    report['status']='PASS'
    if verbose:
        print('V3B_SELF_CHECK: PASS')
        print('  exact tangent Jv:', ', '.join(f"{k}={v:.3e}" for k,v in report['directional'].items()))
        print('  CUF basis Jv:', ', '.join(f"{k}={v:.3e}" for k,v in report['cuf_basis_directional'].items()))
        for name,row in report['matrix_vs_fd'].items():
            print(f"  {name} analytic/FD J rel={row['rel_inf']:.3e}, analytic={row['analytic_s']:.4f}s, FD={row['fd_s']:.4f}s")
        row=report['cuf_15x15_coloring']; st=row['stats']
        print(f"  CUF 15x15 exact coloring: max_abs={row['max_abs']:.3e}, actions={st['tangent_action_evaluations']}/{st['dense_reference_action_evaluations']}")
        for name,row in report['newton'].items():
            print(f"  {name} Newton: state_rel={row['state_rel']:.3e}, iterations={row['analytic_iterations']}, analytic={row['analytic_s']:.4f}s, FD={row['fd_s']:.4f}s")
    return report


if __name__=='__main__':
    rep=run_checks(True)
    (BASE/'V3B_VERIFICATION_RESULTS.json').write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8')
    print('V3B_REPORT_JSON='+json.dumps(rep,sort_keys=True))
