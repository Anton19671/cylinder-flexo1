#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 regression: scaled central finite-difference Jacobians.

Checks the minimum V3 repair from the audit without introducing an analytic
material/geometric tangent:
- h_j = eta*max(|x_j|, characteristic_scale_j),
- central differences,
- CUF TE1 scales h and h/L,
- K2 colored central CUF equals one-column central reference,
- Jv agrees with an independent 5-point directional derivative,
- the same relative eta works in equivalent mm/MPa and SI descriptions,
- CUF/KL x Elastic/frozen-Birger nonlinear Jacobians are covered.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import final_v3_fd as v3fd
import cylinder_te1_mvi_core as core
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage2b as s2
import final_te_basis_provider as teb
import cylinder_cuf_stage7c_general_boundary as cbc
import cylinder_kl_stage8b1_boundary as kbc
import cylinder_cuf_stage7f_general_linear as gl
import cylinder_kl_stage8c1_birger_material as bm
import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as c10b
import cylinder_cuf_stage10d_strong_fdm_full_gl_birger as c10d
import cylinder_kl_stage9c_strong_fdm_vk_elastic as k9c
import cylinder_kl_stage9d2_strong_fdm_vk_birger as k9d2

BASE=Path(__file__).resolve().parent


def _problem(length_scale=1.0):
    # length_scale=1 -> mm/MPa; 1e-3 -> m/Pa equivalent problem.
    sc=float(length_scale)
    L=100.0*sc; R=50.0*sc; h=1.0*sc
    E0=70000.0 if sc==1.0 else 70.0e9
    nu0=0.3
    grid=core.CylinderGrid(L,R,7,7)
    ops=c3.CylinderSpatialOperators.build(grid,7)
    quad=s2.gauss_thickness_quadrature(h,3)
    basis=teb.get_basis('TE1')
    bc_c=cbc.GeneralCylinderEndBC.clamped_clamped(basis.m)
    bc_k=kbc.KLCylinderEndBC.clamped_clamped()
    js,ix=np.meshgrid(np.arange(7),np.arange(7),indexing='ij')
    ph=2*np.pi*js/7 + .4*ix
    q=np.zeros((3,2,7,7))
    q[0,0]=.02*sc*np.sin(ph); q[1,0]=.015*sc*np.cos(.7*ph)
    q[2,0]=.15*sc*np.sin(.5*ph+.2)
    # tau=1 is dimensionless in TE1 and is unchanged by length-unit conversion.
    q[0,1]=2e-4*np.cos(ph); q[1,1]=1.5e-4*np.sin(.8*ph)
    q[2,1]=3e-4*np.cos(.6*ph)
    U=np.zeros((3,7,7))
    U[0]=.02*sc*np.sin(ph); U[1]=.015*sc*np.cos(.7*ph)
    U[2]=.15*sc*np.sin(.5*ph+.2)
    loadc=np.zeros_like(q); loadk=np.zeros_like(U)
    ec=gl.lk.ElasticConstants(E0,nu0)
    G0=bm.shear_modulus_from_E_nu(E0,nu0)
    K0=bm.bulk_modulus_from_E_nu(E0,nu0)
    z=quad.z[:,None,None]
    GB=G0*(.78+.18*(1+np.sin(.6*z/max(sc,1e-300)+.3*js[None,:,:]+.2*ix[None,:,:]))/2)
    return locals()


def _d5(res,x,v,delta=1e-3):
    return (-res(x+2*delta*v)+8*res(x+delta*v)-8*res(x-delta*v)+res(x-2*delta*v))/(12*delta)


def _jv_errors(J,res,x,scales,seed):
    rng=np.random.default_rng(seed)
    errs=[]
    for _ in range(3):
        z=rng.normal(size=x.size); z/=np.linalg.norm(z)
        v=scales*z
        ref=_d5(res,x,v)
        got=J@v
        errs.append(float(np.linalg.norm(got-ref,np.inf)/max(np.linalg.norm(ref,np.inf),1e-300)))
    return errs


def _assert_small(name,value,limit):
    if not np.isfinite(value) or value>limit:
        raise AssertionError(f"{name}: {value:.3e} > {limit:.3e}")


def run_checks(verbose=True):
    p=_problem(1.0)
    ops=p['ops']; quad=p['quad']; basis=p['basis']; q=p['q']; U=p['U']
    E0=p['E0']; nu0=p['nu0']; h=p['h']; bc_c=p['bc_c']; bc_k=p['bc_k']
    loadc=p['loadc']; loadk=p['loadk']; ec=p['ec']; K0=p['K0']; GB=p['GB']

    cscale=v3fd.cuf_characteristic_scale(q,ops,quad)
    # Exact audit TE1 floor: tau0=h, tau1=h/L.
    if not np.allclose(cscale[:,0],h,rtol=0.0,atol=1e-15*max(1.0,abs(h))):
        raise AssertionError("V3 CUF tau0 characteristic scale is not h")
    expected_tau1=h/ops.grid.length_x
    if not np.allclose(cscale[:,1],expected_tau1,rtol=0.0,atol=1e-15*max(1.0,abs(expected_tau1))):
        raise AssertionError("V3 CUF tau1 characteristic scale is not h/L")
    kscale=v3fd.kl_characteristic_scale(U,h)
    if not np.allclose(kscale,h,rtol=0.0,atol=1e-15*max(1.0,abs(h))):
        raise AssertionError("V3 KL characteristic scale is not h")

    eta=1e-5
    report={"relative_step_for_accuracy_check":eta,"cases":{}}

    # CUF elastic production K2 vs one-column central reference.
    rce=c10b.full_gl_elastic_fullgrid_residual(q,ops,basis,bc_c,E0=E0,nu0=nu0,quadrature=quad,generalized_load=loadc)[0]
    Jce,_=c10b.finite_difference_jacobian(q,ops,basis,bc_c,E0=E0,nu0=nu0,quadrature=quad,generalized_load=loadc,fd_step=eta,base_residual=rce)
    Jce_ref,_=c10b._finite_difference_jacobian_dense_reference(q,ops,basis,bc_c,E0=E0,nu0=nu0,quadrature=quad,generalized_load=loadc,fd_step=eta,base_residual=rce)
    de=float(np.max(np.abs(Jce-Jce_ref)))
    if not np.array_equal(Jce,Jce_ref):
        # Exact equality is normally obtained; allow only roundoff-level fallback.
        rel=de/max(float(np.max(np.abs(Jce_ref))),1.0)
        _assert_small("V3 CUF elastic dense/K2",rel,5e-13)
    res=lambda x:c10b.full_gl_elastic_fullgrid_residual(x.reshape(q.shape),ops,basis,bc_c,E0=E0,nu0=nu0,quadrature=quad,generalized_load=loadc)[0]
    errs=_jv_errors(Jce,res,q.ravel(),cscale.ravel(),101)
    _assert_small("V3 CUF elastic Jv",max(errs),2e-7)
    report['cases']['CUF_NONLINEAR_ELASTIC']={"jv_max_rel":max(errs),"dense_k2_max_abs":de,"k2_stats":c10b.finite_difference_jacobian.last_k2_stats}

    # CUF frozen Birger.
    rcb=c10d.full_gl_frozen_gb_fullgrid_residual(q,ops,basis,bc_c,K0=ec.K0,GB=GB,quadrature=quad,generalized_load=loadc)[0]
    Jcb,_=c10d.finite_difference_jacobian_frozen_gb(q,ops,basis,bc_c,K0=ec.K0,GB=GB,quadrature=quad,generalized_load=loadc,fd_step=eta,base_residual=rcb)
    Jcb_ref,_=c10d._finite_difference_jacobian_frozen_gb_dense_reference(q,ops,basis,bc_c,K0=ec.K0,GB=GB,quadrature=quad,generalized_load=loadc,fd_step=eta,base_residual=rcb)
    db=float(np.max(np.abs(Jcb-Jcb_ref)))
    if not np.array_equal(Jcb,Jcb_ref):
        rel=db/max(float(np.max(np.abs(Jcb_ref))),1.0)
        _assert_small("V3 CUF Birger dense/K2",rel,5e-13)
    res=lambda x:c10d.full_gl_frozen_gb_fullgrid_residual(x.reshape(q.shape),ops,basis,bc_c,K0=ec.K0,GB=GB,quadrature=quad,generalized_load=loadc)[0]
    errs=_jv_errors(Jcb,res,q.ravel(),cscale.ravel(),102)
    _assert_small("V3 CUF Birger Jv",max(errs),2e-7)
    report['cases']['CUF_NONLINEAR_BIRGER_FROZEN_GB']={"jv_max_rel":max(errs),"dense_k2_max_abs":db,"k2_stats":c10d.finite_difference_jacobian_frozen_gb.last_k2_stats}

    # KL elastic.
    rke=k9c.strong_collocation_residual_vk(U,ops,bc_k,E=E0,nu=nu0,h=h,generalized_load=loadk)[0]
    Jke,_=k9c.finite_difference_jacobian(U,ops,bc_k,E=E0,nu=nu0,h=h,generalized_load=loadk,fd_step=eta,base_residual=rke)
    res=lambda x:k9c.strong_collocation_residual_vk(x.reshape(U.shape),ops,bc_k,E=E0,nu=nu0,h=h,generalized_load=loadk)[0]
    errs=_jv_errors(Jke,res,U.ravel(),kscale.ravel(),103)
    _assert_small("V3 KL elastic Jv",max(errs),2e-7)
    report['cases']['KL_NONLINEAR_ELASTIC']={"jv_max_rel":max(errs)}

    # KL frozen Birger.
    rkb=k9d2.frozen_birger_strong_residual_vk(U,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk)[0]
    Jkb,_=k9d2.finite_difference_jacobian_frozen_gb(U,ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk,fd_step=eta,base_residual=rkb)
    res=lambda x:k9d2.frozen_birger_strong_residual_vk(x.reshape(U.shape),ops,bc_k,K0=K0,GB=GB,quadrature=quad,generalized_load=loadk)[0]
    errs=_jv_errors(Jkb,res,U.ravel(),kscale.ravel(),104)
    _assert_small("V3 KL Birger Jv",max(errs),2e-7)
    report['cases']['KL_NONLINEAR_BIRGER_FROZEN_GB']={"jv_max_rel":max(errs)}

    # Unit robustness with the same dimensionless relative step.  Use eta=1e-8
    # because it was the legacy absolute value in software regression cases.
    unit_eta=1e-8; unit_err={}
    for tag,sc in [('N_MM_MPA',1.0),('SI',1e-3)]:
        u=_problem(sc); q2=u['q']; ops2=u['ops']; basis2=u['basis']; bc2=u['bc_c']; quad2=u['quad']
        def res2(x):
            return c10b.full_gl_elastic_fullgrid_residual(x.reshape(q2.shape),ops2,basis2,bc2,E0=u['E0'],nu0=u['nu0'],quadrature=quad2,generalized_load=u['loadc'])[0]
        R2=res2(q2.ravel())
        J2,_=c10b.finite_difference_jacobian(q2,ops2,basis2,bc2,E0=u['E0'],nu0=u['nu0'],quadrature=quad2,generalized_load=u['loadc'],fd_step=unit_eta,base_residual=R2)
        sc2=v3fd.cuf_characteristic_scale(q2,ops2,quad2).ravel()
        errs2=_jv_errors(J2,res2,q2.ravel(),sc2,201)
        unit_err[tag]=max(errs2)
        _assert_small(f"V3 unit robustness {tag}",unit_err[tag],3e-6)
    ratio=max(unit_err.values())/max(min(unit_err.values()),1e-300)
    if ratio>10.0:
        raise AssertionError(f"V3 unit-system Jv errors differ excessively: ratio={ratio:.3g}")
    report['unit_robustness']={"relative_step":unit_eta,"jv_max_rel":unit_err,"error_ratio":ratio}
    report['characteristic_scale']={"CUF_tau0":h,"CUF_tau1":expected_tau1,"KL":h}
    report['status']='PASS'

    if verbose:
        print('V3_SELF_CHECK: PASS')
        print(f"  CUF TE1 characteristic scales: tau0=h={h:g}, tau1=h/L={expected_tau1:g}")
        for name,row in report['cases'].items():
            print(f"  {name}: Jv max rel={row['jv_max_rel']:.3e}")
        print(f"  unit robustness eta={unit_eta:g}: mm/MPa={unit_err['N_MM_MPA']:.3e}, SI={unit_err['SI']:.3e}, ratio={ratio:.3f}")
    return report


if __name__=='__main__':
    rep=run_checks(True)
    (BASE/'V3_VERIFICATION_RESULTS.json').write_text(json.dumps(rep,indent=2,sort_keys=True),encoding='utf-8')
    print('V3_REPORT_JSON='+json.dumps(rep,sort_keys=True))
