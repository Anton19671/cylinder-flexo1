#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression/self-check for KL + full Green-Lagrange + Hooke Stage9E."""
from __future__ import annotations
import numpy as np

import V3_SELF_CHECK as v3
import cylinder_kl_stage9e_strong_fdm_gl_elastic as gl
import cylinder_final_stage11a_dispatcher as dispatch


def _rel_inf(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.linalg.norm(a-b,np.inf)/max(np.linalg.norm(b,np.inf),1e-300))


def run_checks(verbose=True):
    p=v3._problem(1.0)
    ops,U,bc=p['ops'],p['U'],p['bc_k']
    E,nu,h=p['E0'],p['nu0'],p['h']
    load=p['loadk']
    rng=np.random.default_rng(20260826)
    d=rng.normal(size=U.shape); d/=np.linalg.norm(d)

    # Independent 5-point directional derivative.
    y=gl.analytic_tangent_action(U,d,ops,bc,E=E,nu=nu,h=h)
    def res(x):
        return gl.strong_collocation_residual_gl(
            x.reshape(U.shape),ops,bc,E=E,nu=nu,h=h,
            generalized_load=load
        )[0]
    delta=1e-5; x=U.ravel(); v=d.ravel()
    ref=(-res(x+2*delta*v)+8*res(x+delta*v)-8*res(x-delta*v)+res(x-2*delta*v))/(12*delta)
    directional=_rel_inf(y,ref)
    if directional>3e-7:
        raise AssertionError(f'Stage9E directional tangent error {directional:.3e}')

    Ja,_=gl.analytic_jacobian(U,ops,bc,E=E,nu=nu,h=h,generalized_load=load)
    Jf,_=gl.finite_difference_jacobian(U,ops,bc,E=E,nu=nu,h=h,
                                       generalized_load=load,fd_step=1e-5)
    matrix=_rel_inf(Ja,Jf)
    if matrix>2e-7:
        raise AssertionError(f'Stage9E analytic/FD Jacobian error {matrix:.3e}')

    # Real Newton solve from zero.
    load2=np.zeros_like(U); load2[2]=1.0
    settings=gl.KLStrongFDMNewtonSettings(1e-8,1e-12,1e-8,10)
    sol=gl.solve_gl_elastic_strong_fdm(
        np.zeros_like(U),ops,bc,E=E,nu=nu,h=h,
        generalized_load=load2,settings=settings
    )
    if not sol.converged:
        raise AssertionError(sol.status)

    # Routing: all KL geometry/material combinations, with explicit GL+Birger FDM ban.
    expected={
        ('LINEAR','HOOKE','FDM'):'KL_LINEAR_ELASTIC',
        ('LINEAR','BIRGER','FDM'):'KL_LINEAR_BIRGER',
        ('VON_KARMAN','HOOKE','FDM'):'KL_VK_ELASTIC',
        ('VON_KARMAN','BIRGER','FDM'):'KL_VK_BIRGER',
        ('GREEN_LAGRANGE','HOOKE','FDM'):'KL_GL_ELASTIC',
        ('GREEN_LAGRANGE','HOOKE','MVI'):'KL_GL_ELASTIC',
        ('GREEN_LAGRANGE','BIRGER','MVI'):'KL_GL_BIRGER',
    }
    for (geometry,material,method),want in expected.items():
        got=dispatch._kl_child_mode({'geometry_model':geometry,'material_model':material,'method':method})
        if got!=want:
            raise AssertionError((geometry,material,method,got,want))
    try:
        dispatch._kl_child_mode({'geometry_model':'GREEN_LAGRANGE','material_model':'BIRGER','method':'FDM'})
    except dispatch.FinalCaseError as exc:
        if 'refusing fallback to von Karman Birger' not in str(exc):
            raise
    else:
        raise AssertionError('KL GL+Birger FDM was not rejected')

    report={
        'status':'PASS',
        'directional_rel_inf':directional,
        'matrix_rel_inf':matrix,
        'newton_iterations':int(sol.iterations),
        'newton_residual_relative':float(sol.residual_relative),
        'routing':'PASS',
        'gl_birger_fdm_rejected':True,
    }
    if verbose:
        print('KL_GL_STAGE9E_SELF_CHECK: PASS')
        print(f'  directional tangent rel = {directional:.3e}')
        print(f'  analytic/FD J rel       = {matrix:.3e}')
        print(f'  Newton iterations       = {sol.iterations}')
        print(f'  Newton residual rel     = {sol.residual_relative:.3e}')
        print('  KL GL+Birger FDM ban    = PASS')
    return report


if __name__=='__main__':
    run_checks(True)
