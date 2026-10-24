#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted reviewer V-6 regression for the Birger outer acceptance criteria.

Checks only the audit requirement:
1) eta_G and eta_sigma alone cannot accept a drifting displacement field;
2) a stable field is accepted;
3) for MVI the V-6 field norm is evaluated on the reconstructed field, not on
   gauge-dependent raw X/Y factors.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np

import cylinder_te1_stage5a_birger as b5
import cylinder_te1_mvi_core as mvi


def _settings(tol_u):
    return b5.BirgerSettings(
        epsilon_tol=1.0e-12,
        omega=1.0,
        G_ref=10.0,
        sigma_ref=1.0,
        tol_G=1.0e-12,
        tol_sigma=1.0e-12,
        tol_u=float(tol_u),
        max_outer_iterations=2,
        epsilon_i_limit=1.0,
        delta_J_limit=1.0,
        initial_slope_rtol=0.05,
    )


def _phi(eps):
    return 30.0*np.asarray(eps,float)  # 3*G0*eps, G0=10


def _material(_state,_GB):
    z=np.zeros((1,),float)
    return b5.MaterialPointState(z,z,np.ones_like(z))


def _drifting_field_must_not_accept():
    calls={'n':0}
    def inner(_state,_GB):
        calls['n']+=1
        return SimpleNamespace(
            state=np.array([float(calls['n'])]),
            converged=True,
            status='OK',
        )
    r=b5.solve_birger_outer_cycle(
        np.array([0.0]),np.array([10.0]),
        G0=10.0,material_diagram=_phi,inner_solve=inner,
        evaluate_material=_material,
        state_vector=lambda st: np.asarray(st,float),
        settings=_settings(0.1),
    )
    if r.converged or r.status!='MAX_BIRGER_OUTER_ITERATIONS':
        raise AssertionError('V-6 accepted a drifting field although eta_G=eta_sigma=0')
    if not all(h['eta_G_fin']==0.0 and h['eta_sigma_fin']==0.0 for h in r.history):
        raise AssertionError('synthetic V-6 test did not isolate eta_u')
    if not all(h['eta_u']>0.1 for h in r.history):
        raise AssertionError('drifting field did not violate tol_u')
    return r


def _stable_field_must_accept():
    def inner(_state,_GB):
        return SimpleNamespace(state=np.array([2.0]),converged=True,status='OK')
    r=b5.solve_birger_outer_cycle(
        np.array([0.0]),np.array([10.0]),
        G0=10.0,material_diagram=_phi,inner_solve=inner,
        evaluate_material=_material,
        state_vector=lambda st: np.asarray(st,float),
        settings=_settings(1.0e-12),
    )
    if not r.converged or r.outer_iterations!=1 or r.eta_u!=0.0:
        raise AssertionError('V-6 stable field was not accepted')
    return r


def _mvi_gauge_must_not_fake_field_change():
    grid=mvi.CylinderGrid(1.0,1.0,7,7)
    def state(scale):
        X=np.zeros((6,1,grid.nx),float)
        Y=np.zeros((6,1,grid.ns),float)
        X[0,0]=float(scale)
        Y[0,0]=1.0/float(scale)
        return mvi.ProfileStateTE1(X,Y)
    A=state(2.0); B=state(4.0)
    raw_diff=np.linalg.norm(np.r_[A.X.ravel()-B.X.ravel(),A.Y.ravel()-B.Y.ravel()])
    if raw_diff<=0.0:
        raise AssertionError('MVI gauge test is degenerate')
    seq=[A,B]
    calls={'n':0}
    def inner(_state,_GB):
        i=min(calls['n'],len(seq)-1); calls['n']+=1
        return SimpleNamespace(state=seq[i],converged=True,status='OK')
    r=b5.solve_birger_outer_cycle(
        A,np.array([10.0]),
        G0=10.0,material_diagram=_phi,inner_solve=inner,
        evaluate_material=_material,
        state_vector=lambda st: mvi.reconstruct_vector(st,grid),
        settings=_settings(1.0e-12),
    )
    if not r.converged or r.eta_u>1.0e-14:
        raise AssertionError(f'MVI raw-factor rescaling polluted V-6 eta_u: {r.eta_u:.3e}')
    return raw_diff,r


def run_checks(verbose=True):
    d=_drifting_field_must_not_accept()
    s=_stable_field_must_accept()
    raw,g=_mvi_gauge_must_not_fake_field_change()
    rep={
        'status':'PASS',
        'drifting_status':d.status,
        'drifting_eta_u':[float(h['eta_u']) for h in d.history],
        'stable_eta_u':float(s.eta_u),
        'mvi_raw_factor_difference':float(raw),
        'mvi_reconstructed_eta_u':float(g.eta_u),
    }
    if verbose:
        print('REVIEWER_V6_SELF_CHECK: PASS')
        print('  eta_G=eta_sigma=0 + drifting U: NOT ACCEPTED')
        print('  stable U: ACCEPTED')
        print(f'  MVI raw X/Y rescaling: raw_diff={raw:.6e}, reconstructed eta_u={g.eta_u:.6e}')
    return rep


if __name__=='__main__':
    run_checks(True)
