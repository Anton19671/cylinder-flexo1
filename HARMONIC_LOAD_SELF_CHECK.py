#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Software self-check for generic HARMONIC_RADIAL_FACE.

No literature benchmark is embedded in the solver.  The check only verifies
that the generated load is exactly q*cos(n*s/R)*sin(m*pi*x/L), is accepted for
KL/CUF, and rejects invalid harmonic indices.
"""
from __future__ import annotations
import numpy as np

import cylinder_te1_mvi_core as mvi
import final_load_builder as loadbuilder
import final_te_basis_provider as tebasis


def run_checks(verbose=True):
    L=250.0; R=500.0; h=5.0; nx=21; ns=64; q0=1.0e-3; mm=1; nn=4
    spec={'type':'HARMONIC_RADIAL_FACE','q':q0,'m':mm,'n':nn,
          'load_factors':[0.25,1.0],'kinematics':'DEAD_REFERENCE'}
    loads,factors,meta,radial=loadbuilder.build_generated_loads(
        spec,theory='KL',L=L,R=R,h=h,nx=nx,ns=ns,stencil=7
    )
    grid=mvi.CylinderGrid(L,R,nx,ns)
    expected=q0*np.outer(np.cos(nn*grid.s/R),np.sin(mm*np.pi*grid.x/L))
    err=float(np.max(np.abs(radial-expected)))
    if err > 2e-15*max(1.0,abs(q0)):
        raise AssertionError(f'harmonic field mismatch: {err:.3e}')
    if loads.shape != (2,3,ns,nx):
        raise AssertionError(f'unexpected KL harmonic load shape: {loads.shape}')
    if not np.array_equal(factors,np.array([0.25,1.0])):
        raise AssertionError('harmonic load factors mismatch')
    if not np.allclose(radial[:,0],0.0,rtol=0.0,atol=2e-18):
        raise AssertionError('harmonic field is not zero at x=0')
    if not np.allclose(radial[:,-1],0.0,rtol=0.0,atol=2e-18):
        raise AssertionError('harmonic field is not zero at x=L')
    imid=(nx-1)//2
    if abs(radial[0,imid]-q0) > 2e-15:
        raise AssertionError('harmonic peak at x=L/2, theta=0 is incorrect')
    if meta.get('harmonic_m')!=mm or meta.get('harmonic_n')!=nn:
        raise AssertionError('harmonic metadata mismatch')
    if meta.get('load_front_surface_nodes') != 0:
        raise AssertionError('smooth harmonic load must not create PATCH front nodes')

    cuf_spec=dict(spec); cuf_spec['face']='OUTER'
    cuf_loads,cuf_factors,cuf_meta,cuf_radial=loadbuilder.build_generated_loads(
        cuf_spec,theory='CUF',L=L,R=R,h=h,nx=nx,ns=ns,stencil=7,basis=tebasis.get_basis('TE1')
    )
    if cuf_loads.shape != (2,3,2,ns,nx):
        raise AssertionError(f'unexpected CUF-TE1 harmonic load shape: {cuf_loads.shape}')
    if not np.array_equal(cuf_radial,radial):
        raise AssertionError('CUF and KL harmonic radial generator fields differ')
    if not np.array_equal(cuf_factors,factors):
        raise AssertionError('CUF/KL harmonic load factors differ')

    for bad in (
        {'type':'HARMONIC_RADIAL_FACE','q':q0,'m':0,'n':4,'load_factors':[1.0]},
        {'type':'HARMONIC_RADIAL_FACE','q':q0,'m':1.5,'n':4,'load_factors':[1.0]},
        {'type':'HARMONIC_RADIAL_FACE','q':q0,'m':1,'n':-1,'load_factors':[1.0]},
    ):
        try:
            loadbuilder.build_generated_loads(bad,theory='KL',L=L,R=R,h=h,nx=nx,ns=ns,stencil=7)
        except loadbuilder.LoadDefinitionError:
            pass
        else:
            raise AssertionError(f'invalid harmonic modes were accepted: {bad}')

    if verbose:
        print('HARMONIC_LOAD_SELF_CHECK: PASS')
        print(f'  analytic field max error = {err:.3e}')
        print('  KL/CUF-TE1 field identity + mode guards: PASS')
    return {'field_max_error':err,'status':'PASS'}


if __name__=='__main__':
    run_checks(verbose=True)
