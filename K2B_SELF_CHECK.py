#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2b verification: scalar thickness loops vs vectorized production paths."""
from __future__ import annotations
import json, time
import numpy as np

import cylinder_te1_stage2b as s2
import cylinder_cuf_stage7a_general_mechanics as g7
import cylinder_cuf_stage7f_general_linear as gl


def basis3():
    return g7.CallableThicknessBasis(
        funcs=(lambda z:1.0, lambda z:z, lambda z:z*z),
        dfuncs=(lambda z:0.0, lambda z:1.0, lambda z:2.0*z),
        names=("1","z","z2"),
    )


def fields(m,ns,nx):
    # Deterministic small field, including all components/basis terms.
    ss=np.arange(ns,dtype=float)[:,None]
    xx=np.arange(nx,dtype=float)[None,:]
    q=np.empty((3,m,ns,nx),float)
    qx=np.empty_like(q); qs=np.empty_like(q)
    for c in range(3):
        for t in range(m):
            a=(c+1)*(t+1)
            q[c,t]=1e-5*a*(np.sin((ss+1)/(3+t))+0.7*np.cos((xx+1)/(4+c)))
            qx[c,t]=2e-6*a*(np.cos((ss+1)/(5+c))+0.3*np.sin((xx+1)/(2+t)))
            qs[c,t]=1.5e-6*a*(np.sin((ss+1)/(4+t))-0.2*np.cos((xx+1)/(3+c)))
    return q,qx,qs


def compare_resultants(tag,a,b):
    max_abs=0.0; max_rel=0.0
    for key in g7.RESULTANT_KEYS:
        x=np.asarray(getattr(a,key),float); y=np.asarray(getattr(b,key),float)
        d=float(np.max(np.abs(x-y)))
        scale=max(float(np.max(np.abs(x))),float(np.max(np.abs(y))),1e-30)
        rel=d/scale
        max_abs=max(max_abs,d); max_rel=max(max_rel,rel)
        if d > 5e-11*max(1.0,scale):
            raise AssertionError(f"{tag} {key}: max_abs={d:.3e}, max_rel={rel:.3e}")
    return max_abs,max_rel


def timed(fn,reps=2):
    vals=[]; out=None
    for _ in range(reps):
        t=time.perf_counter(); out=fn(); vals.append(time.perf_counter()-t)
    return min(vals),out


def run():
    basis=basis3(); R=50.0; K0=58333.333333333336
    qd=s2.gauss_thickness_quadrature(1.0,5)
    q,qx,qs=fields(basis.m,15,15)
    G0=26923.076923076922
    surf=G0*(1.0+0.05*np.sin(np.arange(15)[:,None]/4.0)+0.03*np.cos(np.arange(15)[None,:]/5.0))
    full=np.stack([(1.0+0.02*g)*surf for g in range(qd.z.size)],axis=0)

    report={}
    for label,GB in (("scalar",G0),("surface",surf),("full",full)):
        ts,rs=timed(lambda: g7.general_thickness_resultants_scalar_reference(
            q,qx,qs,R=R,K0=K0,GB=GB,quadrature=qd,basis=basis),1)
        tv,rv=timed(lambda: g7.general_thickness_resultants(
            q,qx,qs,R=R,K0=K0,GB=GB,quadrature=qd,basis=basis),3)
        da,dr=compare_resultants("NONLINEAR "+label,rs,rv)
        report["nonlinear_"+label]={"scalar_s":ts,"vector_s":tv,"speedup":ts/tv,"max_abs":da,"max_rel":dr}

        ts,rs=timed(lambda: gl.general_linear_thickness_resultants_scalar_reference(
            q,qx,qs,R=R,K0=K0,GB=GB,quadrature=qd,basis=basis),1)
        tv,rv=timed(lambda: gl.general_linear_thickness_resultants(
            q,qx,qs,R=R,K0=K0,GB=GB,quadrature=qd,basis=basis),3)
        da,dr=compare_resultants("LINEAR "+label,rs,rv)
        report["linear_"+label]={"scalar_s":ts,"vector_s":tv,"speedup":ts/tv,"max_abs":da,"max_rel":dr}

    # Mandatory nonlinear detF admissibility guard must still fire in production.
    qb=np.zeros((3,1,3,3),float); qxb=np.zeros_like(qb); qsb=np.zeros_like(qb)
    b1=g7.CallableThicknessBasis((lambda z:1.0,),(lambda z:0.0,),('1',))
    # u_x=-2 -> F_xx=-1 and detF<0.
    qxb[0,0]=-2.0
    try:
        g7.general_thickness_resultants(qb,qxb,qsb,R=R,K0=K0,GB=G0,
                                        quadrature=s2.gauss_thickness_quadrature(1.0,3),basis=b1)
    except s2.InvalidDeformationMapping:
        report["invalid_detF_guard"]="PASS"
    else:
        raise AssertionError("K2b production path lost detF<=0 admissibility guard")

    print("K2B_SELF_CHECK: PASS")
    for k,v in report.items():
        if isinstance(v,dict):
            print(f"  {k}: scalar={v['scalar_s']:.6f}s vector={v['vector_s']:.6f}s "
                  f"speedup={v['speedup']:.2f}x max_rel={v['max_rel']:.3e}")
        else:
            print(f"  {k}: {v}")
    print("K2B_REPORT_JSON="+json.dumps(report,sort_keys=True))
    return report

if __name__=='__main__':
    run()
