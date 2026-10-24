#!/usr/bin/env python3
from __future__ import annotations
import json, math
import numpy as np
import cylinder_te1_stage3a_connector as c3


def _raw_cond(h):
    dx=np.arange(-3,4,dtype=float)*h
    A=np.vstack([dx**k for k in range(7)])
    return float(np.linalg.cond(A))

def _scaled_cond(h):
    dx=np.arange(-3,4,dtype=float)*h
    t=dx/np.max(np.abs(dx))
    A=np.vstack([t**k for k in range(7)])
    return float(np.linalg.cond(A))

def run_checks(verbose=True):
    hs=[1.0,1e-2,1e-5,1e3]
    sc=[_scaled_cond(h) for h in hs]
    if max(sc)/min(sc) > 1.0+1e-10:
        raise AssertionError(f'Z1 scaled Vandermonde condition changed with h: {sc}')
    # Exact derivative of a degree-6 polynomial at x0 must be reproduced to roundoff-scaled tolerance.
    errs=[]
    for h in hs:
        x0=2.5*h; xs=x0+np.arange(-3,4)*h
        w=c3._fd_weights(x0,xs,1)
        f=0.3+0.7*xs-0.2*xs**2+0.05*xs**3
        exact=0.7-0.4*x0+0.15*x0*x0
        got=float(w@f)
        scale=max(1.0,abs(exact))
        errs.append(abs(got-exact)/scale)
        if errs[-1] > 5e-8:
            raise AssertionError(f'Z1 derivative error too large for h={h}: {errs[-1]}')
    rep={'status':'PASS','h':hs,'scaled_cond':sc,'raw_cond':[_raw_cond(h) for h in hs],'relative_errors':errs}
    if verbose:
        print('Z1_SELF_CHECK: PASS')
        print(f'  scaled cond range: {min(sc):.6g} .. {max(sc):.6g} (h-independent)')
        print(f'  raw cond max: {max(rep["raw_cond"]):.3e}')
        print(f'  max polynomial derivative relative error: {max(errs):.3e}')
        print('Z1_REPORT_JSON='+json.dumps(rep,sort_keys=True))
    return rep
if __name__=='__main__': run_checks(True)
