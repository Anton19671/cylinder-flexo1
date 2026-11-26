"""Native FDM/MVI convergence matrix. Every method executes its own solver.
Reuse only the audited, constant constitutive tangent for the same material.
No FDM displacement is passed to MVI; every rank uses its own zero initial
displacement and prescribed rank seed. Save failures separately.
"""
import argparse
import contextlib
import itertools
import json
import signal
import time
import traceback
from pathlib import Path
import numpy as np
from cylinder_paths import activate
activate()
import gao_park_plane_strain as bench

GRIDS=((21,9),(31,13),(41,17))
METHODS=(('FDM',None),)+tuple(('MVI',r) for r in range(1,9))


def _timeout(*_):
    raise TimeoutError('GAO_PARK_CASE_TIMEOUT')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--order',type=int,choices=range(1,5),required=True)
    p.add_argument('--ell-index',type=int,choices=range(4),required=True)
    p.add_argument('--output',default='evidence')
    p.add_argument('--timeout',type=int,default=300)
    a=p.parse_args()
    if a.timeout<=0: p.error('positive timeout required')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    bench.CACHE_LINEAR_LOCAL=True
    bad=0
    signal.signal(signal.SIGALRM,_timeout)
    for (nx,ns),(method,rank) in itertools.product(GRIDS,METHODS):
        bench.NX,bench.NS=nx,ns
        name=f'{nx}x{ns}/{method}_{rank or 0}'
        dest=out/name
        dest.mkdir(parents=True,exist_ok=True)
        t0=time.monotonic()
        rc=1
        print('START '+name,flush=True)
        with (dest/'run.log').open('w',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                signal.alarm(a.timeout)
                result,profile=bench.run(a.order,a.ell_index,method,rank)
                result['elapsed_seconds']=time.monotonic()-t0
                (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                np.savez_compressed(dest/'profile.npz',profile_nm=profile,radius_um=(bench.R+bench.H/2*np.linspace(-1.,1.,401))*1e6,nx=nx,ns=ns)
                print(json.dumps(result),flush=True)
                rc=0 if result['benchmark_pass'] else 2
            except Exception:
                traceback.print_exc()
                rc=124 if isinstance(__import__('sys').exc_info()[1],TimeoutError) else 1
            finally:
                signal.alarm(0)
                (dest/'exit_code.txt').write_text(str(rc)+'\n')
        print(f'END {name} exit={rc} seconds={time.monotonic()-t0:.3f}',flush=True)
        if (dest/'result.json').exists(): print((dest/'result.json').read_text(),flush=True)
        if rc:
            bad+=1
            print((dest/'run.log').read_text()[-3000:],flush=True)
    print(f'CASES=27 FAILED={bad}',flush=True)
    raise SystemExit(1 if bad else 0)


if __name__=='__main__': main()
