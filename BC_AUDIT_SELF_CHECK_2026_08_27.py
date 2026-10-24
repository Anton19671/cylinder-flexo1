#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import final_boundary_registry as reg
import final_residual_scaling as rs
import cylinder_final_stage11a_dispatcher as disp
import cylinder_te1_stage2b as s2


def run():
    assert reg.kl_kinds('CLAMPED') == ('GEOMETRIC','GEOMETRIC','GEOMETRIC','GEOMETRIC')
    assert reg.kl_kinds('SIMPLY_SUPPORTED') == ('GEOMETRIC','GEOMETRIC','GEOMETRIC','NATURAL')
    assert reg.cuf_kinds('SIMPLY_SUPPORTED') == ('NATURAL','NATURAL','GEOMETRIC')

    # ELASTIC CUF rows must be accepted by V4 measure scaling and be traction-like.
    eng={'left':{c:[{'stiffness':1000.0,'reference':0.0,'traction':0.0} for _ in range(2)] for c in ('u','v','w')}}
    bc=disp._make_cuf_simple_bc(('ELASTIC_ENGINEERING','CLAMPED'),2,eng)
    m=rs.cuf_residual_measure_multipliers(basis_m=2,ns=7,nx=7,L=100.0,bc=bc,E0=70000.0,h=1.0)
    assert np.allclose(m[:,:, :,0],1.0/100.0)

    # Movable both-end pairs are physical but non-unique without a rigid-mode gauge.
    for pair in (('PINNED_S1','PINNED_S1'),('PINNED_S4','PINNED_S4'),('GUIDED','GUIDED')):
        try:
            disp._make_kl_simple_bc(pair,{})
        except Exception as exc:
            assert 'RIGID_MODE_HANDLING_REQUIRED' in str(exc)
        else:
            raise AssertionError(f'KL pair should require rigid-mode handling: {pair}')
    for mte in (2,3,4,5):
        try:
            disp._make_cuf_simple_bc(('SIMPLY_SUPPORTED','SIMPLY_SUPPORTED'),mte,{})
        except Exception as exc:
            assert 'RIGID_MODE_HANDLING_REQUIRED' in str(exc)
        else:
            raise AssertionError(f'CUF m={mte} S4-S4 should require rigid-mode handling')

    print('BC_AUDIT_SELF_CHECK_2026_08_27: PASS')
    return True

if __name__=='__main__':
    run()
