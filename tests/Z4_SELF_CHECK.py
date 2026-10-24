#!/usr/bin/env python3
from __future__ import annotations
import json,tempfile
from pathlib import Path
import numpy as np
import cylinder_te1_stage2b as s2
import cylinder_final_stage11a_dispatcher as d


def run_checks(verbose=True):
    rep={'status':'PASS'}; G0=10.0; ns=7; nx=7; h=1.0; q3=s2.gauss_thickness_quadrature(h,3)
    with tempfile.TemporaryDirectory(prefix='z4_gb_') as td:
        td=Path(td); casep=td/'case.json'; casep.write_text('{}')
        gb=np.full((3,ns,nx),G0)
        good=td/'good.npz'; np.savez_compressed(good,GB=gb,quadrature_z=q3.z,quadrature_w=q3.w,thickness_h=np.asarray(h))
        cfg={'initial_GB':{'source':'NPZ','npz':str(good)}}
        got=d._initial_GB(cfg,casep,G0,q3,h,ns,nx)
        assert np.array_equal(got,gb); rep['matching_metadata']='PASS'

        legacy=td/'legacy.npz'; np.savez_compressed(legacy,GB=gb)
        cfg['initial_GB']['npz']=str(legacy)
        try:d._initial_GB(cfg,casep,G0,q3,h,ns,nx)
        except d.FinalCaseError as exc:
            assert 'lacks quadrature metadata' in str(exc); rep['legacy_rejected']=True
        else:raise AssertionError('Z4 failed: legacy GB without integration metadata accepted')

        wrongh=td/'wrongh.npz'; np.savez_compressed(wrongh,GB=gb,quadrature_z=q3.z,quadrature_w=q3.w,thickness_h=np.asarray(2.0))
        cfg['initial_GB']['npz']=str(wrongh)
        try:d._initial_GB(cfg,casep,G0,q3,h,ns,nx)
        except d.FinalCaseError as exc:
            assert 'thickness_h' in str(exc); rep['wrong_h_rejected']=True
        else:raise AssertionError('Z4 failed: GB from different thickness accepted')

        # Same nz but different integration-point locations must also be rejected.
        qlayer=s2.layered_thickness_quadrature(h,[-0.5,0.0,0.5],[1,2])
        assert qlayer.z.size==q3.z.size
        badq=td/'badq.npz'; np.savez_compressed(badq,GB=gb,quadrature_z=qlayer.z,quadrature_w=qlayer.w,thickness_h=np.asarray(h))
        cfg['initial_GB']['npz']=str(badq)
        try:d._initial_GB(cfg,casep,G0,q3,h,ns,nx)
        except d.FinalCaseError as exc:
            assert 'quadrature points/weights' in str(exc); rep['same_nz_wrong_quadrature_rejected']=True
        else:raise AssertionError('Z4 failed: same-nz but different quadrature accepted')

    if verbose:
        print('Z4_SELF_CHECK: PASS')
        print('  matching h/z/w metadata: ACCEPTED')
        print('  legacy GB without metadata: REJECTED')
        print('  different thickness: REJECTED')
        print('  same nz but different quadrature: REJECTED')
        print('Z4_REPORT_JSON='+json.dumps(rep,sort_keys=True))
    return rep
if __name__=='__main__':run_checks(True)
