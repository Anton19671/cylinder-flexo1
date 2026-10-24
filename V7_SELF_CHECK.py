#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V7 regression: physical plastic-volume fraction uses quadrature and H=1+z/R."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import numpy as np

import cylinder_te1_stage2b as s2
import FINAL_POSTPROCESS as pp

BASE=Path(__file__).resolve().parent


def run_checks(verbose=True):
    q=s2.gauss_thickness_quadrature(2.0,3)
    # Both extreme Gauss surfaces plastic, center elastic: audit example.
    mask=np.zeros((3,2,2),bool); mask[0]=True; mask[2]=True
    vf=pp._weighted_volume_fraction(mask,q.z,q.w,R=50.0)
    legacy=float(np.mean(mask))
    expected=5.0/9.0
    if not np.isclose(vf,expected,rtol=0.0,atol=2e-15):
        raise AssertionError(f"V7 symmetric Gauss3 fraction {vf} != {expected}")
    if not np.isclose(legacy,2.0/3.0,rtol=0.0,atol=2e-15):
        raise AssertionError("V7 legacy point fraction setup mismatch")

    # One outer (+z) surface only: cylindrical H multiplier must matter.
    mask2=np.zeros((3,3,4),bool); mask2[2]=True
    vf2=pp._weighted_volume_fraction(mask2,q.z,q.w,R=2.0)
    wz=q.w*(1.0+q.z/2.0)
    expected2=float(wz[2]/np.sum(wz))
    noH=float(q.w[2]/np.sum(q.w))
    if not np.isclose(vf2,expected2,rtol=0.0,atol=2e-15):
        raise AssertionError(f"V7 H-weighted result mismatch: {vf2} vs {expected2}")
    if abs(vf2-noH)<1e-3:
        raise AssertionError("V7 test failed to exercise cylindrical H weighting")

    # Layered/non-equal quadrature must use the actual weights, not infer them
    # from nz or rebuild a single global Gauss rule.
    ql=s2.layered_thickness_quadrature(2.0,[-1.0,-0.2,0.6,1.0],[2,3,2])
    ml=np.zeros((ql.z.size,2,1),bool); ml[1:5]=True
    vfl=pp._weighted_volume_fraction(ml,ql.z,ql.w,R=3.0)
    manual=float(np.sum(ql.w[1:5]*(1.0+ql.z[1:5]/3.0))/np.sum(ql.w*(1.0+ql.z/3.0)))
    if not np.isclose(vfl,manual,rtol=0.0,atol=3e-15):
        raise AssertionError("V7 layered weighted fraction mismatch")

    # Check public Birger report semantics without producing plots.
    orig_map,orig_vol=pp.save_map,pp.save_volume_scatter
    try:
        pp.save_map=lambda *a,**k: None
        pp.save_volume_scatter=lambda *a,**k: None
        with tempfile.TemporaryDirectory(prefix="v7_") as td:
            root=Path(td)
            roots={"birger":root/"birger","birger3d":root/"birger3d","report":root/"report"}
            for p in roots.values():p.mkdir(parents=True,exist_ok=True)
            ei=np.zeros(mask.shape,float); ei[mask]=1.0
            d={"GB":np.ones(mask.shape,float),"epsilon_i":ei,"sigma_i":ei,
               "z":q.z.copy(),"quadrature_weights":q.w.copy()}
            summary={"birger_definition":{"curve":{"yield_strain":0.5}},"postprocess":{}}
            info=pp._birger_outputs(d,summary,roots,L=10.0,R=50.0,h=2.0,G0=1.0)
            if "plastic_volume_fraction" not in info:
                raise AssertionError("V7 physical plastic_volume_fraction missing from report")
            if "plastic_volume_fraction_grid" in info:
                raise AssertionError("V7 obsolete ambiguous plastic_volume_fraction_grid still emitted")
            if not np.isclose(info["plastic_volume_fraction"],expected,rtol=0.0,atol=2e-15):
                raise AssertionError("V7 Birger report physical fraction mismatch")
            if not np.isclose(info["plastic_point_fraction_unweighted"],2.0/3.0,rtol=0.0,atol=2e-15):
                raise AssertionError("V7 diagnostic unweighted point fraction mismatch")
            disk=json.loads((roots["report"]/"PLASTIC_ZONE.json").read_text(encoding="utf-8"))
            if not np.isclose(disk["plastic_volume_fraction"],expected,rtol=0.0,atol=2e-15):
                raise AssertionError("V7 PLASTIC_ZONE.json weighted fraction mismatch")
    finally:
        pp.save_map,pp.save_volume_scatter=orig_map,orig_vol

    rep={
        "status":"PASS",
        "audit_gauss3_example":{"weighted_volume_fraction":vf,"unweighted_point_fraction":legacy,"expected":expected},
        "cylindrical_metric_example":{"weighted":vf2,"manual":expected2,"without_H":noH},
        "layered_example":{"points":int(ql.z.size),"weighted":vfl,"manual":manual},
        "report_key":"plastic_volume_fraction",
        "diagnostic_key":"plastic_point_fraction_unweighted",
    }
    if verbose:
        print("V7_SELF_CHECK: PASS")
        print(f"  audit Gauss3 mask: weighted={vf:.12g}, old unweighted={legacy:.12g}")
        print(f"  H=1+z/R exercised: weighted={vf2:.12g}, without H={noH:.12g}")
        print(f"  LAYERED actual weights: PASS ({ql.z.size} points)")
        print("  report key: plastic_volume_fraction; old ambiguous key removed")
    return rep


if __name__=="__main__":
    rep=run_checks(True)
    (BASE/"V7_VERIFICATION_RESULTS.json").write_text(json.dumps(rep,indent=2,sort_keys=True),encoding="utf-8")
    print("V7_REPORT_JSON="+json.dumps(rep,sort_keys=True))
