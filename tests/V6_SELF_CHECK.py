#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6 regression: layered through-thickness quadrature is available end-to-end."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import numpy as np

import cylinder_te1_stage2b as s2
import cylinder_final_stage11a_dispatcher as disp
import cylinder_kl_stage8d3_launcher as kl_launcher

BASE=Path(__file__).resolve().parent


def _case(theory, qspec, out="OUT"):
    cfg={
        "theory":theory,"cuf_model":"TE1","method":"FDM","mode":"LINEAR_ELASTIC",
        "units":{"system":"N_MM_MPA"},
        "geometry":{"L":100.0,"R":50.0,"h":1.0},
        "material":{"E0":70000.0,"nu0":0.3},
        "fdm_grid":{"nx":7,"ns":7,"stencil":7},
        "mvi_grid":{"nx":7,"ns":7,"stencil":7},
        "quadrature":qspec,
        "boundary_conditions":{"left":"CLAMPED","right":"CLAMPED","advanced":None},
        "load_definition":{"type":"UNIFORM_RADIAL_FACE","q":0.01,"face":"OUTER","load_factors":[1.0]},
        "linear":{"residual_tolerance_relative":1.0e-10},
        "output_dir":out,
    }
    if str(theory).upper()=="CUF":
        cfg["load_definition"]["face"]="OUTER"
    else:
        cfg["load_definition"].pop("face",None)
    return cfg


def _run(cfg, td, name):
    p=Path(td)/f"{name}.json"
    p.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    v=disp.validate_final_case(p)
    r=disp.run_final_case(p)
    if not r.get("converged"):
        raise AssertionError(f"{name}: final dispatcher did not converge: {r.get('status')}")
    key="q" if cfg["theory"]=="CUF" else "U"
    with np.load(Path(td)/cfg["output_dir"]/"STEP_0001_STATE.npz",allow_pickle=False) as z:
        st=np.asarray(z[key],float)
    w=st[2,0] if st.ndim==4 else st[2]
    return v,r,float(np.max(np.abs(w)))


def run_quick_checks(verbose=False):
    h=2.0; zk=0.23; exact=1.0+zk*zk
    q3=s2.gauss_thickness_quadrature(h,3)
    layered=s2.layered_thickness_quadrature(h,[-1.0,zk,1.0],2)
    f3=float(q3.integrate(np.abs(q3.z-zk))); fl=float(layered.integrate(np.abs(layered.z-zk)))
    if abs(fl-exact)>5e-15 or abs(f3-exact)<=1e-5:
        raise AssertionError("V6 quick kink quadrature guard failed")
    _,_,warn=disp._quadrature_config({"type":"GAUSS","n_gauss":3},1.0,"NONLINEAR_BIRGER")
    _,_,warn7=disp._quadrature_config({"type":"GAUSS","n_gauss":7},1.0,"NONLINEAR_BIRGER")
    if not warn or warn7:
        raise AssertionError("V6 quick Birger quadrature warning guard failed")
    ql=disp._quadrature_config({"type":"LAYERED","layer_bounds":[-0.5,0.0,0.5],"n_gauss":2},1.0,"LINEAR_ELASTIC")[1]
    if ql.z.size!=4 or not np.isclose(np.sum(ql.w),1.0):
        raise AssertionError("V6 quick LAYERED dispatcher guard failed")
    if verbose: print("V6_QUICK_CHECK: PASS")
    return {"status":"PASS","kink_gauss3_error":abs(f3-exact),"kink_layered_error":abs(fl-exact)}


def run_checks(verbose=True):
    # A kinked through-thickness function: splitting exactly at the kink lets
    # 2-point Gauss integrate each linear piece exactly.  A single 3-point rule
    # cannot have that property when the kink is not a quadrature node.
    h=2.0; zk=0.23
    exact=1.0+zk*zk
    q3=s2.gauss_thickness_quadrature(h,3)
    layered=s2.layered_thickness_quadrature(h,[-1.0,zk,1.0],2)
    f3=float(q3.integrate(np.abs(q3.z-zk)))
    fl=float(layered.integrate(np.abs(layered.z-zk)))
    err3=abs(f3-exact); errl=abs(fl-exact)
    if errl>5e-15:
        raise AssertionError(f"V6 layered kink integration not exact enough: {errl:.3e}")
    if not err3>1e-5:
        raise AssertionError("V6 kink test did not distinguish GAUSS3 from layer-aligned quadrature")

    # Dispatcher supports both GAUSS and LAYERED.  For the elastic polynomial
    # smoke problem they should give the same physical solution.
    rows={}
    with tempfile.TemporaryDirectory(prefix="v6_") as td:
        for theory in ("CUF","KL"):
            ga=_case(theory,{"type":"GAUSS","n_gauss":3},out=f"{theory}_G")
            la=_case(theory,{"type":"LAYERED","layer_bounds":[-0.5,0.0,0.5],"n_gauss":2},out=f"{theory}_L")
            vg,rg,wg=_run(ga,td,f"{theory}_gauss")
            vl,rl,wl=_run(la,td,f"{theory}_layered")
            rel=abs(wg-wl)/max(abs(wg),abs(wl),1e-300)
            if rel>2e-8:
                raise AssertionError(f"V6 {theory}: GAUSS/LAYERED elastic mismatch {rel:.3e}")
            if vl.get("quadrature",{}).get("type")!="LAYERED":
                raise AssertionError(f"V6 {theory}: dispatcher did not preserve LAYERED spec")
            rows[theory]={"w_gauss":wg,"w_layered":wl,"rel_difference":rel,
                          "layered_points":int(sum(vl["quadrature"]["n_gauss_per_layer"]))}

    # Birger with fewer than 7 global Gauss points is a warning, not a ban.
    spec,quad,warn=disp._quadrature_config({"type":"GAUSS","n_gauss":3},1.0,"NONLINEAR_BIRGER")
    if not warn or "n_gauss=3" not in warn[0]:
        raise AssertionError("V6 Birger GAUSS3 warning missing")
    spec7,quad7,warn7=disp._quadrature_config({"type":"GAUSS","n_gauss":7},1.0,"NONLINEAR_BIRGER")
    if warn7:
        raise AssertionError(f"V6 unexpected Birger GAUSS7 warning: {warn7}")

    # KL MVI launcher path accepts the same canonical LAYERED description.
    qkl=kl_launcher._build_quadrature(
        {"type":"LAYERED","layer_bounds":[-0.5,0.0,0.5],"n_gauss_per_layer":[2,3]},1.0
    )
    if qkl.z.size!=5 or not np.isclose(np.sum(qkl.w),1.0,rtol=0.0,atol=2e-15):
        raise AssertionError("V6 KL MVI launcher LAYERED quadrature mismatch")

    # Invalid bounds must be rejected at the final dispatcher boundary.
    try:
        disp._quadrature_config({"type":"LAYERED","layer_bounds":[-0.4,0.5],"n_gauss":2},1.0,"LINEAR_ELASTIC")
    except disp.FinalCaseError:
        invalid_rejected=True
    else:
        raise AssertionError("V6 accepted layer bounds that do not span -h/2..+h/2")

    rep={
        "status":"PASS",
        "kink_integral":{"exact":exact,"gauss3":f3,"gauss3_abs_error":err3,
                         "layered":fl,"layered_abs_error":errl,"kink_z":zk},
        "elastic_dispatcher_cases":rows,
        "birger_gauss3_warning":warn,
        "birger_gauss7_warning":warn7,
        "kl_mvi_layered_points":int(qkl.z.size),
        "invalid_bounds_rejected":invalid_rejected,
    }
    if verbose:
        print("V6_SELF_CHECK: PASS")
        print(f"  kink integral: GAUSS3 error={err3:.3e}; LAYERED error={errl:.3e}")
        for theory,row in rows.items():
            print(f"  {theory}: LAYERED dispatcher run PASS, elastic rel diff={row['rel_difference']:.3e}")
        print("  Birger GAUSS3: warning emitted; GAUSS7: no warning")
        print("  invalid layer bounds: REJECTED")
    return rep


if __name__=="__main__":
    rep=run_checks(True)
    (BASE/"V6_VERIFICATION_RESULTS.json").write_text(json.dumps(rep,indent=2,sort_keys=True),encoding="utf-8")
    print("V6_REPORT_JSON="+json.dumps(rep,sort_keys=True))
