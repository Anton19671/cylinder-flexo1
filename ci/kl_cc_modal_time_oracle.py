#!/usr/bin/env python3
"""Exact-in-time semi-discrete linear KL forced response from independent K/M.

Physical operators and KL clamped traces are copied from the exact same
9x9 variational spatial model used by the Newmark FDM/MVI solvers.
Eigensolve is NOT a new continuum model or a fixed-rank MVI computation.
The sine-convolution formula is analytic for a LINEAR, undamped
semi-discrete system and provides a genuine independent time reference.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh,solve

from cylinder_reference_case import prepare_reference_case
from cylinder_modal_reference import _modal_spec,_linear_matrices

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"config/verification/kl_cc_dynamic_long/FDM_COARSE.json"
OUT=ROOT/"config/verification/kl_cc_full_period"
OUT.mkdir(parents=True,exist_ok=True)


def physical_case():
    case=json.loads(CONFIG.read_text(encoding="utf-8"))
    assert case["theory"]=="KL" and case["geometry_model"]=="LINEAR"
    assert case["method"]=="FDM" and case["scale_model"]=="MACRO"
    assert case["dynamics"]["time_law"]["frequency_hz"]==25
    assert case["dynamics"]["damping"]=="NONE"
    assert case["dynamics"]["initial_conditions"]["type"]=="ZERO"
    assert case["fdm_grid"]["nx"]==case["fdm_grid"]["ns"]==9
    assert case["fdm_grid"]["operator_scheme"]=="SBP42_FOURIER_ODD"
    original=prepare_reference_case(case,base_dir=CONFIG.parent)
    modal=copy.deepcopy(case)
    modal.pop("load_definition")
    modal.pop("dynamics")
    modal.pop("postprocess",None)
    modal["calculation_kind"]="MODAL"
    modal["modal"]={"density_kg_m3":7800,"n_modes":12}
    prepared,rho,_=_modal_spec(modal,base_dir=CONFIG.parent)
    return original,prepared,rho,case


def main():
    original,prepared,rho,case=physical_case()
    print("ASSEMBLING_K_M_REAL_KL_9x9",flush=True)
    k,m,bm,be,shape,p_shape=_linear_matrices(
        prepared,rho,
        progress_callback=lambda idx,total: print(
            "K_COLUMN",idx,"/",total,flush=True)
            if idx%24==0 or idx==total else None)
    if be.size or p_shape is not None:
        raise ValueError("MACRO_KL_ORACLE_UNEXPECTED_ELECTRICAL_DOFS")
    if k.shape!=m.shape or k.shape[0]<10:
        raise ValueError("UNEXPECTED_FREE_KL_SHAPE")
    w,phi=eigh(k,m,check_finite=True)
    if np.min(w)<=0 or not np.isfinite(w).all():
        raise ValueError("NONPOSITIVE_KL_ELASTIC_EIGENVALUE")
    hz=np.sqrt(w)/(2*np.pi)
    print("FIRST_12_REAL_MODAL_FREQUENCIES_HZ",hz[:12],flush=True)

    q0=prepared.mechanical
    unloaded=prepared.problem.evaluate_effective_residual(
        q0,prepared.electrical)[1]
    loaded=prepared.problem.evaluate_effective_residual(
        q0,prepared.electrical,dead_load=original.dead_load)[1]
    generalized_force=bm.T@(unloaded-loaded)
    modal_force=phi.T@generalized_force
    # Reference *static* displacement checks the physical forcing and the
    # continuity of these operators with the accepted static comparison.
    static=np.reshape(bm@solve(k,generalized_force,assume_a="pos"),shape)
    static_w=float(np.mean(static[2,shape[1]//2,:]))
    print("K_STATIC_MEAN_W_AT_100KPA_M",static_w,flush=True)
    if static_w>=0 or not (5e-6<abs(static_w)<15e-6):
        raise ValueError("ORACLE_PRESSURE_OR_STIFFNESS_NOT_CONSISTENT")
    mid=np.array([np.mean((bm@phi[:,j]).reshape(shape)[2,shape[1]//2,:])
                  for j in range(len(hz))])
    projected_mid=mid*modal_force/np.maximum(w,1e-300)
    modes=sorted(range(len(w)),key=lambda j:abs(projected_mid[j]),reverse=True)
    print("DOMINANT_STATIC_FORCED_KL_MODES",
          [(int(j+1),float(hz[j]),float(projected_mid[j]))
           for j in modes[:12]],flush=True)

    forcing_omega=2*np.pi*25.
    omega=np.sqrt(w)
    denominator=w-forcing_omega**2
    if min(abs(denominator)) < .01*min(w):
        raise ValueError("NEAR_EXACT_RESONANCE_NEEDS_SEPARATE_FORMULA")
    factor=modal_force/denominator
    times=np.arange(0,81,dtype=float)*0.0005
    rows=[]
    for t in times:
        qmodal=factor*(np.sin(forcing_omega*t)
                         -(forcing_omega/omega)*np.sin(omega*t))
        vmodal=factor*forcing_omega*(np.cos(forcing_omega*t)
                                     -np.cos(omega*t))
        amodal=factor*(-forcing_omega**2*np.sin(forcing_omega*t)
                       +forcing_omega*omega*np.sin(omega*t))
        modal_balance=m@phi@amodal+k@phi@qmodal-generalized_force*np.sin(forcing_omega*t)
        balance=np.linalg.norm(modal_balance)/max(np.linalg.norm(generalized_force),1e-30)
        if balance>3e-7:
            raise ValueError("ANALYTIC_MODAL_TRAJECTORY_NOT_IN_EQUILIBRIUM")
        q=(bm@(phi@qmodal)).reshape(shape)
        v=(bm@(phi@vmodal)).reshape(shape)
        acc=(bm@(phi@amodal)).reshape(shape)
        rows.append(dict(time_s=float(t),load_factor=float(np.sin(forcing_omega*t)),
                         mean_w_midspan_m=float(np.mean(q[2,shape[1]//2,:])),
                         max_abs_w_m=float(np.max(abs(q[2]))),
                         max_abs_velocity_m_per_s=float(np.max(abs(v))),
                         max_abs_acceleration_m_per_s2=float(np.max(abs(acc))),
                         balance_relative=float(balance)))
    csv_path=OUT/"KL_CC_25HZ_9X9_EXACT_SEMIDISCRETE_TIME_HISTORY.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A second, algebraically independent check: integrate the *same*
    # 171-dimensional linear system with average-acceleration Newmark in its
    # diagonal M-normalized modal coordinates. This uses no MVI iteration,
    # FDM Newton solver or numerical tangent during time integration.
    # It must reproduce the actual accepted MVI1/FDM discretized histories
    # even when dt is far too large for phase accuracy.
    replay=[]
    frozen_accepted_midspan={
        .002: 0.058247490e-6,
        .001: -0.136024749e-6,
        .0005: 0.123355274e-6,
    }
    for dt, steps in ((.002, 10), (.001, 20), (.0005, 40)):
        beta,gamma=.25,.5
        qm=np.zeros_like(w)
        vm=np.zeros_like(w)
        am=np.zeros_like(w)
        for i in range(1, steps+1):
            t=i*dt
            qp=qm+dt*vm+dt*dt*(.5-beta)*am
            qnext=(modal_force*np.sin(forcing_omega*t)+
                   qp/(beta*dt*dt))/(w+1./(beta*dt*dt))
            anext=(qnext-qp)/(beta*dt*dt)
            vnext=vm+dt*((1.-gamma)*am+gamma*anext)
            qm,vm,am=qnext,vnext,anext
            exact_index=int(round(t/.0005))
            if abs(exact_index*.0005-t)>1e-11:
                raise ValueError("REPLAY_TIME_NOT_ON_COMMON_GRID")
            replay.append(dict(
                dt_s=dt,step=i,time_s=float(t),
                mean_w_midspan_newmark_m=float(mid@qm),
                mean_w_midspan_exact_time_m=rows[exact_index]["mean_w_midspan_m"],
                newmark_minus_exact_midspan_m=float(
                    mid@qm-rows[exact_index]["mean_w_midspan_m"])))
        calculated=float(mid@qm)
        expected=frozen_accepted_midspan[dt]
        print("SPECTRAL_NEWMARK_REPLAY",dt,steps,"W_mid_m",calculated,
              "accepted_real_MVI1_W_mid_m",expected,
              "difference_m",calculated-expected,flush=True)
        # Grounded reference values were independently extracted from
        # accepted physical GitHub Action NPZ state artifacts.
        if abs(calculated-expected)>2.e-10:
            raise ValueError("SPECTRAL_NEWMARK_DIFFERS_FROM_REAL_MVI1")
    path_replay=OUT/"KL_CC_NEWMARK_SPECTRAL_REPLAY_3_DT.csv"
    with path_replay.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(replay[0]))
        writer.writeheader()
        writer.writerows(replay)

    manifest={
       "scope":"exact_in_time_for_same_linear_discrete_KL_9x9_model",
       "reference_type":"generalized_K_M_spectrum_and_sinusoidal_convolution",
       "independent_3D_validation":False,
       "external_literature_benchmark":False,
       "dt_newmark_comparison_required":True,
       "material":{"E_Pa":case["material"]["E0"],
                   "nu":case["material"]["nu0"],
                   "rho_kg_m3":rho},
       "geometry":case["geometry"],
       "forcing_hz":25.,
       "spatial_scheme":case["fdm_grid"]["operator_scheme"],
       "free_dofs":len(w),
       "frequency_hz_first_12":hz[:12].tolist(),
       "dominant_forced_modes":[
          {"mode":int(j+1),"frequency_hz":float(hz[j]),
           "static_midspan_contribution_m":float(projected_mid[j])}
          for j in modes[:14]],
       "static_midspan_w_m":static_w,
       "endpoint_reference_midspan_w_m":rows[-1]["mean_w_midspan_m"],
       "method":"MKR_FDM_MODAL_KM_NO_NEWMARK_NO_GAUSS",
       "full_interval_s":.04,
       "n_samples":81,
    }
    out=OUT/"KL_CC_25HZ_9X9_MODAL_TIME_ORACLE.json"
    out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print("EXACT_SEMIDISCRETE_DYNAMIC_REFERENCE",json.dumps({
        "frequency_hz_top":hz[:12].tolist(),
        "dominant_midspan_contributors":manifest["dominant_forced_modes"][:8],
        "static_midspan_m":static_w,
        "mean_w_0_020s_m":rows[40]["mean_w_midspan_m"],
        "mean_w_0_040s_m":rows[80]["mean_w_midspan_m"],
    }),flush=True)


if __name__=="__main__":
    main()
