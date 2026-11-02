#!/usr/bin/env python3
"""Independent mathematical oracle for modal Y/X fixed-rank optimization.

A known nonseparable FIRST eigenmode has separation rank exactly two.
The constrained full-space K=4I-3qq^T, M=I has exact lowest eigenvalue 1.
For MVI rank 1, the best possible eigenvalue is 4-3*(.7)=1.9.
For MVI rank >= 2, the best possible eigenvalue is 1.

This is an algorithm isolation experiment; no cylinder physics is claimed.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigh

from cylinder_paths import activate
activate()
from cylinder_modal_mvi import _SeparatedModalProblem, solve_modal_mvi

HERE=Path(__file__).resolve().parents[1]
DEST=HERE/"modal_mvi_nonseparable_oracle_result.json"
NX,NS=7,9
TARGET_SINGULAR_SQUARES=(.7,.3)


def known_mode():
    axial=np.arange(NX,dtype=float)
    circum=np.arange(NS,dtype=float)
    u1=np.sin(np.pi*axial/(NX-1))
    u2=np.sin(2*np.pi*axial/(NX-1))
    u1/=np.linalg.norm(u1)
    u2/=np.linalg.norm(u2)
    v1=np.cos(2*np.pi*circum/NS)
    v2=np.sin(4*np.pi*circum/NS)
    v1/=np.linalg.norm(v1)
    v2/=np.linalg.norm(v2)
    q=np.sqrt(.7)*np.outer(u1,v1)+np.sqrt(.3)*np.outer(u2,v2)
    q[0]=q[-1]=0.
    return q.ravel()/np.linalg.norm(q)


def operator_for_rank(rank, target):
    prepared=SimpleNamespace(
        method="MVI",
        mvi_settings=SimpleNamespace(rank=rank,max_sweeps=60))
    op=object.__new__(_SeparatedModalProblem)
    op.prepared=prepared
    op.f=SimpleNamespace(theory="SYNTHETIC_NONSEPARABLE_RANK_TWO")
    op.rank=rank
    op.nx,op.ns=NX,NS
    op.n_components=1
    op.nm=NX*NS
    op.ne=0
    op.shape=(1,NX,NS)
    op.scales=np.ones(op.nm)
    op.large_harmonic=False
    op._previous_mass_cache={}
    op.cm=np.zeros((2*NS,op.nm))
    for j in range(NS):
        op.cm[j,j]=1.
        op.cm[NS+j,(NX-1)*NS+j]=1.
    op.ce=np.empty((0,0))
    op.n_mechanical_free_dofs=(NX-2)*NS
    op.n_electric_free_dofs=0
    op.mass=lambda q:np.asarray(q,dtype=float).ravel()
    op.stiffness=lambda q:(
        4.*np.asarray(q,dtype=float).ravel()
        -3.*target*np.dot(target,np.asarray(q,dtype=float).ravel()))
    return prepared,op


def main():
    q=known_mode()
    singular=np.linalg.svd(q.reshape(NX,NS),compute_uv=False)
    report={
        "problem":"K=4I-3qqT, M=I, C: axial endpoints zero",
        "theoretical_eigenvalues":{
            "full_space":1.,
            "rank_1_minimum":4.-3.*TARGET_SINGULAR_SQUARES[0],
            "rank_2_minimum":1.},
        "target_component_rank":int(np.count_nonzero(singular>1.e-10)),
        "target_singular_value_squares":(singular[:2]**2).tolist(),
        "runs":{},"all_passed":False}
    expected={1:1.9,2:1.,3:1.}
    for rank in (1,2,3):
        name=f"MVI_{rank}"
        prep,op=operator_for_rank(rank,q)
        try:
            r=solve_modal_mvi(prep,1.,1,
                              _cached_operator=op,
                              eigen_tolerance=1.e-9,
                              mode_tolerance=1.e-5)
            candidate=np.asarray(r["mechanical_modes"][0]).ravel()
            lam=float(r["omega_squared_rad2_s2"][0])
            actual_rank=int(np.linalg.matrix_rank(candidate.reshape(NX,NS),
                                                   tol=1.e-9))
            dim=r["mvi_final_xy_trial_dimensions"][0]
            diff=abs(lam-expected[rank])
            match=(diff<2.e-4 and actual_rank<=rank)
            report["runs"][name]={
                "status":"SOLVED" if match else "FAILED_ORACLE",
                "lambda_computed":lam,"lambda_expected":expected[rank],
                "lambda_abs_error":diff,
                "frequency_hz":float(r["frequencies_hz"][0]),
                "effective_component_rank":actual_rank,
                "mvi_sweeps":r["mvi_sweeps"],
                "xy_dimensions":dim}
            print("[RANK ORACLE] "+name+" "+json.dumps(report["runs"][name]),
                  flush=True)
        except Exception as exc:
            report["runs"][name]={
                "status":"ERROR","exception":repr(exc),
                "traceback":traceback.format_exc(limit=9)}
            print("[RANK ORACLE] "+name+" ERROR "+repr(exc),flush=True)
    report["all_passed"]=(
        report["target_component_rank"]==2
        and all(report["runs"].get(f"MVI_{rank}",{}).get("status")=="SOLVED"
                for rank in (1,2,3)))
    DEST.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                    encoding="utf-8")
    print("[RANK ORACLE] SUMMARY " +
          json.dumps({"all_passed":report["all_passed"],
                      "statuses":{n:x["status"] for n,x in report["runs"].items()}}),
          flush=True)
    return 0 if report["all_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
