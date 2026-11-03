#!/usr/bin/env python3
"""Diagnostic only: inspect exact Gauss2 MVI endpoint rank failure.

Does not change stage equations, solver settings, rank acceptance, or returned
fields. The original strict rank check is called unmodified and still raises.
"""
import argparse
from pathlib import Path
import numpy as np
import cylinder_gauss2_mvi as gauss2

_original=gauss2._rank_factors

def _audited_rank_factors(field,rank,tolerance,label):
    try:
        return _original(field,rank,tolerance,label)
    except ValueError as exc:
        if "FIELD_OUTSIDE_FIXED_RANK" in str(exc):
            print("STRICT_ENDPOINT_RANK_REJECTION",label,
                  "requested_rank",rank,"relative_threshold",tolerance,
                  flush=True)
            for index,component in enumerate(np.asarray(field,float)):
                # Read-only pivoted QR diagnostic: same factorization
                # as the strict fixed-rank code, no SVD or Gram matrix.
                from scipy.linalg import qr
                basis, triangular, _ = qr(component.T,mode="economic",
                                          pivoting=True,check_finite=False)
                subset=basis[:,:rank]
                reconstructed=(component@subset)@subset.T
                amplitude=float(np.max(abs(component),initial=0))
                defect=float(np.max(abs(reconstructed-component),initial=0))
                relative=(defect/amplitude if amplitude>0 else 0.)
                diagonal=np.abs(np.diag(triangular))
                ratio=(float(diagonal[rank]/diagonal[0])
                       if rank<len(diagonal) and diagonal[0]>0 else 0.)
                print("COMPONENT_RANK_DIAGNOSTIC",
                      "component",index,"amplitude",amplitude,
                      "max_relative_qr_rank_defect",relative,
                      "next_pivoted_qr_diagonal_ratio",ratio,
                      "requested_rank",rank,
                      flush=True)
        raise

def main():
    from cylinder_gauss2_case import run_gauss2_mvi_case
    parser=argparse.ArgumentParser()
    parser.add_argument("--rank",type=int,choices=(2,4),default=2)
    args=parser.parse_args()
    name=f"MVI{args.rank}_RK_STEP1"
    base=Path(__file__).resolve().parents[1]/"config/verification/kl_cc_rk4_followup"
    gauss2._rank_factors=_audited_rank_factors
    result=run_gauss2_mvi_case(base/(name+".json"),
             output_dir=base/"output"/("DIAGNOSTIC_GAUSS2_"+name))
    print("GAUSS2_PHYSICAL_STEP_ACCEPTED",
          result.accepted_steps,"rank",args.rank,flush=True)

if __name__=="__main__":
    main()
