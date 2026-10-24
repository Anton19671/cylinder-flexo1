#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedicated heavy K2 verification: dense reference vs safe coloring."""
from __future__ import annotations
import json
import FINAL_SELF_CHECK as f

if __name__=="__main__":
    report=f._test_k2_safe_coloring()
    print("K2_SELF_CHECK: PASS")
    print(
        "  LINEAR: residual evaluations "
        f"{report['linear_stats']['perturbed_residual_evaluations']}/"
        f"{report['linear_stats']['n_columns']}; "
        f"time K2={report['linear_k2_seconds']:.6f}s, "
        f"dense={report['linear_dense_seconds']:.6f}s"
    )
    print(
        "  NONLINEAR J: residual evaluations "
        f"{report['nonlinear_stats']['perturbed_residual_evaluations']}/"
        f"{report['nonlinear_stats']['n_columns']}; "
        f"time K2={report['nonlinear_k2_seconds']:.6f}s, "
        f"dense={report['nonlinear_dense_seconds']:.6f}s"
    )
    e=report['te1_15x15']
    print(
        "  TE1 15x15 coloring count: "
        f"{e['k2_perturbed_evaluations']}/{e['dense_perturbed_evaluations']} "
        f"({e['reduction_factor']:.6f}x fewer perturbed evaluations)"
    )
    print(f"  TE1 dense/K2 q and wmax identical; wmax={report['te1_wmax']:.17g}")
    print("K2_REPORT_JSON="+json.dumps(report,sort_keys=True))
