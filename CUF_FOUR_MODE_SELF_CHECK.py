#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUF-TE1 Linear/Nonlinear x Elastic/Birger control run."""
from __future__ import annotations
import shutil
import FINAL_SELF_CHECK as f

if __name__=="__main__":
    if f.WORK.exists():
        shutil.rmtree(f.WORK)
    f.WORK.mkdir(parents=True)
    elastic=[]
    for mode in ("LINEAR_ELASTIC","NONLINEAR_ELASTIC"):
        w,r,it=f._run_nonzero_fdm("CUF_TE1",mode)
        elastic.append((mode,w,r,it))
    birger=f._test_cuf_four_modes()
    print("CUF_FOUR_MODE_SELF_CHECK: PASS")
    for mode,w,r,it in elastic:
        print(f"  {mode}: wmax={w:.17g}, residual={r:.3e}, iterations={it}")
    for mode,w,outer in birger:
        print(f"  {mode}: wmax={w:.17g}, outer={outer}")
