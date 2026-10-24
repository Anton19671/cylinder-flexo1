#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensure a load-front residual can never be hidden from convergence."""
import numpy as np

import final_residual_regions as rr


def main():
    masks=rr.ResidualRegionMasks(
        interior=np.array([True,False,False]),
        boundary=np.array([False,True,False]),
        load_front=np.array([False,False,True]),
        # Legacy callers may still pass True; metrics must ignore it safely.
        exclude_load_front_from_convergence=True,
    )
    residual=np.array([1e-8,2e-8,4.0])
    metrics=rr.metrics(residual,masks,scale=2.0)
    assert metrics.residual_inf_load_front==4.0
    assert metrics.residual_convergence_inf==4.0
    assert metrics.residual_convergence_relative==2.0
    assert not rr.converged(metrics,1e-4)

    resolved=rr.resolve_front_exclusion(masks,np.array([0.0,0.0,1.0]))
    assert not resolved.exclude_load_front_from_convergence
    print("RESIDUAL_REGION_CONVERGENCE_SELF_CHECK: PASS")


if __name__=="__main__":
    main()
