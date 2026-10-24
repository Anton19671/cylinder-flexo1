#!/usr/bin/env python3
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

import cylinder_cuf_stage7d_general_internal_mvi as mvi


class ScalarState:
    def __init__(self, value=1.0):
        self.value=float(value)

    def copy(self):
        return ScalarState(self.value)


@contextmanager
def patched(items):
    old=[(obj,name,getattr(obj,name)) for obj,name,_new in items]
    try:
        for obj,name,new in items:
            setattr(obj,name,new)
        yield
    finally:
        for obj,name,value in reversed(old):
            setattr(obj,name,value)


def run_checks(verbose=True):
    # R(c)=T*c+b, T^T*b=0.  The original reduced equation is therefore c=0.
    # Left-scaling the full residual before projection produces a different
    # quantity which is nonzero at c=0.  A half-step must stop on the equation
    # it solved; the scaled measure is retained for the completed XY-cycle.
    Tn=np.array([[1.0],[1.0]])/np.sqrt(2.0)
    b=np.array([-1.0,1.0])
    ids=np.array([0]); mask=np.array([True]); norms=np.array([1.0]); active=np.array([True])

    def residual(_fn,state,_grid):
        return Tn[:,0]*state.value+b

    def direction_data(*_args,**_kwargs):
        return mask.copy(),Tn.copy(),ids.copy(),norms.copy(),active.copy(),Tn.copy()

    def tangent(*_args,**_kwargs):
        return Tn.copy(),Tn.copy(),mask.copy(),ids.copy(),norms.copy(),active.copy()

    def get_parameters(state,_direction,_mask):
        return np.array([state.value]),ids.copy()

    def set_parameters(state,_direction,_ids,values):
        state.value=float(np.asarray(values)[0])

    settings=mvi.GeneralInternalMVISettings(
        projected_tolerance=1e-12,increment_tolerance=1e-12,
        tangent_fd_step=1e-7,max_halfstep_iterations=3,max_xy_cycles=2,
    )
    state=ScalarState(1.0)
    multipliers=np.array([1.0,2.0])
    with patched([
        (mvi,"_residual_vector",residual),
        (mvi,"_direction_data",direction_data),
        (mvi,"directional_tangent_action_fd",tangent),
        (mvi.g8,"n_full_dofs",lambda _state,_grid:2),
        (mvi.g8,"residual_projection_weights",lambda _state,_grid:np.ones(2)),
        (mvi.g8,"get_active_parameters",get_parameters),
        (mvi.g8,"set_active_parameters",set_parameters),
    ]):
        result=mvi.solve_strict_halfstep(
            state,None,None,"X",None,settings,weights=np.ones(2),
            measure_multipliers=multipliers,quadrature=None,cycle=1,
            projected_scale=1.0,
        )
        raw_inf=mvi.projected_inf_for_direction(
            state,None,None,"X",None,np.ones(2),measure_multipliers=None
        )
        scaled_inf=mvi.projected_inf_for_direction(
            state,None,None,"X",None,np.ones(2),measure_multipliers=multipliers
        )

    if result.status!="CONVERGED_REDUCED_MVI" or result.iterations!=1:
        raise AssertionError("half-step did not converge on its original equation")
    if raw_inf>1e-12:
        raise AssertionError("original reduced equation was not solved")
    if scaled_inf<=1e-3:
        raise AssertionError("test did not expose the distinct V-4 cycle measure")

    if verbose:
        print("MVI_HALFSTEP_RESIDUAL_ALIGNMENT_SELF_CHECK_V11: PASS")
        print(f"  original reduced residual after correction: {raw_inf:.6e}")
        print(f"  distinct V-4 completed-cycle measure: {scaled_inf:.6e}")
    return {"status":"PASS","raw_inf":raw_inf,"v4_measure_inf":scaled_inf}


if __name__=="__main__":
    run_checks(True)
