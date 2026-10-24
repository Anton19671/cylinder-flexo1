#!/usr/bin/env python3
"""Lightweight checks for hard Birger step-admissibility criteria."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
import numpy as np


progress=types.ModuleType("final_progress")
progress.emit=lambda *args,**kwargs: None
sys.modules.setdefault("final_progress",progress)

stage2=types.ModuleType("cylinder_te1_stage2b")
class InvalidDeformationMapping(RuntimeError):
    pass
stage2.InvalidDeformationMapping=InvalidDeformationMapping
sys.modules.setdefault("cylinder_te1_stage2b",stage2)
for name in (
    "cylinder_te1_local_kernel",
    "cylinder_te1_stage3a_connector",
    "cylinder_te1_stage3c_internal_mvi",
):
    sys.modules.setdefault(name,types.ModuleType(name))

import cylinder_te1_stage5a_birger as b5


def settings(**changes):
    values=dict(
        epsilon_tol=1.0e-14,omega=1.0,G_ref=1.0,sigma_ref=1.0,
        tol_G=1.0e-10,tol_sigma=1.0e-10,tol_u=1.0e-10,
        max_outer_iterations=3,epsilon_i_limit=0.02,delta_J_limit=0.05,
        reject_local_unloading=True,unloading_tolerance=1.0e-12,
    )
    values.update(changes)
    return b5.BirgerSettings(**values)


def mat(eps,det):
    e=np.asarray(eps,float)
    return b5.MaterialPointState(e,3.0*e,np.asarray(det,float))


def expect_code(code,fn):
    try:
        fn()
    except b5.BirgerAdmissibilityError as exc:
        assert exc.code==code,(exc.code,code)
    else:
        raise AssertionError(f"expected {code}")


def main():
    ok=mat([[0.01,0.015]],[[1.01,0.98]])
    d=b5.require_material_admissible(ok,settings())
    assert d["critical_epsilon_i_index"]==(0,1)
    assert d["critical_J_index"]==(0,1)

    expect_code(
        "BIRGER_EPSILON_I_LIMIT_EXCEEDED",
        lambda:b5.require_material_admissible(
            mat([[0.01,0.021]],[[1.0,1.0]]),settings()
        ),
    )
    expect_code(
        "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
        lambda:b5.require_material_admissible(
            mat([[0.01,0.01]],[[1.0,1.051]]),settings()
        ),
    )
    prev=np.array([[0.01,0.01]])
    expect_code(
        "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
        lambda:b5.require_material_admissible(
            mat([[0.01,0.009]],[[1.0,1.0]]),settings(),prev
        ),
    )
    # A decrement smaller than the declared numerical tolerance is round-off,
    # not physical unloading.
    b5.require_material_admissible(
        mat([[0.01,0.01-0.5e-12]],[[1.0,1.0]]),settings(),prev
    )

    @dataclass
    class InnerResult:
        state: np.ndarray
        converged: bool=True
        status: str="OK"

    excessive=mat([[0.03]],[[1.0]])
    result=b5.solve_birger_outer_cycle(
        np.zeros(1),np.ones((1,1)),G0=1.0,
        material_diagram=lambda e:3.0*np.asarray(e,float),
        inner_solve=lambda state,GB:InnerResult(np.asarray(state,float)),
        evaluate_material=lambda state,GB:excessive,
        state_vector=lambda state:np.asarray(state,float),
        settings=settings(),
    )
    assert not result.converged
    assert result.status.startswith("BIRGER_EPSILON_I_LIMIT_EXCEEDED")
    assert result.epsilon_limit_exceeded
    print("BIRGER_ADMISSIBILITY_SELF_CHECK: PASS")


if __name__=="__main__":
    main()
