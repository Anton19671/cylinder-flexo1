"""Native strong stationary-boundary compatibility: algebraic tests only.

The helper checks physical outward normal derivatives of each prescribed
essential axial trace. No loaded shell solution is started.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_native_mkr_initial import _verify_fixed_initial_supports


def engine_with_slope_support():
    grid=DirectStrongGrid(11,8,2.,1.)
    ends={
        "left":[[{"kind":"ESSENTIAL","value":0.},
                 {"kind":"ESSENTIAL","value":-2.}]],
        "right":[[{"kind":"ESSENTIAL","value":4.},
                  {"kind":"ESSENTIAL","value":2.}]],
    }
    return SimpleNamespace(model=SimpleNamespace(grid=grid),
                           mechanical_ends=ends)


def test_initial_support_outward_normal_sign_at_both_ends():
    engine=engine_with_slope_support()
    grid=engine.model.grid
    x=np.arange(grid.nx)*grid.dx
    q=np.broadcast_to(2*x[None,:,None],
                      (1,grid.nx,grid.ns)).copy()
    v=np.zeros_like(q)
    a=np.zeros_like(q)
    _verify_fixed_initial_supports(engine,q,v,acceleration=a,
                                   absolute_tolerance=1.e-9)


@pytest.mark.parametrize("kind",("displacement","velocity","acceleration"))
def test_initial_second_order_support_checks_every_prescribed_trace(kind):
    engine=engine_with_slope_support()
    grid=engine.model.grid
    x=np.arange(grid.nx)*grid.dx
    q=np.broadcast_to(2*x[None,:,None],
                      (1,grid.nx,grid.ns)).copy()
    v=np.zeros_like(q)
    a=np.zeros_like(q)
    if kind=="displacement":
        q[0,1,:]+=.01
    elif kind=="velocity":
        v[0,1,:]+=.01
    else:
        a[0,1,:]+=.01
    expected={
        "displacement":"INITIAL_DISPLACEMENT_ESSENTIAL_TRACE_MISMATCH",
        "velocity":"INITIAL_MOVING_SUPPORT_NOT_SUPPORTED",
        "acceleration":"INITIAL_ACCELERATION_ESSENTIAL_TRACE_MISMATCH",
    }[kind]
    with pytest.raises(ValueError,match=expected):
        _verify_fixed_initial_supports(engine,q,v,acceleration=a,
                                       absolute_tolerance=1.e-9)


def test_acceleration_trace_is_optional_only_for_initialization():
    engine=engine_with_slope_support()
    grid=engine.model.grid
    x=np.arange(grid.nx)*grid.dx
    q=np.broadcast_to(2*x[None,:,None],
                      (1,grid.nx,grid.ns)).copy()
    v=np.zeros_like(q)
    a=np.ones_like(q)
    _verify_fixed_initial_supports(engine,q,v,absolute_tolerance=1.e-9)
    with pytest.raises(ValueError,match="INITIAL_ACCELERATION"):
        _verify_fixed_initial_supports(engine,q,v,acceleration=a,
                                       absolute_tolerance=1.e-9)
