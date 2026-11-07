"""Method-specific physical loading/trace preflight with NO shell solve."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cylinder_paths import activate
activate()

from cylinder_gauss2_mvi_physics import (
    _stage_loads,_physical_assembled,_require_fixed_supports
)
import cylinder_gauss2_variational_mvi as adapter


def test_gauss_mvi_production_adapter_uses_own_physical_trace_compiler():
    assert adapter._stage_loads is _stage_loads
    assert adapter._physical_assembled is _physical_assembled
    assert adapter._require_fixed_supports is _require_fixed_supports
    assert adapter._stage_loads.__module__=="cylinder_gauss2_mvi_physics"


def test_mvi_loads_allow_electric_on_with_no_imposed_free_charge():
    mechanical=np.ones((6,4,5))
    assert _stage_loads(lambda t:(mechanical,None),.2)[1] is None
    for wrong in ((mechanical,),{"force":mechanical},None):
        with pytest.raises(TypeError):
            _stage_loads(lambda t,result=wrong:result,.2)


def _stub(moving=False,velocity=False):
    mechanical=np.zeros((2,3,4))
    if velocity:
        speed=np.ones_like(mechanical)
    else:
        speed=np.zeros_like(mechanical)
    state=SimpleNamespace(
        mechanical=mechanical,electrical=None,velocity=speed,time_s=0.)
    settings=SimpleNamespace(dt_s=.01)
    row=np.zeros((1,mechanical.size))
    row[0,0]=1.
    class ArtificialBoundary:
        functional=SimpleNamespace(theory="CUF_TE1")
        def __init__(self):
            self.times=[]
        def assemble(self,q,p,**kwargs):
            self.times.append(kwargs["time_s"])
            c=row.copy()
            target=np.array([.01*kwargs["time_s"] if moving else 0.])
            return SimpleNamespace(space=SimpleNamespace(
                matrix=c,target=target))
    return ArtificialBoundary(),state,settings


def test_gauss_mvi_supports_are_checked_at_both_true_collocation_times():
    problem,state,settings=_stub()
    c,d=_require_fixed_supports(
        problem,state,settings,7800.,lambda t:(None,None))
    assert c.shape==(1,state.mechanical.size)
    np.testing.assert_array_equal(d,[0.])
    assert len(problem.times)==4
    roots=((3-np.sqrt(3))/6,(3+np.sqrt(3))/6)
    np.testing.assert_allclose(problem.times,[0.,roots[0]*.01,
                                              roots[1]*.01,.01])
    problem,state,settings=_stub(moving=True)
    with pytest.raises(ValueError,match="MOVING_MECHANICAL_SUPPORT"):
        _require_fixed_supports(
            problem,state,settings,7800.,lambda t:(None,None))
    problem,state,settings=_stub(velocity=True)
    with pytest.raises(ValueError,match="INITIAL_SUPPORT_INCONSISTENT"):
        _require_fixed_supports(
            problem,state,settings,7800.,lambda t:(None,None))


def test_gauss_mvi_stage_assembly_touches_physical_problem_not_fdm_solver():
    mechanical=np.zeros((2,3,4))
    class Store:
        functional=SimpleNamespace(theory="CUF_TE1")
        called=None
        def assemble(self,q,p,**kwargs):
            self.called=kwargs
            return "ASSEMBLED_PHYSICAL_BOUNDARY"
    p=Store()
    got=_physical_assembled(
        p,mechanical,None,time_s=.3,density_kg_m3=None,
        loads_at=lambda t:(mechanical,None))
    assert got=="ASSEMBLED_PHYSICAL_BOUNDARY"
    assert p.called["dead_load"] is mechanical
    assert p.called["free_charge"] is None
    assert p.called["time_s"]==.3
    assert "acceleration" not in p.called
