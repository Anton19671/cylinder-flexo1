"""Execute the public CLI with an already-produced synthetic report; no solve."""
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("change,code",[
    ({},0),({"accepted_steps":0},2),({"accepted_steps":True},2),
    ({"full_nodal_equilibrium_satisfied":True},2),
    ({"solution_status":"FAILED"},2),({"calculation_kind":"DYNAMIC"},2),
    ({"solver_backend":"INDEPENDENT_MKR"},2),
])
def test_cli_retains_accepted_projected_static_quality(monkeypatch,capsys,change,code):
    report=dict(status="EXPERIMENTAL_PROJECTED_STATIONARY_UNCERTIFIED",
                solver_backend="INDEPENDENT_MVI",calculation_kind="STATIC",
                scientific_accuracy_certified=False,accepted_steps=1,
                solution_status="CONVERGED_PROJECTED",
                full_nodal_equilibrium_satisfied=False)
    report.update(change)
    dispatcher=ModuleType("cylinder_final_stage11a_dispatcher")
    dispatcher.run_final_case=lambda path:report
    monkeypatch.setitem(sys.modules,dispatcher.__name__,dispatcher)
    monkeypatch.setattr(sys,"argv",["RUN_FINAL_CASE.py","synthetic.json"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(__file__).resolve().parents[1]/
                          "interface"/"RUN_FINAL_CASE.py"),run_name="__main__")
    assert result.value.code==code
    assert "EXPERIMENTAL_PROJECTED_STATIONARY_UNCERTIFIED" in capsys.readouterr().out
