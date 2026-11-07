"""Headless static GUI contracts: 63 native CASEs; never starts a solver."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import pytest

from independent_mkr import gui as dyn_mkr,static_gui as sta_mkr
from independent_mvi import gui as dyn_mvi,static_gui as sta_mvi

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config"/"verification"/"independent_static"


@pytest.mark.parametrize("method,ui",(("MKR",sta_mkr),("MVI",sta_mvi)))
@pytest.mark.parametrize("theory",("KL","FSDT","REDDY",
    "CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("rank",(1,2,3,4,5,6,7,8))
def test_all_static_gui_passports_preflight_without_computation(
        method,ui,theory,rank):
    if method=="MKR" and rank!=1:
        pytest.skip("No rank for MKR")
    stem=f"MKR_{theory}" if method=="MKR" else f"MVI_R{rank}_{theory}"
    doc=json.loads((CASES/f"{stem}.json").read_text())
    result=ui.inspect_static_document(doc)
    assert result.theory==theory
    assert doc["schema"]==ui.OWN_SCHEMA
    assert "time" not in doc and "dynamics" not in doc


@pytest.mark.parametrize("ui", (sta_mkr,sta_mvi))
def test_static_gui_refuses_dynamic_and_foreign_physics(ui):
    target="MKR_KL.json" if ui is sta_mkr else "MVI_R1_KL.json"
    valid=json.loads((CASES/target).read_text())
    invalid=json.loads(json.dumps(valid))
    invalid["time"]={"integrator":"NEWMARK","steps":1,"dt_s":.002}
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        ui.inspect_static_document(invalid)
    invalid=json.loads(json.dumps(valid))
    invalid["physics"]["gradient_model"]="G3D"
    with pytest.raises(ValueError,match="BAD_PHYSICS_KEYS"):
        ui.inspect_static_document(invalid)
    invalid=json.loads(json.dumps(valid))
    invalid["schema"]="INDEPENDENT_OTHER_STATIC_SEVEN_MODELS_V1"
    with pytest.raises(ValueError,match="FOREIGN_SOLVER_SCHEMA"):
        ui.inspect_static_document(invalid)


@pytest.mark.parametrize("method,stat,dynamic",(
    ("MKR",sta_mkr,dyn_mkr),("MVI",sta_mvi,dyn_mvi)))
def test_both_static_windows_use_own_process_and_are_reachable(method,stat,dynamic):
    stat_source=Path(stat.__file__).read_text(encoding="utf-8")
    dyn_source=Path(dynamic.__file__).read_text(encoding="utf-8")
    tree=ast.parse(stat_source)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef)
             and node.name=="IndependentStaticWindow")
    actions={node.name for node in cls.body if isinstance(node,ast.FunctionDef)}
    assert {"template","load","validate","save","run","stop",
            "view_results","open_output","_poll","_close"}<=actions
    assert "subprocess.Popen(command" in stat_source
    assert f'independent_{method.lower()}.run_static' in stat_source
    assert '"--run"' in stat_source
    assert '"--validate-only"' not in stat_source
    other="independent_mvi" if method=="MKR" else "independent_mkr"
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not (node.module or "").startswith(other)
        elif isinstance(node,ast.Import):
            assert all(not item.name.startswith(other) for item in node.names)
    assert 'text="СТАТИКА"' in dyn_source
    assert f"independent_{method.lower()}.static_gui" in dyn_source
    assert "self.proc.terminate()" in stat_source


def test_own_static_templates_separate_mvi_rank_output_paths():
    one=json.loads((CASES/"MVI_R1_CUF_TE4.json").read_text())
    eight=json.loads((CASES/"MVI_R8_CUF_TE4.json").read_text())
    assert one["output_dir"]!=eight["output_dir"]
    assert sta_mvi._template_path("CUF_TE4",8)==(
        CASES/"MVI_R8_CUF_TE4.json")
    assert sta_mkr._template_path("CUF_TE4")==(
        CASES/"MKR_CUF_TE4.json")
