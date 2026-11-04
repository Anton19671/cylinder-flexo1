"""Forty-two separated solver/theory/integrator CASEs, validation only.

No forced physical simulation. This checks that every combination has
a real independent entrypoint with own input and output, no cross dispatch.
"""
import ast
from pathlib import Path
import pytest

from independent_mkr.run_models import load_case as mkr_load
from independent_mkr.run_models import run as mkr_run
from independent_mvi.run_models import load_case as mvi_load
from independent_mvi.run_models import run as mvi_run
from independent_mkr.linear_models import MODELS as MKR_MODELS
from independent_mvi.linear_models import MODELS as MVI_MODELS

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config/verification/independent_all_models"
METHODS=("MKR","MVI")
TIMES=("BATHE","NEWMARK","GAUSS_IRK4")


def test_exactly_42_disjoint_case_files():
    found={p.name for p in CASES.glob("*.json")}
    expected={f"{method}_{model}_{time}.json"
              for method in METHODS for model in MKR_MODELS for time in TIMES}
    assert set(MKR_MODELS)==set(MVI_MODELS)
    assert found==expected
    assert len(found)==42


@pytest.mark.parametrize("method",METHODS)
@pytest.mark.parametrize("model",MKR_MODELS)
@pytest.mark.parametrize("integrator",TIMES)
def test_independent_case_route_without_physics(method,model,integrator,tmp_path):
    filename=CASES/f"{method}_{model}_{integrator}.json"
    import json
    data=json.loads(filename.read_text(encoding="utf-8"))
    output=tmp_path/method/model/integrator
    data["output_dir"]=str(output)
    revised=tmp_path/f"{method}_{model}_{integrator}.json"
    revised.write_text(json.dumps(data),encoding="utf-8")
    parser,runner=(mkr_load,mkr_run) if method=="MKR" else (
        mvi_load,mvi_run)
    other=mvi_load if method=="MKR" else mkr_load
    with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
        other(revised)
    result=runner(revised,validate_only=True)
    assert result["validated_only"] is True
    assert result["integrator"]==integrator
    assert result["physics"]["theory"]==model
    assert result["steps"]==1
    assert result["dt_s"]==.002
    assert not output.exists()


@pytest.mark.parametrize("directory",("independent_mkr","independent_mvi"))
def test_seven_model_runners_import_only_own_method(directory):
    path=ROOT/directory/"run_models.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    forbidden=("independent_mvi" if directory=="independent_mkr" else
               "independent_mkr")
    assert not any(name.startswith((forbidden,"cylinder_","cuf_g3d"))
                   for name in imports)
    for method in ("step_newmark","step_bathe","step_gauss_irk4"):
        implemented=(
            ROOT/directory/"linear_models.py").read_text(encoding="utf-8")
        assert f"    def {method}(" in implemented
