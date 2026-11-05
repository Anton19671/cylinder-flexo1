"""No-display, no-time-marching GUI routing and CASE validation.

Test independent GUI imports without creating Tk, starting subprocesses or
running a physical cylinder calculation. Each method owns all computation.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from independent_mkr.gui import inspect_document as inspect_mkr
from independent_mvi.gui import inspect_document as inspect_mvi
from independent_mkr.run_models import SCHEMA as MKR_SCHEMA
from independent_mvi.run_models import SCHEMA as MVI_SCHEMA

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config"/"verification"/"independent_all_models"


@pytest.mark.parametrize("theory",("KL","FSDT","REDDY",
    "CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
@pytest.mark.parametrize("integrator",("BATHE","NEWMARK","GAUSS_IRK4"))
def test_both_gui_editors_preflight_their_own_case_only(theory,integrator):
    mkr=json.loads((CASES/f"MKR_{theory}_{integrator}.json").read_text())
    mvi=json.loads((CASES/f"MVI_{theory}_{integrator}.json").read_text())
    assert mkr["schema"]==MKR_SCHEMA
    assert mvi["schema"]==MVI_SCHEMA
    assert inspect_mkr(mkr).theory==theory
    assert inspect_mvi(mvi).theory==theory
    with pytest.raises(ValueError,match="CASE"):
        inspect_mkr(mvi)
    with pytest.raises(ValueError,match="CASE"):
        inspect_mvi(mkr)


@pytest.mark.parametrize("method",("mkr","mvi"))
def test_gui_method_isolation_by_ast(method):
    source=ROOT/f"independent_{method}"/"gui.py"
    tree=ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append(node.module or "")
    other="independent_mvi" if method=="mkr" else "independent_mkr"
    assert not any(x.startswith((other,"cylinder_","mkr.","mvi.")) for x in imports)
    text=source.read_text(encoding="utf-8")
    assert '"--run"' in text
    assert '"--case"' in text
    assert "subprocess.Popen(" in text
    assert "self.proc.terminate()" in text
    assert "messagebox.showerror" in text


def test_final_gui_exposes_both_independent_windows_without_rerouting():
    source=ROOT/"interface"/"FINAL_GUI.py"
    tree=ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and
             x.name=="FinalSolverGUI")
    own={x.name:x for x in cls.body if isinstance(x,ast.FunctionDef)}
    assert "open_independent_mkr" in own and "open_independent_mvi" in own
    mkr=ast.unparse(own["open_independent_mkr"])
    mvi=ast.unparse(own["open_independent_mvi"])
    assert "independent_mkr.gui" in mkr
    assert "independent_mvi.gui" in mvi
    assert "independent_mvi." not in mkr
    assert "independent_mkr." not in mvi
    last_build=[x for x in cls.body if isinstance(x,ast.FunctionDef)
                and x.name=="_build_ui"][-1]
    gui_source=ast.unparse(last_build)
    assert "open_independent_mkr" in gui_source
    assert "open_independent_mvi" in gui_source
    assert "self.run_case" in gui_source
    # Legacy calculation controls remain visible and untouched; independent
    # solvers must only run through their explicit own GUI buttons.
    assert "independent_mkr_btn" in gui_source
    assert "independent_mvi_btn" in gui_source


def test_invalid_nonlinear_or_electrical_case_is_never_accepted():
    document=json.loads((CASES/"MVI_CUF_TE4_NEWMARK.json").read_text())
    for field,value in (
        ("geometry_model","GREEN_LAGRANGE"),
        ("scale_model","MICRO"),
        ("enable_flexoelectric",True),
        ("plasticity","J2"),
    ):
        modified=json.loads(json.dumps(document))
        modified["physics"][field]=value
        with pytest.raises(TypeError):
            inspect_mvi(modified)
    modified=json.loads(json.dumps(document))
    modified["rank"]=99
    with pytest.raises(ValueError,match="ранг"):
        inspect_mvi(modified)
