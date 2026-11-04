"""Six independent temporal routes: algebra and CASE preflight only.

No physical shell loading/time-marching simulation is executed. Fake step
solvers are used for Newmark/Bathe kinematics; zero-source manufactured
Gauss stages verify the rank/factor and endpoint guards.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from independent_mkr.linear_te4 import PhysicalCase,StrongDifferenceTE4
from independent_mvi.linear_te4 import VariationalCase,VariationalTE4
from independent_mkr.run_case import load_case as mkr_load,run as mkr_run
from independent_mvi.run_case import load_case as mvi_load,run as mvi_run

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"config/verification/independent_te4"


@pytest.mark.parametrize("scheme",["BATHE","NEWMARK","GAUSS_IRK4"])
def test_independent_case_files_preflight_only(scheme,tmp_path):
    for method,loader,runner in (
        ("MKR",mkr_load,mkr_run),("MVI",mvi_load,mvi_run)):
        case=CASES/f"{method}_{scheme}.json"
        config=json.loads(case.read_text())
        config["output_dir"]=str(tmp_path/method/scheme)
        temp=tmp_path/f"{method}_{scheme}.json"
        temp.write_text(json.dumps(config),encoding="utf-8")
        result=runner(temp,validate_only=True)
        assert result["validated_only"] is True
        assert result["integrator"]==scheme
        assert result["steps"]==1
        assert not (tmp_path/method/scheme).exists()
        assert loader(temp)[0].nx==11


def test_cross_method_cases_are_not_accepted(tmp_path):
    for source,wrong_loader in (
        (CASES/"MKR_NEWMARK.json",mvi_load),
        (CASES/"MVI_NEWMARK.json",mkr_load)):
        with pytest.raises(ValueError,match="CASE_KEYS_INVALID"):
            wrong_loader(source)


def test_each_runner_is_exclusive_and_imports_no_other_solver_or_dispatcher():
    forbidden={
        "independent_mkr":("independent_mvi","cylinder_variational",
                           "cylinder_bathe_mvi","cylinder_reference_case"),
        "independent_mvi":("independent_mkr","cylinder_variational",
                           "cylinder_bathe_mvi","cylinder_reference_case"),
    }
    for directory,names in forbidden.items():
        for filename in ("linear_te4.py","run_case.py"):
            p=ROOT/directory/filename
            tree=ast.parse(p.read_text(encoding="utf-8"))
            imports=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node,ast.ImportFrom):
                    imports.append(node.module or "")
            assert not any(mod.startswith(names) for mod in imports),(p,imports)


def test_independent_newmark_predictors_are_second_order_and_complete():
    fd=StrongDifferenceTE4(PhysicalCase())
    mvi=VariationalTE4(VariationalCase(),rank=1)
    shape=(15,9,9)
    q=np.zeros(shape);v=np.zeros(shape);a=np.zeros(shape)
    q[:,1:-1,:]=.001
    v[:,1:-1,:]=.002
    a[:,1:-1,:]=.003
    dt=.002
    fd_record=[]
    def fd_fake(pred,t,coefficient):
        fd_record.append((pred.copy(),t,coefficient))
        return pred.copy()
    with patch.object(fd,"_solve_endpoint",side_effect=fd_fake):
        u,w,z=fd.step_newmark(q,v,a,.01,dt)
    expected=q+dt*v+dt*dt*.25*a
    np.testing.assert_allclose(fd_record[0][0],expected)
    np.testing.assert_allclose(u,expected)
    np.testing.assert_allclose(z,0.,atol=2.e-14)
    np.testing.assert_allclose(w,v+dt*.5*a)

    mvi_record=[]
    def mvi_fake(pred,t,coef,previous=None,max_sweeps=35):
        mvi_record.append((pred.copy(),t,coef))
        return pred.copy(),mvi._initial(),((1,"X",0),(1,"Y",0))
    with patch.object(mvi,"_linear_endpoint",side_effect=mvi_fake):
        result,profiles,history=mvi.step_newmark(q,v,a,.01,dt)
    np.testing.assert_allclose(mvi_record[0][0],expected)
    np.testing.assert_allclose(result[0],expected)
    np.testing.assert_allclose(result[1],v+dt*.5*a)
    np.testing.assert_allclose(result[2],0.,atol=2.e-14)
    assert len(history)==2


def test_mvi_gauss_endpoint_refuses_genuine_rank_two_on_rank_one_case():
    mvi=VariationalTE4(VariationalCase(),rank=1)
    x=np.linspace(0,1,9)
    theta=2*np.pi*np.arange(9)/9
    q=np.zeros((15,9,9))
    q[10]=np.outer(np.sin(np.pi*x),np.cos(theta))+(
        .35*np.outer(np.sin(2*np.pi*x),np.sin(theta)))
    with pytest.raises(RuntimeError,match="MVI_IRK4_ENDPOINT_EXCEEDS_FIXED_RANK"):
        mvi._recover_rank_factors(q)
    v=VariationalTE4(VariationalCase(),rank=2)
    xfac,yfac=v._recover_rank_factors(q)
    np.testing.assert_allclose(v._field(xfac,yfac),q,atol=1.e-13)


def test_zero_manufactured_gauss4_on_both_independent_routes():
    fd=StrongDifferenceTE4(PhysicalCase())
    vi=VariationalTE4(VariationalCase(),rank=1)
    shape=(15,9,9)
    q=np.zeros(shape)
    with patch.object(fd,"force",np.zeros(shape)):
        stages,end=fd.step_runge_kutta_gauss4(q,q,q,0.,dt_s=.002)
    assert len(stages)==2
    for vec in (*stages,*[end]):
        for field in vec:
            np.testing.assert_allclose(field,0.,atol=1.e-11)

    with patch.object(vi,"pressure",np.zeros(shape)):
        stages,end,fac,history=vi.step_runge_kutta_gauss4(
            q,q,q,0.,dt_s=.002)
    assert len(stages)==2
    for vec in (*stages,*[end]):
        for field in vec:
            np.testing.assert_allclose(field,0.,atol=1.e-11)
    assert fac[0].shape==(15,1,9)
    assert fac[1].shape==(15,1,9)
