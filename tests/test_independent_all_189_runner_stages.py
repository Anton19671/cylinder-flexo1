"""Method-owned runner smoke with mocked stage operators only.

No physical shell solve or time integration is performed: fake solver
instances return prescribed rank-one fields. This exercises output
wiring for every one of 189 theory-method-integrator-rank CASE routes.
"""
from __future__ import annotations
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from cylinder_paths import activate
activate()
import independent_mkr.run_models as strong_runner
import independent_mvi.run_models as weak_runner
from scientific_complete_method_matrix import independent_catalog,independent_template


@pytest.mark.parametrize("row",independent_catalog(),ids=lambda x:x["id"])
def test_all_189_runner_stage_contracts_without_numerical_solver(
        row,tmp_path,monkeypatch):
    module=(strong_runner if row["method"]=="MKR" else weak_runner)
    item=independent_template(row["theory"],row["integrator"],row["method"])
    if row["method"]=="MVI":
        item["rank"]=row["rank"]
    result_dir=tmp_path/"method_owned_result"
    item["output_dir"]=str(result_dir)
    case_file=tmp_path/"method_owned_case.json"
    case_file.write_text(json.dumps(item),encoding="utf-8")
    calls=[]
    class ManufacturedStep:
        def __init__(self,physics,*args):
            self.physics=physics
            name=physics.theory
            self.n=(3 if name=="KL" else 5 if name in ("FSDT","REDDY")
                    else 3*(int(name[-1])+1))
            self.rank=row["rank"] if row["method"]=="MVI" else None
        def _make_stage(self,amplitude):
            c=self.physics
            shape=(self.n,c.nx,c.ns)
            q=np.zeros(shape)
            v=np.zeros(shape)
            a=np.zeros(shape)
            radial=(2 if c.theory in ("KL","FSDT","REDDY") else
                    2*(int(c.theory[-1])+1))
            q[radial,c.nx//2,c.ns//2]=amplitude
            v[radial,c.nx//2,c.ns//2]=amplitude*.1
            return q,v,a
        def _factors(self,amplitude):
            c=self.physics
            X=np.zeros((self.n,self.rank,c.nx))
            Y=np.zeros((self.n,self.rank,c.ns))
            radial=(2 if c.theory in ("KL","FSDT","REDDY") else
                    2*(int(c.theory[-1])+1))
            X[radial,0,c.nx//2]=amplitude
            Y[radial,0,c.ns//2]=1.
            return X,Y
        def step_newmark(self,q,v,a,t,dt_s,**kwargs):
            calls.append(("NEWMARK",t,dt_s))
            result=self._make_stage(1.e-6)
            if self.rank is not None:
                return result,self._factors(1.e-6),()
            return result
        def step_bathe(self,q,v,a,t,dt_s,**kwargs):
            calls.append(("BATHE",t,dt_s))
            first=self._make_stage(1.e-6)
            final=self._make_stage(2.e-6)
            if self.rank is not None:
                return first,final,self._factors(2.e-6),()
            return first,final
        def step_gauss_irk4(self,q,v,a,t,dt_s,**kwargs):
            calls.append(("GAUSS_IRK4",t,dt_s))
            stage1=self._make_stage(.5e-6)
            stage2=self._make_stage(1.5e-6)
            final=self._make_stage(2.e-6)
            if self.rank is not None:
                return (stage1,stage2),final,self._factors(2.e-6),()
            return (stage1,stage2),final

    monkeypatch.setattr(module,("IndependentStrongModel"
                          if row["method"]=="MKR"
                          else "IndependentVariationalModel"),ManufacturedStep)
    accepted=module.run(case_file,validate_only=False)
    assert accepted["status"]=="CONVERGED"
    assert accepted["accepted_steps"]==accepted["requested_steps"]==1
    assert accepted["theory"]==row["theory"]
    assert accepted["integrator"]==row["integrator"]
    assert accepted["solver_module"]==(
        "independent_mkr.linear_models" if row["method"]=="MKR" else
        "independent_mvi.linear_models")
    assert len(calls)==1
    assert calls[0][0]==row["integrator"]
    dt=item["time"]["dt_s"]
    assert calls[0][1]==0.
    assert calls[0][2]==dt
    names={
        "NEWMARK":("END",),
        "BATHE":("GAMMA","END"),
        "GAUSS_IRK4":("GAUSS_1","GAUSS_2","END"),
    }[row["integrator"]]
    with (result_dir/f"{row['method']}_OWN_PHYSICAL_HISTORY.csv").open(
        newline="",encoding="utf-8") as handle:
        history=list(csv.DictReader(handle))
    assert [item["stage"] for item in history]==list(names)
    expected_times=(dt,) if len(names)==1 else (
        ((2-sqrt(2))*dt,dt) if len(names)==2
        else ((.5-sqrt(3)/6)*dt,(.5+sqrt(3)/6)*dt,dt))
    np.testing.assert_allclose([float(x["time_s"]) for x in history],
                               expected_times,rtol=0,atol=1.e-12)
    suffix=("MKR_STEP_0001_" if row["method"]=="MKR"
            else f"MVI_R{row['rank']}_STEP_0001_")
    for stage,time in zip(names,expected_times):
        file=result_dir/(suffix+stage+".npz")
        assert file.is_file()
        with np.load(file,allow_pickle=False) as data:
            assert data["mechanical"].shape==(
                3 if row["theory"]=="KL" else
                5 if row["theory"] in ("FSDT","REDDY")
                else 3*(int(row["theory"][-1])+1),
                item["physics"]["nx"],item["physics"]["ns"])
            np.testing.assert_allclose(data["time_s"],time,rtol=0,atol=1.e-12)
            assert np.isfinite(data["mechanical"]).all()
            assert np.linalg.norm(data["mechanical"])>0
    if row["method"]=="MVI":
        factor=result_dir/(suffix+"FACTORS.npz")
        assert factor.is_file()
        with np.load(factor,allow_pickle=False) as f, np.load(
                result_dir/(suffix+"END.npz"),allow_pickle=False) as endpoint:
            np.testing.assert_allclose(
                np.einsum("krx,krs->kxs",
                          f["x_profiles"],f["y_profiles"]),
                endpoint["mechanical"],rtol=0,atol=0.)
    else:
        assert not list(result_dir.glob("MVI_R*_STEP_*"))
    assert (result_dir/f"{row['method']}_OWN_CASE.json").is_file()


def test_mvi_rejects_false_rank_factors_before_saving_any_stage(
        tmp_path,monkeypatch):
    module=weak_runner
    item=independent_template("CUF_TE1","NEWMARK","MVI")
    item["rank"]=1
    folder=tmp_path/"runner_result"
    item["output_dir"]=str(folder)
    case_file=tmp_path/"source.json"
    case_file.write_text(json.dumps(item),encoding="utf-8")
    class InvalidFactorSolver:
        def __init__(self,physics,rank):
            self.physics=physics
            self.n=6
        def step_newmark(self,q,v,a,t,dt_s,**kwargs):
            bad=q.copy()
            bad[4,5,3]=1.e-9
            X=np.zeros((6,1,self.physics.nx))
            Y=np.zeros((6,1,self.physics.ns))
            return (bad,v.copy(),a.copy()),(X,Y),()
    monkeypatch.setattr(module,"IndependentVariationalModel",
                        InvalidFactorSolver)
    with pytest.raises(RuntimeError,match="FACTORS_RECONSTRUCTION_MISMATCH"):
        module.run(case_file,validate_only=False)
    summary=json.loads(
        (folder/"MVI_OWN_RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"]=="FAILED"
    assert summary["accepted_steps"]==0
    assert not list(folder.glob("*.npz"))
    with (folder/"MVI_OWN_PHYSICAL_HISTORY.csv").open(
            newline="",encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))==[]


def test_mvi_reconstruction_guard_detects_tiny_nonzero_field():
    from independent_mvi.run_models import (
        _validate_rank_factors_for_accepted_endpoint as check)
    q=np.zeros((6,15,9))
    X=np.zeros((6,1,15))
    Y=np.zeros((6,1,9))
    assert check(q,X,Y)==0.
    q[4,5,3]=1.e-13
    with pytest.raises(RuntimeError,match="FACTORS_RECONSTRUCTION_MISMATCH"):
        check(q,X,Y)
    X[4,0,5]=1.e-13
    Y[4,0,3]=1.
    assert check(q,X,Y)==0.
