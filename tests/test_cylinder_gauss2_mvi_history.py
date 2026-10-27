"""Regression tests for multi-step Gauss IRK4 fixed-rank MVI history.

Uses manufactured coupled equations only; does not run a shell simulation.
"""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from cylinder_gauss2_rk import Gauss2Settings
from cylinder_gauss2_mvi import Gauss2MVISettings
from cylinder_gauss2_mvi_history import (
    run_gauss2_variational_mvi_history, verify_gauss2_mvi_step)
from cylinder_gauss2_variational_mvi import solve_gauss2_variational_mvi_step
from test_cylinder_gauss2_variational_mvi import ToyProblem, GaussPhysicalMviTests


class GaussHistoryTest(unittest.TestCase):
    def fixture(self, electric=True, moving=False):
        p=ToyProblem(electrical=electric,moving=moving)
        state=GaussPhysicalMviTests().state(electric=electric)
        return p,state,Gauss2Settings(.04),Gauss2MVISettings(
            rank=1,absolute_tolerance=1e-10,
            field_change_tolerance=1e-9,relative_tolerance=1e-9)

    def run_history(self,p,s,dt,r,folder,steps=2):
        return run_gauss2_variational_mvi_history(
            p,s,dt,mvi_settings=r,n_steps=steps,density_kg_m3=1.,
            loads_at=lambda t:(None,None),output_dir=folder)

    def test_two_steps_full_state_rank_and_electricity_persist(self):
        p,s,dt,r=self.fixture()
        with TemporaryDirectory() as d:
            folder=Path(d)/"run"
            completed=self.run_history(p,s,dt,r,folder)
            self.assertEqual(completed.accepted_steps,2)
            self.assertEqual(completed.summary["status"],"CONVERGED")
            for i in (1,2):
                with np.load(folder/f"GAUSS2_MVI_STEP_{i:04d}.npz") as stored:
                    values=np.einsum("crx,crs->cxs",
                                     stored["x_profiles"],stored["y_profiles"])*stored["state_scales"]
                    np.testing.assert_allclose(values,np.concatenate(
                        (stored["mechanical"],stored["electrical"])),atol=1e-12)
                    self.assertEqual(stored["stage_acceleration"].shape,(2,)+s.mechanical.shape)
                    self.assertEqual(stored["stage_time_s"].shape,(2,))
            saved=json.loads((folder/"GAUSS2_MVI_HISTORY.json").read_text())
            self.assertEqual(saved["n_steps_accepted"],2)

    def test_moving_support_refused_before_accepting_step(self):
        p,s,dt,r=self.fixture(moving=True)
        with TemporaryDirectory() as d:
            folder=Path(d)/"run"
            with self.assertRaisesRegex(RuntimeError,"MOVING_MECHANICAL_SUPPORT"):
                self.run_history(p,s,dt,r,folder)
            self.assertFalse((folder/"GAUSS2_MVI_STEP_0001.npz").exists())
            saved=json.loads((folder/"GAUSS2_MVI_HISTORY.json").read_text())
            self.assertEqual(saved["n_steps_accepted"],0)
            self.assertEqual(saved["status"],"FAILED")

    def test_failure_does_not_overwrite_first_accepted_state(self):
        p,s,dt,r=self.fixture(electric=False)
        import cylinder_gauss2_mvi_history as mod
        original=mod.solve_gauss2_variational_mvi_step
        calls=[0]
        def fail_second(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise ValueError("forced failure")
            return original(*args,**kwargs)
        with TemporaryDirectory() as d:
            folder=Path(d)/"run"
            with patch.object(mod,"solve_gauss2_variational_mvi_step",fail_second):
                with self.assertRaisesRegex(RuntimeError,"STEP_2_FAILED"):
                    self.run_history(p,s,dt,r,folder,steps=3)
            self.assertTrue((folder/"GAUSS2_MVI_STEP_0001.npz").exists())
            self.assertFalse((folder/"GAUSS2_MVI_STEP_0002.npz").exists())
            saved=json.loads((folder/"GAUSS2_MVI_HISTORY.json").read_text())
            self.assertEqual(saved["n_steps_accepted"],1)

    def test_stage_time_and_rank_profile_tampering_refused(self):
        p,s,dt,r=self.fixture(electric=False)
        result=solve_gauss2_variational_mvi_step(
            p,s,dt,mvi_settings=r,density_kg_m3=1.,
            loads_at=lambda t:(None,None))
        verify_gauss2_mvi_step(s,result,dt,rank=1)
        bad=replace(result,stages=(replace(result.stages[0],time_s=.001),result.stages[1]))
        with self.assertRaisesRegex(ValueError,"INCORRECT_STAGE_TIME"):
            verify_gauss2_mvi_step(s,bad,dt,rank=1)
        x=result.x_profiles.copy();x[0,0,1]+=.01
        with self.assertRaisesRegex(ValueError,"FIXED_RANK_ENDPOINT"):
            verify_gauss2_mvi_step(s,replace(result,x_profiles=x),dt,rank=1)

    def test_nonempty_folder_and_invalid_steps_rejected(self):
        p,s,dt,r=self.fixture(electric=False)
        with TemporaryDirectory() as d:
            folder=Path(d)/"run";folder.mkdir()
            (folder/"do_not_change").write_text("original")
            with self.assertRaisesRegex(ValueError,"OUTPUT_DIR_NOT_EMPTY"):
                self.run_history(p,s,dt,r,folder)
            self.assertEqual((folder/"do_not_change").read_text(),"original")
            for steps in (0,True,1.2,10001):
                with self.assertRaisesRegex(ValueError,"n_steps"):
                    self.run_history(p,s,dt,r,Path(d)/"unused",steps=steps)


if __name__=="__main__":
    unittest.main()
