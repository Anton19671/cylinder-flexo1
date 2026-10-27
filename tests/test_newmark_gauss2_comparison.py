"""Saved-history comparison tests: no full shell calculations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from cylinder_newmark_gauss2_comparison import compare_newmark_gauss2_mvi

class TimeMethodComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        root=Path(self.tmp.name)
        self.n=root/"newmark";self.g=root/"gauss"
        self.n.mkdir();self.g.mkdir()
        self.digest="f"*64
        self.nsummary=dict(calculation_kind="DYNAMIC",method="MVI",
             status="CONVERGED",converged=True,rank=2,
             physical_case_sha256=self.digest,electrical_enabled=True,
             requested_steps=1,accepted_steps=1,
             newmark=dict(dt_s=.1,start_time_s=0.),
             steps=[dict(index=1,time_s=.1,state_file="REFERENCE_DYNAMIC_STEP_0001.npz")])
        self.gsummary=dict(method="EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI",
             status="CONVERGED",rank=2,physical_case_sha256=self.digest,
             n_steps_requested=1,n_steps_accepted=1,dt_s=.1,t_start_s=0.,
             steps=[dict(index=1,time_s=.1,state_file="GAUSS2_MVI_STEP_0001.npz")])
        initial=dict(time_s=np.array(0.),mechanical=np.zeros((1,3,3)),
                     velocity=np.zeros((1,3,3)),electrical=np.zeros((1,3,3)))
        end=dict(time_s=np.array(.1),mechanical=np.ones((1,3,3)),
                 velocity=2*np.ones((1,3,3)),electrical=3*np.ones((1,3,3)))
        np.savez_compressed(self.n/"REFERENCE_DYNAMIC_INITIAL.npz",**initial)
        np.savez_compressed(self.g/"GAUSS2_MVI_INITIAL.npz",**initial)
        np.savez_compressed(self.n/"REFERENCE_DYNAMIC_STEP_0001.npz",**end)
        end["mechanical"]=1.01*end["mechanical"]
        np.savez_compressed(self.g/"GAUSS2_MVI_STEP_0001.npz",**end)
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.n/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(self.nsummary))
        (self.g/"GAUSS2_MVI_HISTORY.json").write_text(json.dumps(self.gsummary))
    def test_measured_difference(self):
        r=compare_newmark_gauss2_mvi(self.n,self.g)
        self.assertEqual(len(r["steps"]),1)
        self.assertAlmostEqual(r["steps"][0]["field_errors"]["mechanical"]["max_abs_difference"],.01)
        self.assertFalse(r["convergence_study_verified"])
    def test_mismatched_physics_rejected(self):
        self.gsummary["physical_case_sha256"]="0"*64;self.save()
        with self.assertRaisesRegex(ValueError,"PHYSICAL_CASE_MISMATCH"):
            compare_newmark_gauss2_mvi(self.n,self.g)
    def test_mismatched_time_rejected(self):
        self.gsummary["dt_s"]=.2;self.save()
        with self.assertRaisesRegex(ValueError,"TIME_GRID_MISMATCH"):
            compare_newmark_gauss2_mvi(self.n,self.g)
    def test_missing_digest_rejected(self):
        self.nsummary.pop("physical_case_sha256");self.save()
        with self.assertRaisesRegex(ValueError,"MISSING"):
            compare_newmark_gauss2_mvi(self.n,self.g)
    def test_missing_field_rejected(self):
        target=self.g/"GAUSS2_MVI_STEP_0001.npz"
        with np.load(target) as f:
            vals={k:f[k] for k in f.files if k!="electrical"}
        np.savez_compressed(target,**vals)
        with self.assertRaisesRegex(ValueError,"MISSING_FIELDS"):
            compare_newmark_gauss2_mvi(self.n,self.g)
    def test_incomplete_trajectory_rejected(self):
        self.gsummary["n_steps_accepted"]=0;self.save()
        with self.assertRaisesRegex(ValueError,"INCOMPLETE_TRAJECTORY"):
            compare_newmark_gauss2_mvi(self.n,self.g)

if __name__=="__main__":unittest.main()
