"""Saved-history comparison tests: no full shell calculations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from cylinder_newmark_gauss2_comparison import compare_newmark_gauss2_mvi
from cylinder_gauss2_rk import GAUSS2_C

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
             newmark=dict(dt_s=.1,start_time_s=0.,beta=.25,gamma=.5),
             steps=[dict(index=1,time_s=.1,state_file="REFERENCE_DYNAMIC_STEP_0001.npz")])
        self.gsummary=dict(method="EXPERIMENTAL_GAUSS2_FIXED_RANK_MVI",
             status="CONVERGED",rank=2,physical_case_sha256=self.digest,
             n_steps_requested=1,n_steps_accepted=1,dt_s=.1,t_start_s=0.,
             steps=[dict(index=1,time_s=.1,state_file="GAUSS2_MVI_STEP_0001.npz")])
        grid=np.ones((1,3,3))
        initial=dict(time_s=np.array(0.),mechanical=np.zeros_like(grid),
                     velocity=np.zeros_like(grid),electrical=np.zeros_like(grid))
        qend=.01*grid;vend=.2*grid
        end=dict(time_s=np.array(.1),mechanical=qend,
                 velocity=vend,electrical=.03*grid)
        np.savez_compressed(self.n/"REFERENCE_DYNAMIC_INITIAL.npz",
                            **initial,acceleration=2*grid)
        np.savez_compressed(self.g/"GAUSS2_MVI_INITIAL.npz",**initial)
        np.savez_compressed(self.n/"REFERENCE_DYNAMIC_STEP_0001.npz",
                            **end,acceleration=2*grid)
        stage_times=.1*GAUSS2_C
        gauss=dict(**end,
            stage_time_s=stage_times,
            stage_mechanical=stage_times[:,None,None,None]**2*
                             np.ones((2,1,3,3)),
            stage_velocity=2*stage_times[:,None,None,None]*
                           np.ones((2,1,3,3)),
            stage_acceleration=2*np.ones((2,1,3,3)),
            x_profiles=np.concatenate((
                np.stack((np.ones((3,)),np.zeros((3,))),axis=0)[None],
                np.stack((np.ones((3,)),np.zeros((3,))),axis=0)[None]),
                axis=0),
            y_profiles=np.concatenate((
                np.stack((.01*np.ones((3,)),np.zeros((3,))),axis=0)[None],
                np.stack((.03*np.ones((3,)),np.zeros((3,))),axis=0)[None]),
                axis=0),
            state_scales=np.ones((2,3,3)))
        np.savez_compressed(self.g/"GAUSS2_MVI_STEP_0001.npz",**gauss)
        physical=dict(calculation_kind="DYNAMIC",method="MVI",rank=2,
            mvi_grid=dict(nx=3,ns=3),
            dynamics=dict(n_steps=1,time_step_s=.1,density_kg_m3=1.))
        from cylinder_dynamic_case_passport import dynamic_physics_sha256
        self.digest=dynamic_physics_sha256(physical)
        self.nsummary["physical_case_sha256"]=self.digest
        self.gsummary["physical_case_sha256"]=self.digest
        (self.n/"REFERENCE_CASE.json").write_text(json.dumps(physical))
        (self.g/"GAUSS2_PHYSICAL_CASE.json").write_text(json.dumps(physical))
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.n/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(self.nsummary))
        (self.g/"GAUSS2_MVI_HISTORY.json").write_text(json.dumps(self.gsummary))
    def test_measured_difference(self):
        r=compare_newmark_gauss2_mvi(self.n,self.g)
        self.assertEqual(len(r["steps"]),1)
        self.assertAlmostEqual(r["steps"][0]["field_errors"]["mechanical"]["max_abs_difference"],0.)
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
    def test_corrupt_gauss_stage_rejected(self):
        path=self.g/"GAUSS2_MVI_STEP_0001.npz"
        with np.load(path) as data:
            arrays={k:data[k] for k in data.files}
        arrays["stage_velocity"]=arrays["stage_velocity"]+.02
        np.savez_compressed(path,**arrays)
        with self.assertRaisesRegex(ValueError,"KINEMATICS_INCONSISTENT"):
            compare_newmark_gauss2_mvi(self.n,self.g)

    def test_corrupt_newmark_acceleration_rejected(self):
        path=self.n/"REFERENCE_DYNAMIC_STEP_0001.npz"
        with np.load(path) as data:
            arrays={k:data[k] for k in data.files}
        arrays["acceleration"]=arrays["acceleration"]+3
        np.savez_compressed(path,**arrays)
        with self.assertRaisesRegex(ValueError,"NEWMARK_KINEMATICS_DRIFT"):
            compare_newmark_gauss2_mvi(self.n,self.g)

    def test_corrupt_gauss_profiles_rejected(self):
        path=self.g/"GAUSS2_MVI_STEP_0001.npz"
        with np.load(path) as data:
            arrays={k:data[k] for k in data.files}
        arrays["y_profiles"]=arrays["y_profiles"]+.1
        np.savez_compressed(path,**arrays)
        with self.assertRaisesRegex(ValueError,"FIXED_RANK_ENDPOINT"):
            compare_newmark_gauss2_mvi(self.n,self.g)

    def test_incomplete_trajectory_rejected(self):
        self.gsummary["n_steps_accepted"]=0;self.save()
        with self.assertRaisesRegex(ValueError,"INCOMPLETE_TRAJECTORY"):
            compare_newmark_gauss2_mvi(self.n,self.g)

if __name__=="__main__":unittest.main()
