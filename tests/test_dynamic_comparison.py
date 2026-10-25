"""Regression suite for the fail-closed dynamic MVI/FDM comparator."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import numpy as np
from cylinder_dynamic_comparison import compare_dynamic_runs, export_comparison


class TestDynamicComparison(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.mvi=self.root/"MVI";self.fdm=self.root/"FDM"
        for method,folder in (("MVI",self.mvi),("FDM",self.fdm)):
            folder.mkdir()
            c={"solver_backend":"VARIATIONAL_REFERENCE",
               "method":method,"calculation_kind":"DYNAMIC","theory":"CUF",
               "cuf_model":"TE1","scale_model":"MICRO",
               "geometry":{"L":1.2,"R":2.,"h":.2},
               "material":{"E0":100.,"nu0":.25},
               "geometry_model":"GREEN_LAGRANGE","material_model":"HOOKE",
               "boundary_conditions":{"variational":{"mechanical":{"left":{"support":"CLAMPED"}}}},
               "dynamics":{"time_step_s":.01,"n_steps":2,"damping":"NONE"},
               "mvi_grid" if method=="MVI" else "fdm_grid":{"nx":4,"ns":4}}
            if method=="MVI":c.update(rank=1,reference_mvi={"max_sweeps":30})
            (folder/"REFERENCE_CASE.json").write_text(json.dumps(c))
            summary={"method":method,"calculation_kind":"DYNAMIC","theory":"CUF_TE1",
                     "scale_model":"MICRO","geometry_model":"GREEN_LAGRANGE",
                     "material_model":"HOOKE","field_order":"component,x,s",
                     "thickness_basis":"(2*z/h)**k","electrical_enabled":True,
                     "density_kg_m3":1300.,"newmark":{"dt_s":.01,"beta":.25,"gamma":.5},
                     "damping":{"kind":"NONE","mass_damping_rate_per_s":0.},
                     "converged":True,"status":"CONVERGED","accepted_steps":2,
                     "requested_steps":2,"steps":[]}
            if method=="MVI":summary["rank"]=1
            for i in (1,2):
                name=f"REFERENCE_DYNAMIC_STEP_{i:04d}.npz"
                q=np.zeros((6,4,4));q[4]=.0002*i
                if method=="MVI":q[4,2,1]+=.000002*i
                p=np.full((2,4,4),.1*i)
                np.savez_compressed(folder/name,mechanical=q,electrical=p,
                    velocity=q*3,acceleration=q*8,damping_force=q*0,
                    full_residual=np.zeros(q.size+p.size),
                    constraint_reaction=np.zeros(q.size+p.size),
                    time_s=np.array(.01*i),load_factor=np.array(i))
                summary["steps"].append({"index":i,"time_s":.01*i,"load_factor":i,
                    "converged":True,"full_equilibrium_satisfied":True,
                    "admissible_residual_scaled":1.e-10,
                    "residual_relative_to_initial":1.e-8,
                    "constraint_defect_scaled":1.e-13,"state_file":name})
            (folder/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary))

    def _edit(self,folder,file,change):
        p=folder/file
        data=json.loads(p.read_text());change(data);p.write_text(json.dumps(data))

    def test_computed_fields_and_export(self):
        result=compare_dynamic_runs(self.mvi,self.fdm)
        self.assertEqual(len(result["steps"]),2)
        self.assertAlmostEqual(result["steps"][0]["field_errors"]["radial_displacement"]["max_abs_difference"],2.e-6)
        self.assertAlmostEqual(result["steps"][0]["field_errors"]["radial_displacement"]["max_rel_to_fdm_peak"],.01)
        self.assertIsNone(result["steps"][0]["field_errors"]["damping_force"]["max_rel_to_fdm_peak"])
        self.assertFalse(result["convergence_study_verified"])
        export_comparison(result,self.root/"errors.json",self.root/"errors.csv")
        self.assertTrue((self.root/"errors.json").is_file())
        self.assertIn("radial_displacement",(self.root/"errors.csv").read_text())

    def test_paired_runner_checks_fdm_before_mvi(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from cylinder_dynamic_comparison import run_dynamic_pair
        original=self.mvi/"REFERENCE_CASE.json"
        output=self.root/"paired_fdm_first"
        order=[]
        def prepare(case,base_dir=None):
            order.append(("validate",case["method"]))
            return object()
        def run(case,case_path):
            order.append(("run",case["method"]))
            # A failed independent baseline must stop the experiment before
            # the reduced-space MVI is even attempted.
            return {"converged":False,"status":"REFERENCE_NOT_CONVERGED"}
        with patch.dict(sys.modules,{"cylinder_reference_case":SimpleNamespace(
                prepare_reference_case=prepare,run_reference_dynamic_case=run)}):
            with self.assertRaisesRegex(RuntimeError,"FDM did not converge"):
                run_dynamic_pair(original,output)
        self.assertEqual([method for event,method in order if event=="run"],["FDM"])
        self.assertFalse((output/"MVI").exists())

    def test_rejects_changed_physics(self):
        self._edit(self.mvi,"REFERENCE_CASE.json",lambda d:d["geometry"].__setitem__("h",.25))
        with self.assertRaisesRegex(ValueError,"physics"):
            compare_dynamic_runs(self.mvi,self.fdm)

    def test_rejects_incomplete_run(self):
        self._edit(self.mvi,"FINAL_RUN_SUMMARY.json",lambda d:d.__setitem__("accepted_steps",1))
        with self.assertRaisesRegex(ValueError,"incomplete"):
            compare_dynamic_runs(self.mvi,self.fdm)

    def test_rejects_projected_only(self):
        self._edit(self.mvi,"FINAL_RUN_SUMMARY.json",lambda d:d["steps"][0].__setitem__("full_equilibrium_satisfied",False))
        with self.assertRaisesRegex(ValueError,"equilibrium"):
            compare_dynamic_runs(self.mvi,self.fdm)

    def test_rejects_bad_time(self):
        self._edit(self.mvi,"FINAL_RUN_SUMMARY.json",lambda d:d["steps"][0].__setitem__("time_s",.55))
        with self.assertRaisesRegex(ValueError,"NPZ and summary disagree"):
            compare_dynamic_runs(self.mvi,self.fdm)

    def test_rejects_missing_electrical_field(self):
        p=self.mvi/"REFERENCE_DYNAMIC_STEP_0001.npz"
        with np.load(p) as d:saved={k:d[k] for k in d.files if k!="electrical"}
        np.savez_compressed(p,**saved)
        with self.assertRaisesRegex(ValueError,"electric state"):
            compare_dynamic_runs(self.mvi,self.fdm)
