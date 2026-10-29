"""Independent-run bifurcation pipeline tests without expensive shell PDE solves."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "interface"))
sys.path.insert(0,str(ROOT / "numerics"))
from cylinder_dynamic_parameter_sweep import plan_parameter_sweep, run_parameter_sweep
import numpy as np

def make_run(root, *, theory="CUF_TE2", amplitude=.004, steps=400,
             dt=.01, frequency=2.):
    """Standalone harmonic synthetic dataset used ONLY by export pipeline tests."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    modes=(int(theory[-1])+1) if theory.startswith("CUF_TE") else None
    n=3*modes if modes else (3 if theory=="KL" else 5)
    radial=2*modes if modes else 2
    original=np.zeros((n,4,4))
    electric=np.zeros((3,4,4))
    np.savez_compressed(root/"REFERENCE_DYNAMIC_INITIAL.npz",
        time_s=np.array(0.),mechanical=original,velocity=original,
        acceleration=original,electrical=electric)
    case={"solver_backend":"VARIATIONAL_REFERENCE","calculation_kind":"DYNAMIC",
          "theory":theory,"method":"MVI","rank":4,
          "geometry":{"L":1.2,"R":2.,"h":.2},
          "dynamics":{"time_step_s":dt,"n_steps":steps,
                      "time_law":{"kind":"HARMONIC","frequency_hz":frequency,
                                  "start_time_s":0.}}}
    summary={"theory":theory,"method":"MVI","rank":4,
             "calculation_kind":"DYNAMIC","accepted_steps":steps,
             "requested_steps":steps,"converged":True,"steps":[]}
    for index in range(1,steps+1):
        t=index*dt
        omega=2*np.pi*frequency
        q=original.copy();v=original.copy();a=original.copy()
        q[radial]=amplitude*np.sin(omega*t)
        v[radial]=amplitude*omega*np.cos(omega*t)
        a[radial]=-amplitude*omega*omega*np.sin(omega*t)
        p=electric.copy();p[0]=.01*np.cos(omega*t)
        filename=f"REFERENCE_DYNAMIC_STEP_{index:04d}.npz"
        np.savez_compressed(root/filename,time_s=np.array(t),
            load_factor=np.array(np.sin(omega*t)),mechanical=q,velocity=v,
            acceleration=a,electrical=p,
            face_inner_radial_displacement_m=q[radial]-.0001,
            face_outer_radial_displacement_m=q[radial]+.0001)
        summary["steps"].append({"index":index,"time_s":t,"converged":True,
                                  "state_file":filename})
    (root/"REFERENCE_CASE.json").write_text(json.dumps(case))
    (root/"FINAL_RUN_SUMMARY.json").write_text(json.dumps(summary))
    return root



def fixture(folder, *, values=(.002,.004), settings=None):
    root=Path(folder)
    case={
        "solver_backend":"VARIATIONAL_REFERENCE", "calculation_kind":"DYNAMIC",
        "theory":"CUF_TE2", "method":"MVI", "rank":4,
        "load_definition":{"q":.002},
        "dynamics":{"time_step_s":.01,"n_steps":400,
                    "time_law":{"kind":"HARMONIC","amplitude":1.,
                                "frequency_hz":2.,"phase_rad":0.,
                                "start_time_s":0.}},
        "geometry":{"L":1.2,"R":2.,"h":.2},
        "material":{"E0":100.,"nu0":.25},
        "micro":{"ell_m":.02,"mu1_c_per_m":.01,"mu2_c_per_m":-.02,
                 "kappa_f_per_m":1.5},
    }
    (root/"base.json").write_text(json.dumps(case),encoding="utf-8")
    spec={"base_case":"base.json","parameter_path":"load_definition.q",
          "values":list(values), "output_dir":"SWEEP_RESULTS",
          "transient_fraction":.5,"parameter_name":"pressure_pa",
          "export_each_run":False}
    spec.update(settings or {})
    source=root/"study.json"
    source.write_text(json.dumps(spec),encoding="utf-8")
    return source


def fake_preflight(case, *, base_dir):
    return SimpleNamespace(calculation_kind="DYNAMIC")


def fake_solver(case, case_file):
    output=Path(case["output_dir"])
    make_run(output,theory="CUF_TE2",amplitude=case["load_definition"]["q"],
             steps=400,dt=case["dynamics"]["time_step_s"],
             frequency=case["dynamics"]["time_law"]["frequency_hz"])
    # The actual PDE solver writes its exact CASE, so match this contract.
    (output/"REFERENCE_CASE.json").write_text(json.dumps(case),encoding="utf-8")
    return {"converged":True,"accepted_steps":400}


class ParameterSweepTests(unittest.TestCase):
    def test_preflight_validates_all_before_making_output(self):
        with TemporaryDirectory() as temp:
            path=fixture(temp)
            calls=[]
            def reject_second(case, *, base_dir):
                calls.append(case["load_definition"]["q"])
                if len(calls)==2:
                    raise ValueError("second parameter physically invalid")
                return fake_preflight(case,base_dir=base_dir)
            with self.assertRaisesRegex(ValueError,"physically invalid"):
                run_parameter_sweep(path,prepare_case=reject_second)
            self.assertEqual(len(calls),2)
            self.assertFalse((Path(temp)/"SWEEP_RESULTS").exists())

    def test_preflight_only_produces_no_file(self):
        with TemporaryDirectory() as temp:
            path=fixture(temp)
            report=run_parameter_sweep(path,preflight_only=True,
                                       prepare_case=fake_preflight)
            self.assertEqual(report["preflighted_cases"],2)
            self.assertEqual(report["status"],"PREFLIGHT_VALID")
            self.assertFalse((Path(temp)/"SWEEP_RESULTS").exists())

    def test_two_full_independent_runs_produce_real_bifurcation_workbook(self):
        with TemporaryDirectory() as temp:
            source=fixture(temp)
            result=run_parameter_sweep(source,prepare_case=fake_preflight,
                                       solve_case=fake_solver)
            self.assertEqual(result["status"],"COMPLETED")
            self.assertEqual(result["completed_runs"],2)
            self.assertEqual(result["bifurcation"]["runs"],2)
            root=Path(temp)/"SWEEP_RESULTS"
            self.assertTrue((root/"BIFURCATION_ANALYTICS"/"06_ДИНАМИКА_БИФУРКАЦИИ.xlsx").is_file())
            self.assertTrue((root/"BIFURCATION_RUNS.csv").is_file())
            values=[]
            for i in (1,2):
                case=json.loads((root/"CASES"/f"CASE_{i:04d}.json").read_text())
                values.append(case["load_definition"]["q"])
                self.assertFalse(case["postprocess"]["dynamic_analytics"])
                self.assertTrue((root/"RUNS"/f"RUN_{i:04d}"/"REFERENCE_DYNAMIC_INITIAL.npz").is_file())
            self.assertEqual(values,[.002,.004])
            self.assertFalse((root/"RUNS"/"RUN_0001"/"DYNAMIC_ANALYTICS").exists())
            with self.assertRaisesRegex(ValueError,"OUTPUT_DIR_NOT_EMPTY"):
                run_parameter_sweep(source,prepare_case=fake_preflight,
                                    solve_case=fake_solver)

    def test_failed_parameter_run_retains_prior_steps_and_skips_diagram(self):
        with TemporaryDirectory() as temp:
            source=fixture(temp)
            attempts=[]
            def fail_second(case, path):
                attempts.append(case["load_definition"]["q"])
                if len(attempts)==2:
                    return {"converged":False,"accepted_steps":0}
                return fake_solver(case,path)
            with self.assertRaisesRegex(RuntimeError,"parameter run 2"):
                run_parameter_sweep(source,prepare_case=fake_preflight,
                                    solve_case=fail_second)
            root=Path(temp)/"SWEEP_RESULTS"
            status=json.loads((root/"SWEEP_STATUS.json").read_text())
            self.assertEqual(status["status"],"INCOMPLETE_NO_BIFURCATION")
            self.assertEqual(status["completed_runs"],1)
            self.assertEqual(status["runs"][1]["status"],"FAILED")
            self.assertTrue((root/"RUNS"/"RUN_0001"/"REFERENCE_DYNAMIC_INITIAL.npz").is_file())
            self.assertFalse((root/"BIFURCATION_ANALYTICS").exists())

    def test_reject_invalid_physics_changes_or_values_without_output(self):
        with TemporaryDirectory() as temp:
            source=fixture(temp)
            for change in (
                {"parameter_path":"method"},
                {"parameter_path":"dynamics.time_step_s"},
                {"values":[.01,.01]},
                {"values":[True,.004]},
                {"values":[float("nan"),.004]},
                {"parameter_name":"=DDE"},
                {"transient_fraction":.99},
                {"export_each_run":"false"},
                {"forcing_hz":-1},
            ):
                with self.subTest(change=change):
                    spec=json.loads(source.read_text())
                    spec.update(change)
                    source.write_text(json.dumps(spec))
                    with self.assertRaises(ValueError):
                        plan_parameter_sweep(source)
                    self.assertFalse((Path(temp)/"SWEEP_RESULTS").exists())

    def test_frequency_scan_does_not_accept_forced_poincare_frequency(self):
        with TemporaryDirectory() as temp:
            source=fixture(temp,settings={"parameter_path":"dynamics.time_law.frequency_hz",
                                          "values":[1.,2.],"forcing_hz":3.})
            with self.assertRaisesRegex(ValueError,"override forcing_hz"):
                plan_parameter_sweep(source)

    def test_relative_tabulated_load_path_preserved_after_case_relocation(self):
        with TemporaryDirectory() as temp:
            source=fixture(temp,settings={"parameter_path":"load_definition.q",
                                          "values":[1.,2.]})
            original=Path(temp)/"base.json"
            case=json.loads(original.read_text())
            case["dynamics"]["time_law"]={"kind":"TABULATED","csv_path":"time.csv"}
            original.write_text(json.dumps(case))
            (Path(temp)/"time.csv").write_text("time_s,factor\n0,0\n1,1\n")
            planned=plan_parameter_sweep(source)
            for job in planned["jobs"]:
                self.assertEqual(job["case"]["dynamics"]["time_law"]["csv_path"],
                                 str((Path(temp)/"time.csv").resolve()))


if __name__=="__main__":
    unittest.main()
