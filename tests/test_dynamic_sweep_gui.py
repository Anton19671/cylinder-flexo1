"""Headless checks for the dynamic bifurcation GUI model (no Tk display)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from cylinder_dynamic_sweep_gui import available_sweep_parameters,build_sweep_spec,save_sweep_spec
from cylinder_dynamic_parameter_sweep import plan_parameter_sweep

class DynamicSweepGuiTests(unittest.TestCase):
    def test_saved_case_parameter_selection_and_study(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            case={"solver_backend":"VARIATIONAL_REFERENCE",
                  "calculation_kind":"DYNAMIC","method":"MVI","rank":4,
                  "theory":"CUF","cuf_model":"TE2",
                  "load_definition":{"q":.01},
                  "dynamics":{"time_law":{"kind":"HARMONIC",
                                          "frequency_hz":2.,"amplitude":1.}}}
            path=root/"base.json"
            path.write_text(json.dumps(case),encoding="utf-8")
            fields={x[0] for x in available_sweep_parameters(case)}
            self.assertIn("load_definition.q",fields)
            self.assertNotIn("rank",fields)
            study=build_sweep_spec(path,"load_definition.q",.01,.05,5,
                                   root/"results",probe_name="P07_QUARTER_180")
            self.assertEqual(len(study["values"]),5)
            self.assertEqual(study["values"][0],.01)
            self.assertEqual(study["values"][-1],.05)
            study_path=save_sweep_spec(root/"study.json",study)
            plan=plan_parameter_sweep(study_path)
            self.assertEqual(len(plan["jobs"]),5)
            self.assertEqual(plan["jobs"][0]["case"]["rank"],4)
            self.assertFalse((root/"results").exists())

    def test_invalid_ranges_and_forbidden_fields(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/"case.json"
            path.write_text(json.dumps({"solver_backend":"VARIATIONAL_REFERENCE",
                "calculation_kind":"DYNAMIC","load_definition":{"q":.01}}))
            for key,lo,hi,n in [
                ("rank",0,1,4),("load_definition.q",1,1,4),
                ("load_definition.q",0,1,1),
                ("load_definition.q",0,1,257),
                ("load_definition.q",float("nan"),1,4)]:
                with self.subTest(key=key,lo=lo,hi=hi,n=n):
                    with self.assertRaises(ValueError):
                        build_sweep_spec(path,key,lo,hi,n,root/"output")
            with self.assertRaises(ValueError):
                build_sweep_spec(path,"load_definition.q",0,1,4,root/"output",
                                 transient_fraction=.95)
