"""Integration tests for automatic dynamic analytics after accepted Newmark steps."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"interface"))
sys.path.insert(0,str(ROOT/"numerics"))
from cylinder_reference_case import _postprocess_options, prepare_reference_case, run_reference_dynamic_case

class AutomaticDynamicAnalyticsTests(TestCase):
    def sample(self):
        return json.loads((ROOT/"config"/"CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json").read_text(encoding="utf-8"))

    def test_dynamic_analytics_configuration_defaults_and_validation(self):
        case=self.sample()
        self.assertTrue(_postprocess_options(case)["dynamic_analytics"]["enabled"])
        case["postprocess"]={"dynamic_analytics":False}
        self.assertFalse(_postprocess_options(case)["dynamic_analytics"]["enabled"])
        case["postprocess"]={"dynamic_analytics":{"transient_fraction":.5,"forcing_hz":2.}}
        opts=_postprocess_options(case)["dynamic_analytics"]
        self.assertEqual(opts["transient_fraction"],.5)
        self.assertEqual(opts["forcing_hz"],2.)
        case["postprocess"]["dynamic_analytics"]["transient_fraction"]=.95
        with self.assertRaises(ValueError):
            _postprocess_options(case)
        case["postprocess"]={"dynamic_analytics":{"unknown":True}}
        with self.assertRaises(ValueError):
            _postprocess_options(case)

    def test_short_dynamic_case_exports_six_books(self):
        with TemporaryDirectory() as tmp:
            case=self.sample()
            case["dynamics"]["n_steps"]=2
            case["output_dir"]=str(Path(tmp)/"run")
            case_path=Path(tmp)/"CASE.json"
            case_path.write_text(json.dumps(case),encoding="utf-8")
            prepared=prepare_reference_case(case,base_dir=case_path.parent)
            summary=run_reference_dynamic_case(case,case_path,prepared=prepared)
            self.assertEqual(summary["accepted_steps"],2)
            self.assertTrue(summary["converged"])
            self.assertEqual(summary["dynamic_analytics"]["status"],"EXPORTED",
                             summary["dynamic_analytics"])
            output=Path(summary["dynamic_analytics"]["output_dir"])
            self.assertTrue((output/"00_ДИНАМИКА_ОБЩИЕ_ДАННЫЕ.xlsx").is_file())
            self.assertTrue((output/"01_ДИНАМИКА_ИСТОРИИ_ПЕРЕМЕЩЕНИЙ.xlsx").is_file())
            self.assertTrue((output/"02_ДИНАМИКА_ЧАСТОТНЫЙ_АНАЛИЗ.xlsx").is_file())
            self.assertTrue((output/"03_ДИНАМИКА_ФАЗОВЫЕ_ТРАЕКТОРИИ.xlsx").is_file())
            self.assertTrue((output/"04_ДИНАМИКА_СЕЧЕНИЯ_ПУАНКАРЕ.xlsx").is_file())
            self.assertTrue((output/"05_ДИНАМИКА_ПОКАЗАТЕЛИ_ЛЯПУНОВА.xlsx").is_file())
            persisted=json.loads((Path(case["output_dir"])/"FINAL_RUN_SUMMARY.json").read_text())
            self.assertEqual(persisted["dynamic_analytics"]["status"],"EXPORTED")

    def test_disabled_export_does_not_create_analytics_folder(self):
        with TemporaryDirectory() as tmp:
            case=self.sample()
            case["dynamics"]["n_steps"]=2
            case["output_dir"]=str(Path(tmp)/"run")
            case["postprocess"]={"dynamic_analytics":False}
            case_path=Path(tmp)/"CASE.json"
            case_path.write_text(json.dumps(case),encoding="utf-8")
            summary=run_reference_dynamic_case(case,case_path,
                       prepared=prepare_reference_case(case,base_dir=case_path.parent))
            self.assertTrue(summary["converged"])
            self.assertEqual(summary["dynamic_analytics"]["status"],"DISABLED")
            self.assertFalse((Path(case["output_dir"])/"DYNAMIC_ANALYTICS").exists())

    def test_export_failure_does_not_overwrite_physical_result(self):
        with TemporaryDirectory() as tmp:
            case=self.sample()
            case["dynamics"]["n_steps"]=2
            case["output_dir"]=str(Path(tmp)/"run")
            case_path=Path(tmp)/"CASE.json"
            case_path.write_text(json.dumps(case),encoding="utf-8")
            with mock.patch("cylinder_dynamic_analytics.export_dynamic_analytics",
                            side_effect=RuntimeError("test visualization failure")):
                summary=run_reference_dynamic_case(case,case_path,
                          prepared=prepare_reference_case(case,base_dir=case_path.parent))
            self.assertTrue(summary["converged"])
            self.assertEqual(summary["accepted_steps"],2)
            self.assertEqual(summary["dynamic_analytics"]["status"],"EXPORT_FAILED")
            self.assertTrue((Path(case["output_dir"])/"REFERENCE_DYNAMIC_STEP_0002.npz").is_file())
