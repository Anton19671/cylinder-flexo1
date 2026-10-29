"""Headless temporal integrator GUI-routing tests; no shell calculations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

from cylinder_time_integrator_gui import (
    TIME_INTEGRATOR_CHOICES, time_integrator_code,
    plan_time_integrator_run, execute_time_integrator_plan,
)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"config"/"CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json"


class TemporalGuiRoutingTests(TestCase):
    def fixture(self,folder):
        case=json.loads(SOURCE.read_text(encoding="utf-8"))
        case["output_dir"]=str(folder/"NEWMARK")
        file=folder/"CASE.json"
        file.write_text(json.dumps(case),encoding="utf-8")
        return file,case

    def test_selector_labels_and_default(self):
        self.assertEqual(time_integrator_code(TIME_INTEGRATOR_CHOICES[0]),"NEWMARK")
        self.assertEqual(time_integrator_code(TIME_INTEGRATOR_CHOICES[1]),"GAUSS2_IRK4")
        self.assertEqual(time_integrator_code("STATIC"),"STATIC")
        self.assertEqual(time_integrator_code("MODAL"),"MODAL")
        with self.assertRaises(ValueError):time_integrator_code("RK4_EXPLICIT")

    def test_static_and_modal_tokens_are_distinct(self):
        with TemporaryDirectory() as d:
            folder=Path(d)
            file,case=self.fixture(folder)
            case["calculation_kind"]="STATIC"
            file.write_text(json.dumps(case),encoding="utf-8")
            with patch("cylinder_final_stage11a_dispatcher.validate_final_case",
                       return_value={"calculation_kind":"STATIC","method":"MVI"}):
                self.assertEqual(plan_time_integrator_run(file,"STATIC").integrator,"STATIC")
                with self.assertRaisesRegex(ValueError,"STATIC has no time integrator"):
                    plan_time_integrator_run(file,"GAUSS2_IRK4")
            case["calculation_kind"]="MODAL"
            file.write_text(json.dumps(case),encoding="utf-8")
            with patch("cylinder_final_stage11a_dispatcher.validate_final_case",
                       return_value={"calculation_kind":"MODAL","method":"MVI"}):
                self.assertEqual(plan_time_integrator_run(file,"MODAL").integrator,"MODAL")
                with self.assertRaisesRegex(ValueError,"MODAL has no time integrator"):
                    plan_time_integrator_run(file,"GAUSS2_IRK4")

    def test_gauss_plan_preserves_case_and_separates_outputs(self):
        with TemporaryDirectory() as d:
            folder=Path(d);file,case=self.fixture(folder)
            original=file.read_bytes()
            plan=plan_time_integrator_run(file,TIME_INTEGRATOR_CHOICES[1])
            self.assertEqual(plan.integrator,"GAUSS2_IRK4")
            self.assertEqual(plan.spatial_method,"MVI")
            self.assertEqual(plan.mvi_rank,case["rank"])
            self.assertEqual(plan.output_dir,folder/"NEWMARK_GAUSS2_IRK4")
            self.assertEqual(file.read_bytes(),original)
            self.assertFalse(plan.output_dir.exists())

    def test_newmark_and_gauss_dispatch_independent(self):
        with TemporaryDirectory() as d:
            folder=Path(d);file,case=self.fixture(folder)
            newmark=plan_time_integrator_run(file,"NEWMARK")
            gauss=plan_time_integrator_run(file,"GAUSS2_IRK4")
            with patch("cylinder_final_stage11a_dispatcher.run_final_case",
                       return_value={"status":"NEW"} ) as n:
                self.assertEqual(execute_time_integrator_plan(newmark)["status"],"NEW")
                n.assert_called_once_with(file)
            result=SimpleNamespace(summary={"status":"CONVERGED"},output_dir=gauss.output_dir)
            with patch("cylinder_gauss2_case.run_gauss2_mvi_case",return_value=result) as g:
                out=execute_time_integrator_plan(gauss)
                g.assert_called_once_with(file,output_dir=gauss.output_dir)
                self.assertEqual(out["time_integrator"],"GAUSS2_IRK4")
                self.assertEqual(out["output_dir"],str(gauss.output_dir))

    def test_series_rejected_for_gauss(self):
        with TemporaryDirectory() as d:
            file,_=self.fixture(Path(d))
            with self.assertRaisesRegex(ValueError,"серию"):
                plan_time_integrator_run(file,"GAUSS2_IRK4",series=True)

    def test_changed_case_cannot_execute_stale_preflight(self):
        with TemporaryDirectory() as d:
            folder=Path(d);file,case=self.fixture(folder)
            plan=plan_time_integrator_run(file,"GAUSS2_IRK4")
            case["geometry"]["h"]*=1.1
            file.write_text(json.dumps(case),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"CASE_CHANGED_AFTER_PREFLIGHT"):
                execute_time_integrator_plan(plan)

    def test_existing_gauss_output_rejected(self):
        with TemporaryDirectory() as d:
            folder=Path(d);file,_=self.fixture(folder)
            target=folder/"NEWMARK_GAUSS2_IRK4";target.mkdir()
            (target/"KEEP.txt").write_text("original")
            with self.assertRaisesRegex(ValueError,"OUTPUT_DIR_NOT_EMPTY"):
                plan_time_integrator_run(file,"GAUSS2_IRK4")
            self.assertEqual((target/"KEEP.txt").read_text(),"original")

    def test_gui_both_locations_reference_shared_selector(self):
        main=(ROOT/"interface"/"FINAL_GUI.py").read_text(encoding="utf-8")
        editor=(ROOT/"interface"/"cylinder_reference_gui.py").read_text(encoding="utf-8")
        self.assertIn("self.time_integrator=tk.StringVar",main)
        self.assertIn("plan_time_integrator_run(path,self.time_integrator.get()",main)
        self.assertIn("self.time_integrator=self.parent.time_integrator",editor)
        self.assertIn("values=TIME_INTEGRATOR_CHOICES",editor)
