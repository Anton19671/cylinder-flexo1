"""CASE runner is opt-in; test preflight and no Newmark overwrite."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from cylinder_gauss2_case import prepare_gauss2_mvi_case, run_gauss2_mvi_case

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/"config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json"

class Gauss2CaseTests(unittest.TestCase):
    def load(self):
        return json.loads(SOURCE.read_text(encoding="utf-8"))
    def test_prepared_physics_and_gauss_loads(self):
        case=self.load()
        prepared,settings,mvi,initial,loads=prepare_gauss2_mvi_case(case,base_dir=ROOT/"config")
        self.assertEqual(mvi.rank,case["rank"])
        self.assertEqual(initial.mechanical.shape,prepared.mechanical.shape)
        self.assertEqual(initial.velocity.shape,prepared.mechanical.shape)
        q,p=loads(settings.dt_s*.3)
        self.assertEqual(q.shape,prepared.mechanical.shape)
        if p is not None:self.assertEqual(p.shape,prepared.electrical.shape)
    def test_gauss_refuses_static_or_fdm(self):
        for name,value in (("calculation_kind","STATIC"),("method","FDM")):
            with self.subTest(name=name):
                case=self.load();case[name]=value
                with self.assertRaises(ValueError):
                    prepare_gauss2_mvi_case(case,base_dir=ROOT/"config")
    def test_missing_and_shared_output_paths_rejected(self):
        with TemporaryDirectory() as d:
            case=self.load();path=Path(d)/"case.json"
            case["output_dir"]=str(Path(d)/"newmark")
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError,"EXPLICIT_OUTPUT_DIR"):
                run_gauss2_mvi_case(path)
            with self.assertRaisesRegex(ValueError,"MUST_NOT_OVERWRITE_NEWMARK"):
                run_gauss2_mvi_case(path,output_dir=case["output_dir"])
            self.assertFalse(Path(case["output_dir"]).exists())
    def test_history_uses_case_digest_and_dynamics(self):
        with TemporaryDirectory() as d:
            case=self.load();path=Path(d)/"case.json"
            case["output_dir"]=str(Path(d)/"newmark")
            path.write_text(json.dumps(case))
            with patch("cylinder_gauss2_case.run_gauss2_variational_mvi_history") as runner:
                runner.return_value="called"
                out=run_gauss2_mvi_case(path,output_dir=Path(d)/"gauss")
                self.assertEqual(out,"called")
                kw=runner.call_args.kwargs
                self.assertEqual(kw["n_steps"],case["dynamics"]["n_steps"])
                self.assertEqual(kw["physical_case"],case)
                self.assertEqual(kw["mvi_settings"].rank,case["rank"])
                self.assertEqual(kw["density_kg_m3"],case["dynamics"]["density_kg_m3"])
                self.assertFalse((Path(d)/"gauss").exists())

if __name__=="__main__":unittest.main()
