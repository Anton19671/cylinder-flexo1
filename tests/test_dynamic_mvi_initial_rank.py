"""Preflight guards for dynamic MVI: no hidden additive initial field."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from cylinder_paths import activate
activate()
from cylinder_reference_case import prepare_reference_case

class DynamicMVIInitialRankTests(TestCase):
    def sample(self):
        path=ROOT/"config"/"CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_zero_initial_state_is_representable(self):
        case=self.sample()
        prepared=prepare_reference_case(case)
        self.assertEqual(prepared.method,"MVI")
        self.assertEqual(prepared.calculation_kind,"DYNAMIC")

    def test_unrepresentable_initial_state_aborts_before_any_run(self):
        case=self.sample()
        with patch("cylinder_variational_mvi._initial_profiles",
                   side_effect=ValueError("initial field rank is too large")):
            with self.assertRaisesRegex(
                    ValueError,"DYNAMIC_MVI_INITIAL_STATE_EXCEEDS_FIXED_RANK"):
                prepare_reference_case(case)

    def test_static_case_skips_dynamic_only_guard(self):
        case=json.loads((ROOT/"config"/"CASE_REFERENCE_CUF_G3D_STATIC.json").read_text(encoding="utf-8"))
        case["method"]="MVI"
        # The static preset is an FDM case; switching method must also
        # switch its exclusive spatial grid key.
        case["mvi_grid"]=case.pop("fdm_grid")
        case["rank"]=1
        case["reference_mvi"]={"max_sweeps":3}
        with patch("cylinder_variational_mvi._initial_profiles",
                   side_effect=RuntimeError("only relevant to solve")):
            prepared=prepare_reference_case(case)
            self.assertEqual(prepared.calculation_kind,"STATIC")
