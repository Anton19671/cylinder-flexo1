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

    def test_static_case_does_not_invoke_dynamic_initial_preflight(self):
        case=self.sample()
        case["calculation_kind"]="STATIC"
        case.pop("dynamics",None)
        case.setdefault("load_definition",{})["load_factors"]=[0.]
        with patch("cylinder_variational_mvi._initial_profiles",
                   side_effect=RuntimeError("called only by static solver")):
            # A preflight is not a static MVI solve.
            prepared=prepare_reference_case(case)
            self.assertEqual(prepared.calculation_kind,"STATIC")
