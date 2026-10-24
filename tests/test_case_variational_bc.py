#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-PDE boundary CASE contract regressions."""
import unittest

from cylinder_boundary_case import (
    inspect_variational_case, CLASSICAL_STATUS, REFERENCE_STATUS,
)
from cylinder_variational_boundary import boundary_passport


def case(theory="KL", left="CLAMPED", right="CLAMPED"):
    data = {
        "theory": theory, "cuf_model": "TE1", "calculation_kind": "STATIC",
        "scale_model": "MACRO",
        "boundary_conditions": {
            "registry_version": "BC_REGISTRY_V2",
            "left": left, "right": right, "advanced": None,
            "variational": boundary_passport(
                mechanical_left=left, mechanical_right=right),
        },
    }
    return data


class CaseVariationalBoundaryTest(unittest.TestCase):
    def test_existing_classical_bc_reused_not_reassembled(self):
        for t in ("KL", "CUF", "FSDT", "REDDY"):
            with self.subTest(theory=t):
                result = inspect_variational_case(case(t))
                self.assertEqual(result["status"], CLASSICAL_STATUS)
                self.assertFalse(result["electrical_enabled"])

    def test_missing_passport_is_backward_compatible(self):
        data = case()
        del data["boundary_conditions"]["variational"]
        self.assertIsNone(inspect_variational_case(data))

    def test_preset_mismatch_explicitly_nonrunnable(self):
        data = case()
        data["boundary_conditions"]["variational"]["mechanical"]["right"]["support"] = "FREE"
        result = inspect_variational_case(data)
        self.assertEqual(result["status"], REFERENCE_STATUS)
        self.assertTrue(any("differs" in text for text in result["reasons"]))

    def test_legacy_cuf_s4_must_not_become_s3(self):
        data = case("CUF", left="SIMPLY_SUPPORTED")
        data["boundary_conditions"]["registry_version"] = "BC_REGISTRY_V1"
        self.assertEqual(inspect_variational_case(data)["status"], REFERENCE_STATUS)

    def test_g3d_higher_traces_not_silently_accepted_for_macro(self):
        data = case()
        data["boundary_conditions"]["variational"]["mechanical"]["left"]["higher"] = "FIX_FIRST_NORMAL"
        with self.assertRaisesRegex(ValueError, "MICRO"):
            inspect_variational_case(data)
        data["scale_model"] = "MICRO"
        self.assertEqual(inspect_variational_case(data)["status"], REFERENCE_STATUS)

    def test_flexoelectric_voltage_only_when_enabled(self):
        data = case("CUF")
        data["boundary_conditions"]["variational"]["electrical"]["outer"] = {
            "mode": "POTENTIAL", "value": 2.0,
        }
        with self.assertRaisesRegex(ValueError, "electric model ON"):
            inspect_variational_case(data)
        data["scale_model"] = "MICRO"
        data["micro"] = {"enable_flexoelectric": True}
        status = inspect_variational_case(data)
        self.assertTrue(status["electrical_enabled"])
        self.assertEqual(status["status"], REFERENCE_STATUS)

    def test_static_dynamic_law_rejection(self):
        data = case()
        data["boundary_conditions"]["variational"]["mechanical"]["left"]["time_law"] = {
            "kind": "HARMONIC", "frequency_hz": 10, "amplitude": 1,
        }
        with self.assertRaisesRegex(ValueError, "STATIC"):
            inspect_variational_case(data)
        data["calculation_kind"] = "DYNAMIC"
        self.assertEqual(inspect_variational_case(data)["status"], REFERENCE_STATUS)

    def test_vector_length_and_spring_guard(self):
        data = case()
        data["boundary_conditions"]["variational"]["mechanical"]["left"]["spring"] = [1.0]
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            inspect_variational_case(data)
        data["boundary_conditions"]["variational"]["mechanical"]["left"]["spring"] = [-1, 0, 0]
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            inspect_variational_case(data)

    def test_custom_component_range(self):
        data = case("CUF")
        left = data["boundary_conditions"]["variational"]["mechanical"]["left"]
        left["support"] = "CUSTOM"
        left["fixed_components"] = [0, 100]
        with self.assertRaisesRegex(ValueError, "fixed_components"):
            inspect_variational_case(data)

    def test_insulated_charge_must_not_be_ignored(self):
        data = case()
        data["boundary_conditions"]["variational"]["electrical"]["outer"]["value"] = 1.0
        with self.assertRaisesRegex(ValueError, "INSULATED"):
            inspect_variational_case(data)


if __name__ == "__main__":
    unittest.main()
