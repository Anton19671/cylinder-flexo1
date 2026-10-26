#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-solve regression tests: independent G3D/flexoelectric ON/OFF physics."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from cylinder_paths import activate
activate()
from flexo_scale_policy import (
    flexoelectric_enabled, micro_material_from_case, numerical_status
)
from flexo_gradient_g3d import enthalpy_parts, enthalpy_derivatives
from final_capability_registry import case_capability, UNAVAILABLE


def case(flexo):
    return {
        "scale_model":"MICRO", "geometry_model":"GREEN_LAGRANGE",
        "units":{"system":"SI_N_M_PA"},
        "material":{"E0":110.0,"nu0":0.25},
        "micro":{
            "gradient_model":"G3D","enable_flexoelectric":flexo,
            "ell_m":0.125,"mu1_c_per_m":2.0,"mu2_c_per_m":-0.5,
            "kappa_f_per_m":4.0,"rho_kg_m3":2000.0,
            "eta_s":0.0,"potential_order":2,
        },
    }


class TestIndependentSwitches(unittest.TestCase):
    def test_off_disables_electric_energy_and_direct_inverse_coupling(self):
        configuration=case(False)
        mat=micro_material_from_case(configuration)
        self.assertFalse(flexoelectric_enabled(configuration))
        self.assertEqual(mat.mu1_c_per_m,0)
        self.assertEqual(mat.mu2_c_per_m,0)
        self.assertEqual(mat.kappa_f_per_m,0)
        self.assertGreater(mat.ell_m,0)
        e=np.array([0.01,0.02,0.03,0.01,0.0,0.02])
        g=np.ones((6,3))*0.1
        E=np.array([0.2,-0.4,0.7])
        terms=enthalpy_parts(e,g,E,mat)
        self.assertGreater(terms["gradient"],0)
        self.assertEqual(terms["dielectric"],0)
        self.assertEqual(terms["flexoelectric"],0)
        _s,_m,D=enthalpy_derivatives(e,g,E,mat)
        np.testing.assert_array_equal(D,np.zeros(3))
        self.assertEqual(numerical_status(configuration),"EXPLICIT_VARIATIONAL_BACKEND_REQUIRED")
        configuration["solver_backend"]="VARIATIONAL_REFERENCE"
        self.assertEqual(numerical_status(configuration),"REFERENCE_GRADIENT_G3D")

    def test_on_has_coupled_material_terms(self):
        cfg=case(True);mat=micro_material_from_case(cfg)
        self.assertTrue(flexoelectric_enabled(cfg))
        self.assertEqual(mat.mu1_c_per_m,2)
        self.assertEqual(mat.mu2_c_per_m,-0.5)
        self.assertEqual(mat.kappa_f_per_m,4)
        cfg["solver_backend"]="VARIATIONAL_REFERENCE"
        self.assertEqual(numerical_status(cfg),"REFERENCE_COUPLED_G3D")

    def test_off_ignores_inactive_electrical_input_values(self):
        cfg=case(False)
        cfg["micro"].update(mu1_c_per_m="not-used",mu2_c_per_m="",kappa_f_per_m=None,
                            potential_order="")
        mat=micro_material_from_case(cfg)
        self.assertEqual((mat.mu1_c_per_m,mat.mu2_c_per_m,mat.kappa_f_per_m),(0,0,0))

    def test_on_requires_electric_parameters(self):
        cfg=case(True);cfg["micro"]["kappa_f_per_m"]=""
        with self.assertRaises(ValueError):micro_material_from_case(cfg)

    def test_micro_never_falls_through_to_macro_solver(self):
        for flexo in (True,False):
            cfg=case(flexo)
            capability=case_capability(cfg)
            self.assertEqual(capability.status,UNAVAILABLE)
            self.assertFalse(capability.can_run)

    def test_macro_selection_does_not_enable_electric_terms(self):
        self.assertEqual(numerical_status({"scale_model":"MACRO"}),
                         "EXISTING_CLASSICAL_ELASTIC")

    def test_invalid_non_boolean_switch_rejected(self):
        cfg=case(False)
        cfg["micro"]["enable_flexoelectric"]="OFF"
        with self.assertRaises(ValueError):micro_material_from_case(cfg)


if __name__=="__main__":
    unittest.main()

