"""Repository-level guards for coupled elastic G3D/Gz entrypoints.

These are structural checks, not substitutes for numerical convergence tests.
"""
from __future__ import annotations
import ast
from pathlib import Path
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
LOCAL_PREFIXES=("cylinder_","final_","flexo_","cuf_")

class TestElasticRepositoryIntegrity(unittest.TestCase):
    def test_every_local_python_import_resolves(self):
        paths=list(ROOT.rglob("*.py"))
        basenames={path.stem for path in paths}
        for path in paths:
            tree=ast.parse(path.read_text(encoding="utf-8-sig"),
                           filename=str(path))
            modules=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    modules.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:
                    modules.append(node.module.split(".")[0])
            missing=[name for name in modules
                     if name.startswith(LOCAL_PREFIXES) and name not in basenames]
            self.assertFalse(missing, f"{path.relative_to(ROOT)}: {missing}")

    def test_retired_materials_are_not_selectable_in_public_solvers(self):
        """Validate production CASE routes, not historical stand-alone kernels.

        Frozen-secant research modules are retained in the repository for
        reproducibility. Their mere presence must not be mistaken for a
        public elastoplastic solver; the actual executable CASE preflight
        must reject every retired material before assembling operators.
        """
        from cylinder_reference_case import prepare_reference_case
        from cylinder_modal_reference import _modal_spec
        from restored_gui_capabilities import (
            MATERIAL_MODELS, UNAVAILABLE, physics_capability)
        self.assertEqual(MATERIAL_MODELS, ("HOOKE",))
        static=json.loads((ROOT/"config/CASE_REFERENCE_CUF_G3D_STATIC.json")
                          .read_text(encoding="utf-8"))
        modal=json.loads((ROOT/"config/CASE_REFERENCE_MODAL_SEVEN_THEORIES.json")
                         .read_text(encoding="utf-8"))
        # Preflight must reach the material contract, not first fail due to
        # the legacy 5x5 SBP21 grid invalid in current modal SBP42.
        modal["fdm_grid"].update(nx=9,ns=9,
                                 operator_scheme="SBP42_FOURIER_ODD_CSR")
        for material in ("BIRGER", "J2_INCREMENTAL"):
            for theory in ("KL", "FSDT", "REDDY", "CUF"):
                cap=physics_capability(theory, "LINEAR", material, "FDM",
                                       cuf_model="TE1")
                self.assertEqual(cap.status, UNAVAILABLE,
                                 (theory, material))
            with self.subTest(material=material, calculation="STATIC"):
                case=dict(static,material_model=material)
                with self.assertRaisesRegex(ValueError,"requires HOOKE"):
                    prepare_reference_case(case)
            with self.subTest(material=material, calculation="MODAL"):
                case=dict(modal,material_model=material)
                with self.assertRaisesRegex(ValueError,"requires HOOKE"):
                    _modal_spec(case)
        # Preserve a structural boundary between independent historical
        # constitutive experiments and the current G3D/Gz reference routes.
        retired_imports={
            "cylinder_te1_local_kernel",
            "cylinder_cuf_stage7a_general_mechanics",
            "cylinder_te1_stage2b",
        }
        for name in ("interface/cylinder_final_stage11a_dispatcher.py",
                     "interface/cylinder_reference_case.py",
                     "numerics/cylinder_modal_reference.py",
                     "mechanics/cylinder_functional_assembly.py"):
            tree=ast.parse((ROOT/name).read_text(encoding="utf-8-sig"))
            direct=set()
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    direct.update(alias.name.split(".")[0]
                                  for alias in node.names)
                if isinstance(node,ast.ImportFrom) and node.module:
                    direct.add(node.module.split(".")[0])
            self.assertFalse(direct & retired_imports, name)

    def test_all_active_example_cases_are_elastic(self):
        cases=list((ROOT/"config").glob("CASE_REFERENCE_*.json"))
        self.assertGreaterEqual(len(cases),5)
        gradients=set()
        for path in cases:
            case=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(case.get("solver_backend"),"VARIATIONAL_REFERENCE")
            self.assertEqual(case.get("material_model"),"HOOKE")
            if case.get("calculation_kind")=="MODAL":
                # Unloaded modal eigenproblems use the linearized branch,
                # including when the input geometry is Green-Lagrange.
                self.assertEqual(case.get("mode"),"LINEAR_ELASTIC")
            else:
                self.assertEqual(case.get("mode"),"NONLINEAR_ELASTIC")
            gradient=case.get("micro",{}).get("gradient_model","").upper()
            if gradient:
                gradients.add(gradient)
        self.assertEqual(gradients,{"G3D","GZ"})

if __name__=="__main__":
    unittest.main()
