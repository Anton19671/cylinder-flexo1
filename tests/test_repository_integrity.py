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

    def test_retired_material_code_not_in_working_tree(self):
        retired=("bi"+"rger","j"+"2")
        for path in ROOT.rglob("*"):
            if not path.is_file():continue
            if any(name in path.name.lower() for name in retired):
                self.fail(f"Retired filename: {path.relative_to(ROOT)}")
            # The historical GUI retains visible, DISABLED legacy widgets.
            # Their helper is a raising compatibility stub, not a material
            # backend; excluding this one UI layout must NOT exclude any
            # active mechanical, constitutive, or reference solver module.
            if path.relative_to(ROOT).as_posix()=="interface/FINAL_GUI.py":
                source=path.read_text(encoding="utf-8-sig")
                self.assertIn("from cylinder_gui_removed_material import", source)
                continue
            # Runtime source and configuration must stay elastic-only. Audit
            # journals and documentation may accurately describe retired
            # models; those mentions are not active solver capabilities.
            if path.suffix not in (".py",".json",".yml",".yaml"):
                continue
            data=path.read_text(encoding="utf-8-sig").lower()
            for name in retired:
                self.assertNotIn(name,data,
                    f"Retired material token in {path.relative_to(ROOT)}")

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
