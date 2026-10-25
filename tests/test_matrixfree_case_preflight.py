"""Deferred no-dense-SVD CASE preflight tests for the opt-in FDM backend.

Do not run as part of mathematics-first solver assembly.
"""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from cylinder_reference_case import prepare_reference_case


class SparsePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case_path=(Path(__file__).resolve().parents[1] /
                   "config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json")
        cls.original=json.loads(case_path.read_text(encoding="utf-8"))

    def case(self):
        c=copy.deepcopy(self.original)
        c["method"]="FDM"
        c.pop("rank",None)
        c.pop("mvi_grid",None)
        c.pop("reference_mvi",None)
        c["fdm_grid"]={"nx":6,"ns":5}
        c["fdm_backend"]="MATRIX_FREE_GMRES"
        c["reference_newton"].pop("max_free_dofs",None)
        c["reference_newton"]["max_total_dofs"]=5000
        c["dynamics"]["n_steps"]=1
        return c

    def test_no_dense_reference_assemble_for_dynamic_preflight(self):
        from cylinder_boundary_spatial import BoundarySpatialProblem
        c=self.case()
        with patch.object(BoundarySpatialProblem,"assemble",
                          side_effect=AssertionError("no dense preflight")):
            prepared=prepare_reference_case(c)
        self.assertEqual(prepared.fdm_backend,"MATRIX_FREE_GMRES")

    def test_large_nodal_count_above_dense_cap_is_allowed_to_preflight(self):
        from cylinder_boundary_spatial import BoundarySpatialProblem
        c=self.case()
        c["fdm_grid"]={"nx":18,"ns":17}
        c["reference_newton"]["max_total_dofs"]=4096
        with patch.object(BoundarySpatialProblem,"assemble",
                          side_effect=AssertionError("dense preflight not allowed")):
            prepared=prepare_reference_case(c)
        self.assertGreater(prepared.mechanical.shape[1]*prepared.mechanical.shape[2],256)
        self.assertGreater(prepared.mechanical.size,1000)

    def test_mvi_cannot_select_matrixfree_fdm(self):
        c=self.case()
        c["method"]="MVI"
        c["rank"]=1
        c["mvi_grid"]=c.pop("fdm_grid")
        with self.assertRaisesRegex(ValueError,"FDM only"):
            prepare_reference_case(c)

    def test_case_safety_limit_still_enforced(self):
        c=self.case()
        c["reference_newton"]["max_total_dofs"]=100
        with self.assertRaisesRegex(ValueError,"MATRIX_FREE_NODAL_DOF_SAFETY_LIMIT"):
            prepare_reference_case(c)


if __name__=="__main__":
    unittest.main()
