"""Regression guards for the TE1 initial MVI basis and obsolete KL GL APIs."""
import unittest
import numpy as np
from unittest.mock import patch
from cylinder_kl_stage9e_strong_fdm_gl_elastic import (
    analytic_tangent_action, analytic_jacobian)


class OctoberGuards(unittest.TestCase):
    def test_legacy_gl_jacobian_rejected_before_using_hybrid_formulas(self):
        with self.assertRaisesRegex(NotImplementedError,"GL_ANALYTIC_JACOBIAN"):
            analytic_jacobian(None,None,None,None,E=1.,nu=.2,h=.1,
                              generalized_load=None)
        with self.assertRaisesRegex(NotImplementedError,"GL_ANALYTIC_TANGENT"):
            analytic_tangent_action(None,None,None,None,E=1.,nu=.2,h=.1)

    def test_end_face_constraints_no_redundant_rows(self):
        from cylinder_functional_assembly import CylinderFunctional,surface_trapezoid
        from flexo_gradient_g3d import G3DMaterial
        from cylinder_variational_boundary import compile_boundary_program
        ops=surface_trapezoid(4,5,1.2,2.)
        f=CylinderFunctional("CUF_TE1",
            G3DMaterial(100.,.25,.08,.01,-.02,1.5),
            radius_m=2.,thickness_m=.2,
            surface_operators=ops,n_gauss=3)
        q=np.zeros((f.n_mechanical,*ops.shape))
        p=np.zeros((2,*ops.shape))
        spec={"electrical":{"left":{"mode":"POTENTIAL","value":0.},
                            "outer":{"mode":"POTENTIAL","value":0.}}}
        result=compile_boundary_program(f,q,p,spec=spec)
        electric=[row for row in result.constraints if row.field=="electrical"]
        # The face constraint at x=0 is implied by the two end coefficients.
        self.assertEqual(len(electric),2*ops.shape[1]+
                         (ops.shape[0]-1)*ops.shape[1])


if __name__=="__main__":
    unittest.main()
