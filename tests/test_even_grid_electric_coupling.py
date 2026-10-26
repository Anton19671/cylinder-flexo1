"""Full-shell even-periodic electric checkerboard tests for G3D and Gz.

Tests actual thickness-integrated electrical residuals, not just D_s.
The independent strong-form MKR discretization is a separate pending task.
"""
import unittest
import numpy as np
from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from flexo_gradient_g3d import G3DMaterial


class EvenGridElectroMechanicalTests(unittest.TestCase):
    def test_face_gradient_no_extra_nyquist_zero_for_even_ns(self):
        for ns in (4,6,8,12):
            with self.subTest(ns=ns):
                ops=surface_trapezoid(4,ns,1.2,2.)
                nodal=(-1.)**np.arange(ns)
                diff=ops.Ds@nodal
                h=2*np.pi*2./ns
                np.testing.assert_allclose(
                    diff,-2.*nodal/h,rtol=1.e-13,atol=1.e-13)
                self.assertGreater(float(diff@(ops.ws*diff)),0.)
                np.testing.assert_allclose(ops.Ds@np.ones(ns),0.,
                                           atol=1.e-13)

    def test_full_coupled_electric_residual_on_even_grid_seven_theories(self):
        for gradient in ("G3D","Gz"):
            for theory in ("KL","FSDT","REDDY",
                           "CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
                with self.subTest(gradient=gradient,theory=theory):
                    ops=surface_trapezoid(4,6,1.2,2.)
                    material=G3DMaterial(100.,.25,.08,.001,-.002,1.5,
                                        gradient_model=gradient)
                    ngauss=(2*int(theory[-1])+1 if theory.startswith("CUF")
                            else 7)
                    solver=CylinderFunctional(theory,material,
                        radius_m=2.,thickness_m=.2,
                        surface_operators=ops,n_gauss=ngauss)
                    q=np.zeros((solver.n_mechanical,*ops.shape))
                    p=np.zeros((2,*ops.shape))
                    p[0,:,:]=(-1.)**np.arange(ops.shape[1])[None,:]
                    result=solver.evaluate(q,p,energy_breakdown=False)
                    self.assertEqual(result.electrical_residual.shape,p.shape)
                    self.assertTrue(np.isfinite(result.electrical_residual).all())
                    self.assertGreater(
                        float(np.max(np.abs(result.electrical_residual))),
                        1.e-8, "electrical checkerboard became a false solution")
                    # This mode has zero weighted mean, so the result cannot
                    # be rescued merely by imposing a global potential gauge.
                    self.assertAlmostEqual(
                        float(result.electric_gauge_value),0.,places=9)
                    self.assertLess(result.total_enthalpy_j,0.)

if __name__=="__main__":
    unittest.main()
