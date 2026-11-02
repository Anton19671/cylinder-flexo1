"""Deferred validation of analytically derived G3D axial natural fluxes."""
import unittest
import numpy as np
from cylinder_g3d_natural_edges import axial_natural_end_tractions
from cylinder_functional_assembly import (CylinderFunctional,
                                          MECHANICAL_DERIVATIVES,
                                          surface_trapezoid)
from flexo_gradient_g3d import G3DMaterial


class G3DNaturalEdges(unittest.TestCase):
    def setUp(self):
        self.ops=surface_trapezoid(5,6,1.2,2.)

    def test_first_order_signs(self):
        n,s=self.ops.shape
        c=np.zeros((3,1,n,s))
        c[1]=3.
        e=np.zeros((3,2,n,s));e[1]=7.
        flux=axial_natural_end_tractions(c,MECHANICAL_DERIVATIVES[:3],
            self.ops.Dx,self.ops.Ds,e)
        np.testing.assert_allclose(flux.left.traction,-3.)
        np.testing.assert_allclose(flux.right.traction,3.)
        np.testing.assert_allclose(flux.left.electric_charge_flux,-7.)
        np.testing.assert_allclose(flux.right.electric_charge_flux,7.)

    def test_second_order_outward_double_same_sign(self):
        n,s=self.ops.shape
        c=np.zeros((6,1,n,s))
        c[3]=4.5
        flux=axial_natural_end_tractions(c,MECHANICAL_DERIVATIVES[:6],
            self.ops.Dx,self.ops.Ds)
        np.testing.assert_allclose(flux.left.normal_double_traction,4.5)
        np.testing.assert_allclose(flux.right.normal_double_traction,4.5)
        np.testing.assert_allclose(flux.left.traction,0.,atol=1.e-12)
        np.testing.assert_allclose(flux.right.traction,0.,atol=1.e-12)

    def test_third_order_normal_triple_changes_end_orientation(self):
        n,s=self.ops.shape
        c=np.zeros((10,1,n,s))
        c[6]=3.
        flux=axial_natural_end_tractions(c,MECHANICAL_DERIVATIVES,
            self.ops.Dx,self.ops.Ds)
        np.testing.assert_allclose(flux.left.normal_triple_traction,-3.)
        np.testing.assert_allclose(flux.right.normal_triple_traction,3.)

    def test_mixed_xs_has_no_double_factor(self):
        n,s=self.ops.shape
        c=np.zeros((6,1,n,s))
        length_s=2*np.pi*2.
        s_pos=np.arange(s)*length_s/s
        c[4,0,:,:]=np.sin(s_pos/2.)[None,:]
        flux=axial_natural_end_tractions(c,MECHANICAL_DERIVATIVES[:6],
            self.ops.Dx,self.ops.Ds)
        # Mixed q_xs contributes -d_s(A_11) exactly once.
        exact=-self.ops.Ds@np.sin(s_pos/2.)
        np.testing.assert_allclose(flux.right.traction[0],exact,atol=1.e-14)
        np.testing.assert_allclose(flux.left.traction[0],-exact,atol=1.e-14)

    def test_full_constitutive_weak_coefficients_for_every_theory(self):
        mat=G3DMaterial(100.,.25,.08,.005,-.002,.7)
        for theory in ("KL","FSDT","REDDY","CUF_TE1",
                       "CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                f=CylinderFunctional(theory,mat,radius_m=2.,
                    thickness_m=.2,surface_operators=self.ops,n_gauss=3)
                q=np.zeros((f.n_mechanical,*self.ops.shape))
                p=np.zeros((2,*self.ops.shape))
                flux=f.natural_end_fluxes(q,p)
                self.assertEqual(flux.left.traction.shape,
                                 (f.n_mechanical,self.ops.shape[1]))
                self.assertEqual(flux.left.electric_charge_flux.shape,
                                 (p.shape[0],self.ops.shape[1]))
                self.assertIn("NOT_STRONG_BC_CERTIFIED",flux.status)


if __name__=="__main__":
    unittest.main()
