"""Staged SBP Green identity checks; run only at later integration-test stage."""
import unittest
import numpy as np
from cylinder_sbp_derivatives import (axial_sbp21_gradient,
    periodic_sbp_odd_gradient,green_identity_defects,require_green_identity)
from cylinder_functional_assembly import surface_sbp21


class SBPSurfaceTest(unittest.TestCase):
    def test_exact_matrix_green_identities(self):
        for nx,ns in ((4,5),(5,7),(9,11)):
            with self.subTest(nx=nx,ns=ns):
                ops=surface_sbp21(nx,ns,1.2,2.)
                report=require_green_identity(ops.Dx,ops.Ds,ops.wx,ops.ws)
                self.assertLess(report["relative_max"],3e-12)
                ones_x=np.ones(nx);ones_s=np.ones(ns)
                np.testing.assert_allclose(ops.Dx@ones_x,0.,atol=1e-12)
                np.testing.assert_allclose(ops.Ds@ones_s,0.,atol=1e-12)
                self.assertEqual(np.linalg.matrix_rank(ops.Ds),ns-1)

    def test_green_identity_for_random_fields(self):
        ops=surface_sbp21(7,9,1.6,3.)
        r=np.random.default_rng(160)
        u=r.normal(size=ops.shape)
        v=r.normal(size=ops.shape)
        W=ops.area_weights
        lhs=float(np.sum(W*(u*(ops.Dx@v)+(ops.Dx@u)*v)))
        rhs=float(np.sum(ops.ws*(u[-1]*v[-1]-u[0]*v[0])))
        self.assertAlmostEqual(lhs,rhs,places=11)
        lhs_s=float(np.sum(W*(u*(v@ops.Ds.T)+(u@ops.Ds.T)*v)))
        self.assertAlmostEqual(lhs_s,0.,places=11)

    def test_even_periodic_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,"odd ns"):
            surface_sbp21(5,6,1.2,2.)

    def test_broken_pair_fails_closed(self):
        d,w=axial_sbp21_gradient(6,1.2)
        s,ws=periodic_sbp_odd_gradient(7,2*np.pi)
        d[0,0]*=1.4
        self.assertGreater(green_identity_defects(d,s,w,ws)["relative_max"],1.e-3)
        with self.assertRaisesRegex(ValueError,"SBP_GREEN_IDENTITY_VIOLATED"):
            require_green_identity(d,s,w,ws)

    def test_higher_mixed_derivative_discrete_green_identity(self):
        ops=surface_sbp21(7,9,1.4,2.)
        r=np.random.default_rng(51)
        a=r.normal(size=ops.shape)
        v=r.normal(size=ops.shape)
        Dx,Ds=ops.Dx,ops.Ds
        W=ops.area_weights
        # (A_xxs, D_x^2 D_s q): exact after 2 axial SBP applications
        # and one periodic SBP application, with no factor of two.
        lhs=float(np.sum(W*a*(Dx@Dx@v@Ds.T)))
        bulk=-(Ds@Ds@np.ones(Ds.shape[0])) if False else None
        # Repeated x integration by parts and one periodic s integration:
        rhs_bulk=-float(np.sum(W*(Dx@Dx@a@Ds.T)*v))
        end=float(np.sum(ops.ws*(
            a[-1]*(Dx@v@Ds.T)[-1]-a[0]*(Dx@v@Ds.T)[0]
            -(Dx@a)[-1]*(v@Ds.T)[-1]+(Dx@a)[0]*(v@Ds.T)[0])))
        self.assertAlmostEqual(lhs,rhs_bulk+end,places=10)


if __name__=="__main__": unittest.main()
