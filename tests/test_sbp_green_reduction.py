"""Deferred SBP weak/strong/boundary identity checks (not run in assembly phase)."""
import unittest
import numpy as np
from cylinder_functional_assembly import (
    surface_sbp21,CylinderFunctional,MECHANICAL_DERIVATIVES)
from cylinder_sbp_green_reduction import split_coupled_sbp_residual
from flexo_gradient_g3d import G3DMaterial


class SBPGreenReductionTests(unittest.TestCase):
    def setUp(self):
        self.ops=surface_sbp21(5,7,1.2,2.)
        self.rng=np.random.default_rng(918)

    def test_all_mixed_jet_orders_and_electricity(self):
        nx,ns=self.ops.shape
        for order_count in (6,10):
            mc=self.rng.normal(size=(order_count,3,nx,ns))
            ec=self.rng.normal(size=(3,2,nx,ns))
            split=split_coupled_sbp_residual(
                self.ops,mc,MECHANICAL_DERIVATIVES[:order_count],ec)
            np.testing.assert_allclose(split.mechanical.reconstructed,
                split.mechanical.direct_weak,atol=2e-9,rtol=2e-10)
            np.testing.assert_allclose(split.electrical.reconstructed,
                split.electrical.direct_weak,atol=2e-9,rtol=2e-10)

    def test_all_seven_models_same_conservative_variation(self):
        material=G3DMaterial(100.,.25,.08,.001,-.002,1.5)
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                nquad=2*int(theory[-1])+1 if theory.startswith("CUF_TE") else 7
                f=CylinderFunctional(theory,material,radius_m=2.,thickness_m=.2,
                    surface_operators=self.ops,n_gauss=nquad)
                q=self.rng.normal(0.,1.e-5,(f.n_mechanical,*self.ops.shape))
                p=self.rng.normal(0.,1.e-4,(2,*self.ops.shape))
                reduction=f.sbp_green_reduction(q,p)
                bulk=f.evaluate(q,p,energy_breakdown=False)
                np.testing.assert_allclose(reduction.mechanical.direct_weak,
                    bulk.mechanical_residual,atol=1e-8,rtol=1e-9)
                np.testing.assert_allclose(reduction.electrical.direct_weak,
                    bulk.electrical_residual,atol=1e-8,rtol=1e-9)

    def test_reject_legacy_non_sbp_pair(self):
        from cylinder_functional_assembly import surface_trapezoid
        ops=surface_trapezoid(5,7,1.2,2.)
        c=np.zeros((6,1,*ops.shape))
        with self.assertRaisesRegex(ValueError,"SBP_GREEN_IDENTITY_VIOLATED"):
            split_coupled_sbp_residual(ops,c,MECHANICAL_DERIVATIVES[:6])


if __name__=="__main__":unittest.main()
