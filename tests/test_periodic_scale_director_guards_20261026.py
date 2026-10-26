"""Regression checks for periodic Nyquist, MVI scaling and G3D director jets."""
import unittest
import numpy as np
from cylinder_te1_stage3a_connector import (
    derivative_matrix_periodic_uniform, CylinderSpatialOperators)
from cylinder_te1_mvi_core import CylinderGrid
from cylinder_kl_stage8b5_strict_mvi import _fixed_projected_scale, KLMVIError
from cylinder_director_g3d import _displacement


class PeriodicAndDirectorGuards(unittest.TestCase):
    def test_even_grid_central_first_derivative_rejected_at_source(self):
        for n in (8,10,12):
            with self.subTest(ns=n), self.assertRaisesRegex(ValueError,
                    "CENTRAL_PERIODIC_DERIVATIVE_REQUIRES_ODD_NS"):
                derivative_matrix_periodic_uniform(2*np.pi,n,1,7)
        with self.assertRaisesRegex(ValueError,
                "CENTRAL_PERIODIC_DERIVATIVE_REQUIRES_ODD_NS"):
            CylinderSpatialOperators.build(CylinderGrid(1.,2.,7,8),7)

    def test_odd_grid_derivative_no_checkerboard_null(self):
        d=derivative_matrix_periodic_uniform(2*np.pi,9,1,7)
        self.assertEqual(d.shape,(9,9))
        self.assertLess(np.max(np.abs(d@np.ones(9))),1e-12)

    def test_zero_projected_scale_rejected_without_subnormal_floor(self):
        class F:
            mvi_projected_reference_scale=0.
        with self.assertRaisesRegex(KLMVIError,
                "ZERO_MVI_PROJECTED_REFERENCE_SCALE"):
            _fixed_projected_scale(None,None,None,F())

    def test_invalid_director_order_has_clear_error(self):
        with self.assertRaisesRegex(ValueError,"surface order"):
            _displacement(None,"FSDT",0.,2.,.2,dx=3)
        with self.assertRaisesRegex(ValueError,"surface order"):
            _displacement(None,"REDDY",0.,2.,.2,dx=-1)


if __name__=="__main__":
    unittest.main()
