"""No-SVD exact reduced tangent checks, including the electric saddle case."""
import unittest
from unittest.mock import patch
import numpy as np
from cylinder_variational_fdm import solve_checked_reduced_tangent


class CheckedLUTests(unittest.TestCase):
    def test_indefinite_electric_saddle(self):
        a=np.array([[4.,1.,2.],[1.,-3.,.5],[2.,.5,5.]])
        g=np.array([.25,-.5,.75])
        with patch("numpy.linalg.svd",side_effect=AssertionError("SVD forbidden")):
            update,condition=solve_checked_reduced_tangent(a,g)
        np.testing.assert_allclose(a@update,-g,atol=1e-13)
        self.assertTrue(np.isfinite(condition))

    def test_singular_rejected_without_regularization(self):
        a=np.array([[1.,2.],[2.,4.]])
        with patch("numpy.linalg.svd",side_effect=AssertionError("SVD forbidden")):
            with self.assertRaises(np.linalg.LinAlgError):
                solve_checked_reduced_tangent(a,np.ones(2))

    def test_near_singular_rejected(self):
        a=np.diag([1.,1.e-18])
        with self.assertRaises(np.linalg.LinAlgError):
            solve_checked_reduced_tangent(a,np.ones(2))


if __name__=="__main__":
    unittest.main()
