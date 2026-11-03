"""Gauss IRK4 MVI: reject roundoff trace ranks, not genuine clamped constraints."""
import unittest
import numpy as np
from cylinder_gauss2_mvi import _constrained_directions


class Gauss2TraceRankTests(unittest.TestCase):
    def test_roundoff_only_trace_does_not_destroy_admissible_tangent(self):
        base=np.zeros(4)
        t=np.eye(4)[:, :2]  # already-orthonormal frozen MVI directions
        c=np.array([[1.,0.,0.,0.],[0.,1.e-14,1.,0.]])
        d=np.zeros(2)
        origin,basis=_constrained_directions(base,t,c,d)
        np.testing.assert_allclose(origin,base,atol=1.e-14)
        self.assertEqual(basis.shape,(4,1))
        self.assertLess(np.max(abs(c@basis)),1.e-12)
        self.assertGreater(abs(basis[1,0]),.99)
    def test_real_trace_constraint_is_not_discarded(self):
        base=np.zeros(4)
        t=np.eye(4)[:, :2]
        c=np.array([[1.,0.,0.,0.],[0.,1.e-5,1.,0.]])
        origin,basis=_constrained_directions(base,t,c,np.zeros(2))
        self.assertEqual(basis.shape,(4,0))
        self.assertLess(np.max(abs(c@origin)),1.e-12)
    def test_inconsistent_real_trace_still_rejected(self):
        base=np.zeros(4)
        t=np.eye(4)[:, :2]
        c=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]])
        with self.assertRaisesRegex(ValueError,"TRACE_NOT_REPRESENTABLE"):
            _constrained_directions(base,t,c,np.array([0.,1.]))


if __name__=="__main__":
    unittest.main()
