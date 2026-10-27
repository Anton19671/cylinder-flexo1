"""Gauss collocation: stage rank and endpoint rank are different obligations.

An algebraic manufactured test, not an actual shell run. It prevents
silently claiming that rank-R Gauss stages imply a rank-R endpoint.
"""
import unittest
import numpy as np
from cylinder_gauss2_rk import GAUSS2_A, GAUSS2_B, gauss2_stage_kinematics


class Gauss2FixedRankEndpointTests(unittest.TestCase):
    def test_two_rank_one_stages_can_produce_rank_two_endpoint(self):
        a=np.array([1.,0.,2.,3.]); b=np.array([1.,3.,2.,4.])
        c=np.array([0.,2.,1.,3.]); d=np.array([2.,1.,4.,1.])
        stages=np.stack((np.outer(a,b),np.outer(c,d)))[:,None]
        dt=.08
        # Construct exact stage accelerations via the invertible Gauss A^2.
        aa=np.einsum('ij,j...->i...',np.linalg.inv(GAUSS2_A@GAUSS2_A),stages)/dt**2
        q0=np.zeros_like(stages[0]);v0=np.zeros_like(q0)
        qstage,vstage=gauss2_stage_kinematics(q0,v0,aa,dt)
        np.testing.assert_allclose(qstage,stages,rtol=1e-12,atol=1e-12)
        self.assertEqual(int(np.linalg.matrix_rank(qstage[0,0])),1)
        self.assertEqual(int(np.linalg.matrix_rank(qstage[1,0])),1)
        qend=q0+dt*np.einsum('i,i...->...',GAUSS2_B,vstage)
        self.assertEqual(int(np.linalg.matrix_rank(qend[0],tol=1e-9)),2)

    def test_full_velocity_and_acceleration_are_not_rank_locked(self):
        q=np.zeros((1,4,4))
        aa=np.stack((np.eye(4)[None],np.fliplr(np.eye(4))[None]))
        _,vs=gauss2_stage_kinematics(q,q,aa,.1)
        self.assertEqual(vs.shape,(2,1,4,4))
        self.assertTrue(np.isfinite(vs).all())
        self.assertGreater(int(np.linalg.matrix_rank(vs[0,0])),1)


if __name__=='__main__': unittest.main()
