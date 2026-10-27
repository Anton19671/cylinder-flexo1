"""No spurious temporal separation in the rank-fixed MVI/Newmark contract.

Algebraic manufactured fields only. No cylindrical-shell solve is run.
"""
import unittest
import numpy as np

from cylinder_newmark_audit import audit_newmark_step
from cylinder_variational_mvi import reconstruct_profiles


def numerical_rank(field, atol=1.e-11):
    return int(np.linalg.matrix_rank(field, tol=atol))


class SpatialMviNewmarkContractTests(unittest.TestCase):
    def test_rank_one_displacement_can_have_rank_two_velocity(self):
        # q(t)=X(t) Y(t) is rank one at every instant, but q_dot
        # and q_ddot have mixed product-rule terms.
        x=np.array([1.,2.,4.,8.])
        y=np.array([1.,3.,2.,5.])
        xd=np.array([2.,-1.,3.,1.])
        yd=np.array([3.,1.,-2.,4.])
        q=np.outer(x,y)
        velocity=np.outer(xd,y)+np.outer(x,yd)
        acceleration=2*np.outer(xd,yd)
        self.assertEqual(numerical_rank(q),1)
        self.assertEqual(numerical_rank(velocity),2)
        self.assertEqual(numerical_rank(acceleration),1)
        # The mixed product term must not be dropped.
        self.assertGreater(np.linalg.norm(acceleration),0.)

    def test_newmark_predictor_not_restricted_to_mvi_rank(self):
        x=np.array([1.,2.,4.,7.])
        y=np.array([2.,3.,5.,1.])
        xd=np.array([1.,-2.,1.,3.])
        yd=np.array([3.,-1.,1.,4.])
        q0=np.outer(x,y)[None,...]
        v0=(np.outer(xd,y)+np.outer(x,yd))[None,...]
        # The initial acceleration is an independent Newmark field;
        # it need not have the same spatial rank as endpoint displacement.
        a0=(2*np.outer(xd,yd)+.3*np.outer(x,y))[None,...]
        dt=.08
        beta=.25
        gamma=.5
        q1=np.outer(x+dt*xd,y+dt*yd)[None,...]
        a1=(q1-q0-dt*v0-dt**2*(.5-beta)*a0)/(beta*dt**2)
        v1=v0+dt*((1-gamma)*a0+gamma*a1)
        self.assertEqual(numerical_rank(q0[0]),1)
        self.assertEqual(numerical_rank(q1[0]),1)
        predictor=q0+dt*v0+dt**2*(.5-beta)*a0
        self.assertGreater(numerical_rank(predictor[0]),1)
        audit_newmark_step(q0,v0,a0,q1,v1,a1,
                           dt_s=dt,beta=beta,gamma=gamma)
        # The saved field is represented by rank-one X/Y, while
        # the corresponding Newmark acceleration remains unrestricted.
        x_profiles=np.asarray((x+dt*xd))[None,None,:]
        y_profiles=np.asarray((y+dt*yd))[None,None,:]
        np.testing.assert_allclose(reconstruct_profiles(x_profiles,y_profiles),q1)
        self.assertGreater(numerical_rank(a1[0]),1)

    def test_electric_modes_are_not_part_of_newmark_acceleration(self):
        q=np.zeros((3,4,4))
        v=np.zeros_like(q)
        a=np.zeros_like(q)
        psi=np.ones((2,4,4))
        audit_newmark_step(q,v,a,q,v,a,dt_s=.1,beta=.25,gamma=.5)
        self.assertNotEqual(psi.shape,a.shape)


if __name__=="__main__":
    unittest.main()
