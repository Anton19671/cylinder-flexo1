"""Regression checks for the Newmark full-residual and moving trace gate."""
from types import SimpleNamespace
import unittest
import numpy as np
from cylinder_dynamic_trace import (mechanical_trace_matrix,MechanicalTraceNewmark,
                                    effective_step_acceptance, require_zero_initial_support_motion)


class TraceNewmarkTest(unittest.TestCase):
    def make(self):
        c=np.array([[1.,0.]])
        m=MechanicalTraceNewmark(c,[0.],[0.,0.],[0.,0.],[0.,0.],
                                 t=0.,dt=.1,beta=.25,gamma=.5)
        return c,m

    def test_mechanical_extraction_ignores_electric_rows(self):
        rows=[SimpleNamespace(field="mechanical",indices=(0,1),
                              weights=(2.,-2.),prescribed=4.),
              SimpleNamespace(field="electrical",indices=(0,),
                              weights=(1.,),prescribed=8.)]
        c,d=mechanical_trace_matrix(rows,3)
        np.testing.assert_allclose(c,[[1/np.sqrt(2),-1/np.sqrt(2),0.]])
        np.testing.assert_allclose(d,[np.sqrt(2)])

    def test_no_mechanical_constraints(self):
        c,d=mechanical_trace_matrix([],3)
        self.assertEqual(c.shape,(0,3))
        m=MechanicalTraceNewmark(c,d,[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],
                                  t=0.,dt=.1,beta=.25,gamma=.5)
        p=m.preview(c,d,.1)
        self.assertEqual(m.accept(p,[.1,0,0],[2,0,0],[0,0,0]),
                         dict(displacement=0.,velocity=0.,acceleration=0.))

    def test_empty_trace_and_indices(self):
        row=SimpleNamespace(field="mechanical",indices=(0,1),
                            weights=(1.,-1.),prescribed=0.)
        with self.assertRaisesRegex(ValueError,"trace index out of range"):
            mechanical_trace_matrix([row],1)
        with self.assertRaisesRegex(ValueError,"empty mechanical"):
            mechanical_trace_matrix([SimpleNamespace(field="mechanical",
                indices=(0,),weights=(0.,),prescribed=0.)],2)

    def test_stationary_support(self):
        c,m=self.make();p=m.preview(c,[0.],.1)
        self.assertEqual(m.accept(p,[0.,3.],[0.,1.],[0.,2.]),
                         dict(displacement=0.,velocity=0.,acceleration=0.))

    def test_moving_support_predictors_and_accept(self):
        c,m=self.make()
        p=m.preview(c,[.01],.1)
        self.assertAlmostEqual(p.acceleration[0],4.)
        self.assertAlmostEqual(p.velocity[0],.2)
        m.accept(p,[.01,0],[.2,0],[4.,0])
        p=m.preview(c,[.03],.2)
        self.assertAlmostEqual(p.acceleration[0],-4.)
        self.assertAlmostEqual(p.velocity[0],.2)
        self.assertLess(m.accept(p,[.03,0],[.2,0],[-4.,0])["velocity"],1.e-12)

    def test_rejected_step_does_not_mutate_state(self):
        c,m=self.make();p=m.preview(c,[.01],.1)
        with self.assertRaisesRegex(ValueError,"velocity drift"):
            m.accept(p,[.01,0],[0.,0.],[4.,0])
        self.assertEqual(m.t,0.)
        self.assertEqual(m.d[0],0.)

    def test_changed_trace_matrix_rejected(self):
        c,m=self.make()
        with self.assertRaisesRegex(ValueError,"matrix changed"):
            m.preview(c+.01,[0.],.1)

    def test_initial_displacement_invalid(self):
        with self.assertRaisesRegex(ValueError,"initial displacement drift"):
            MechanicalTraceNewmark(np.array([[1.,0.]]),[.1],[0.,0.],
                                   [0.,0.],[0.,0.],t=0,dt=.1,beta=.25,gamma=.5)

    def test_near_zero_initial_full_residual_uses_absolute_floor(self):
        self.assertEqual(effective_step_acceptance(
            True,1.,1.e-7,absolute_residual=3.e-13,
            absolute_tolerance=1.e-12),(True,"ACCEPTED"))
        self.assertEqual(effective_step_acceptance(
            True,1.,1.e-7,absolute_residual=3.e-10,
            absolute_tolerance=1.e-12),
            (False,"FULL_EFFECTIVE_RESIDUAL_ABOVE_TOLERANCE"))

    def test_absolute_floor_cannot_overrule_nonconvergence(self):
        self.assertEqual(effective_step_acceptance(
            False,1.,1.e-7,absolute_residual=0.,absolute_tolerance=1.e-12),
            (False,"SPATIAL_SOLVE_NOT_CONVERGED"))
        self.assertEqual(effective_step_acceptance(
            True,float("nan"),1.e-7,absolute_residual=0.,absolute_tolerance=1.e-12),
            (False,"INVALID_FULL_EFFECTIVE_RESIDUAL"))
        with self.assertRaisesRegex(ValueError,"supplied together"):
            effective_step_acceptance(True,1.,1.e-7,absolute_residual=0.)

    def test_zero_initial_support_motion_preflight(self):
        from cylinder_dynamic_loads import normalize_time_law
        stationary=normalize_time_law({"kind":"SMOOTHSTEP","amplitude":1.,
            "start_time_s":0.,"duration_s":.24})
        self.assertEqual(require_zero_initial_support_motion(
            stationary,[.002,0.],0.),0.)
        moving=normalize_time_law({"kind":"SMOOTHSTEP","amplitude":1.,
            "start_time_s":-.12,"duration_s":.24})
        with self.assertRaisesRegex(ValueError,"ZERO initial conditions incompatible"):
            require_zero_initial_support_motion(moving,[.002,0.],0.)
        self.assertEqual(require_zero_initial_support_motion(
            moving,[0.,0.],0.),.5)

    def test_full_residual_gate(self):
        self.assertEqual(effective_step_acceptance(True,1.e-8,1.e-7),(True,"ACCEPTED"))
        self.assertEqual(effective_step_acceptance(False,0.,1.e-7),
                         (False,"SPATIAL_SOLVE_NOT_CONVERGED"))
        self.assertEqual(effective_step_acceptance(True,2.e-7,1.e-7),
                         (False,"FULL_EFFECTIVE_RESIDUAL_ABOVE_TOLERANCE"))
        self.assertEqual(effective_step_acceptance(True,np.inf,1.e-7),
                         (False,"INVALID_FULL_EFFECTIVE_RESIDUAL"))
        with self.assertRaises(ValueError):
            effective_step_acceptance(True,0.,0.)


class SparseTraceNewmarkTest(unittest.TestCase):
    def test_sparse_matches_dense_on_moving_support(self):
        from types import SimpleNamespace
        from cylinder_dynamic_trace import (sparse_mechanical_trace_matrix,
                                            SparseMechanicalTraceNewmark)
        rows=[SimpleNamespace(field="mechanical",indices=(0,3),
            weights=(2.,-2.),prescribed=0.),
            SimpleNamespace(field="electrical",indices=(1,),
            weights=(1.,),prescribed=3.)]
        cd,dd=mechanical_trace_matrix(rows,6)
        cs,ds=sparse_mechanical_trace_matrix(rows,6)
        np.testing.assert_allclose(cs.toarray(),cd)
        np.testing.assert_allclose(ds,dd)
        zero=np.zeros(6)
        dense=MechanicalTraceNewmark(cd,dd,zero,zero,zero,
            t=0.,dt=.1,beta=.25,gamma=.5)
        sparse=SparseMechanicalTraceNewmark(cs,ds,zero,zero,zero,
            t=0.,dt=.1,beta=.25,gamma=.5)
        new_target=np.array([.01])
        pd=dense.preview(cd,new_target,.1)
        ps=sparse.preview(cs,new_target,.1)
        np.testing.assert_allclose(pd.velocity,ps.velocity)
        np.testing.assert_allclose(pd.acceleration,ps.acceleration)
        q=np.zeros(6);q[0]=.01*np.sqrt(2)
        v=np.zeros(6);v[0]=.2*np.sqrt(2)
        a=np.zeros(6);a[0]=4*np.sqrt(2)
        np.testing.assert_allclose(
            list(dense.accept(pd,q,v,a).values()),
            list(sparse.accept(ps,q,v,a).values()))

    def test_sparse_rejected_step_does_not_advance(self):
        from scipy import sparse
        from cylinder_dynamic_trace import SparseMechanicalTraceNewmark
        c=sparse.csr_matrix([[1.,0.,0.]])
        z=np.zeros(3)
        monitor=SparseMechanicalTraceNewmark(c,[0.],z,z,z,
            t=0.,dt=.1,beta=.25,gamma=.5)
        step=monitor.preview(c,[.01],.1)
        with self.assertRaisesRegex(ValueError,"velocity drift"):
            monitor.accept(step,[.01,0.,0.],z,[4.,0.,0.])
        self.assertEqual(monitor.t,0.)
        with self.assertRaisesRegex(ValueError,"matrix changed"):
            monitor.preview(sparse.csr_matrix([[1.,.01,0.]]),[0.],.1)

    def test_sparse_large_state_without_dense_matrix(self):
        from types import SimpleNamespace
        from cylinder_dynamic_trace import (sparse_mechanical_trace_matrix,
                                            SparseMechanicalTraceNewmark)
        n=30000
        rows=[SimpleNamespace(field="mechanical",
            indices=(0,n-1),weights=(1.,-1.),prescribed=0.)]
        c,d=sparse_mechanical_trace_matrix(rows,n)
        self.assertEqual(c.nnz,2)
        self.assertEqual(c.shape,(1,n))
        z=np.zeros(n)
        monitor=SparseMechanicalTraceNewmark(c,d,z,z,z,
            t=0.,dt=.1,beta=.25,gamma=.5)
        target=monitor.preview(c,d,.1)
        self.assertEqual(monitor.accept(target,z,z,z)["displacement"],0.)


if __name__=="__main__":unittest.main()
