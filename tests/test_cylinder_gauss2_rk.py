"""Manufactured Gauss-IRK4 stage tests; no cylindrical shell integration."""
import math
import unittest
import numpy as np
from cylinder_gauss2_rk import (GAUSS2_A,GAUSS2_B,GAUSS2_C,
    Gauss2Settings,Gauss2State,gauss2_step,gauss2_stage_kinematics)

class Gauss2Tests(unittest.TestCase):
    def config(self,dt):
        return Gauss2Settings(dt,max_unknowns=32,scaled_equation_tolerance=2e-11)

    def test_gauss_tableau_order(self):
        np.testing.assert_allclose(GAUSS2_A.sum(axis=1),GAUSS2_C)
        self.assertAlmostEqual(float(GAUSS2_B@GAUSS2_C),.5)
        self.assertAlmostEqual(float(GAUSS2_B@(GAUSS2_C**2)),1./3.)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(GAUSS2_B[i]*GAUSS2_A[i,j]+
                    GAUSS2_B[j]*GAUSS2_A[j,i],GAUSS2_B[i]*GAUSS2_B[j])

    def test_constant_acceleration(self):
        q=np.full((1,2,2),.3);v=np.full_like(q,.5)
        aa=np.full((2,)+q.shape,2.)
        Q,V=gauss2_stage_kinematics(q,v,aa,.12)
        for i,c in enumerate(GAUSS2_C):
            np.testing.assert_allclose(Q[i],q+.12*c*v+(.12*c)**2)
            np.testing.assert_allclose(V[i],v+2*.12*c)
        result=gauss2_step(Gauss2State(0.,q,v),self.config(.12),
            equations=lambda q,v,a,p,t:(a-2,None),
            mechanical_state_scale=1.,mechanical_equation_scale=1.)
        np.testing.assert_allclose(result.final.mechanical,q+.12*v+.12**2,atol=1e-13)

    def test_oscillator_fourth_order(self):
        def integrate(n):
            dt=1./n
            state=Gauss2State(0.,np.ones((1,1,1)),np.zeros((1,1,1)))
            for _ in range(n):
                state=gauss2_step(state,self.config(dt),
                    equations=lambda q,v,a,p,t:(a+4*q,None),
                    mechanical_state_scale=1.,mechanical_equation_scale=4.).final
            return float(state.mechanical[0,0,0]),float(state.velocity[0,0,0])
        target=(math.cos(2),-2*math.sin(2))
        coarse=integrate(10);fine=integrate(20)
        e0=max(abs(a-b) for a,b in zip(coarse,target))
        e1=max(abs(a-b) for a,b in zip(fine,target))
        self.assertGreater(e0/e1,12.)
        self.assertLess(e1,1e-6)

    def test_two_way_electric_constraint_at_both_stages(self):
        q=np.ones((1,2,2));v=np.zeros_like(q)
        def eq(q,v,a,p,t):
            return a+1.2*q+.5*p,p-.4*q
        result=gauss2_step(Gauss2State(0.,q,v,.4*q),self.config(.1),
            equations=eq,mechanical_state_scale=1.,mechanical_equation_scale=2.,
            electrical_state_scale=1.,electrical_equation_scale=1.,
            end_electrical_solver=lambda q,v,t,p:.4*q)
        for s in result.stages:
            np.testing.assert_allclose(s.electrical,.4*s.mechanical,atol=1e-11)
            np.testing.assert_allclose(s.acceleration,-1.4*s.mechanical,atol=1e-11)
        np.testing.assert_allclose(result.final.electrical,.4*result.final.mechanical)

    def test_moving_essential_trace_at_stage_times(self):
        q=np.zeros((1,1,1))
        result=gauss2_step(Gauss2State(0.,q,q),self.config(.1),
            equations=lambda q,v,a,p,t:(q-t*t,None),
            mechanical_state_scale=.1,mechanical_equation_scale=.01)
        for s in result.stages:
            np.testing.assert_allclose(s.mechanical,s.time_s**2,atol=1e-12)
        np.testing.assert_allclose(result.final.mechanical,.01,atol=1e-12)

    def test_electric_end_requires_algebraic_solution(self):
        q=np.ones((1,1,1));z=np.zeros_like(q)
        with self.assertRaisesRegex(ValueError,"endpoint"):
            gauss2_step(Gauss2State(0.,q,z,q),self.config(.1),
                equations=lambda q,v,a,p,t:(a+q,p-q),
                mechanical_state_scale=1.,mechanical_equation_scale=1.,
                electrical_state_scale=1.,electrical_equation_scale=1.)

    def test_invalid_shapes_and_dof_limit(self):
        q=np.ones((1,1,1));z=np.zeros_like(q)
        with self.assertRaisesRegex(ValueError,"invalid mechanical stage"):
            gauss2_step(Gauss2State(0.,q,z),self.config(.1),
                equations=lambda q,v,a,p,t:(np.ones((2,1,1)),None),
                mechanical_state_scale=1.,mechanical_equation_scale=1.)
        with self.assertRaisesRegex(ValueError,"UNKNOWN_LIMIT"):
            gauss2_step(Gauss2State(0.,q,z),Gauss2Settings(.1,max_unknowns=1),
                equations=lambda q,v,a,p,t:(a,None),
                mechanical_state_scale=1.,mechanical_equation_scale=1.)

if __name__=="__main__":unittest.main()
