"""Manufactured coupled strong Newton tests; no full cylindrical shell runs."""
import unittest
import numpy as np
from cylinder_direct_strong_newton import StrongNewtonSettings,solve_strong_nonlinear_equations
from cylinder_direct_strong_fdm import DirectStrongGrid,direct_coefficient_derivative

class DirectStrongNewtonTests(unittest.TestCase):
    def test_tangent_probe_respects_deformation_domain(self):
        q=np.full((1,7,4),.9999)
        def equations(q,p):
            if np.any(q >= 1.):
                raise ValueError('INVALID_DEFORMATION_MAPPING: detF<=0')
            return q-.5,None
        result=solve_strong_nonlinear_equations(
            equations,q,mechanical_state_scale=1.,mechanical_equation_scale=1.,
            settings=StrongNewtonSettings(max_dofs=28,difference_step=.001,
                                         line_search_halvings=15))
        self.assertTrue(result.converged,result.status)
        np.testing.assert_allclose(result.mechanical,.5,atol=1.e-8)

    def test_tangent_probe_failure_is_reported(self):
        q=np.full((1,7,4),.9999)
        def equations(q,p):
            if np.any(q >= 1.):
                raise ValueError('INVALID_DEFORMATION_MAPPING: detF<=0')
            return q-.5,None
        result=solve_strong_nonlinear_equations(
            equations,q,mechanical_state_scale=1.,mechanical_equation_scale=1.,
            settings=StrongNewtonSettings(max_dofs=28,difference_step=.001,
                                         line_search_halvings=0))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,'INVALID_TANGENT_PERTURBATION')

    def test_manufactured_coupled_second_order(self):
        grid=DirectStrongGrid(9,4,1.6,2.)
        x=np.arange(grid.nx)[:,None]*grid.dx
        qtrue=np.broadcast_to(x*(grid.length_m-x),(grid.nx,grid.ns))
        ptrue=2*qtrue
        def equations(q,p):
            qx=direct_coefficient_derivative(q,1,0,grid)
            px=direct_coefficient_derivative(p,1,0,grid)
            rm=.12*p-direct_coefficient_derivative(qx,1,0,grid)-(.12*ptrue+2)
            re=.12*q-direct_coefficient_derivative(px,1,0,grid)-(.12*qtrue+4)
            for arr,state in ((rm,q),(re,p)):
                arr[:,0,:]=state[:,0,:]
                arr[:,-1,:]=state[:,-1,:]
            return rm,re
        z=np.zeros((1,grid.nx,grid.ns))
        result=solve_strong_nonlinear_equations(equations,z,z,
            mechanical_state_scale=1.,electrical_state_scale=1.,
            mechanical_equation_scale=1.,electrical_equation_scale=1.,
            settings=StrongNewtonSettings(max_dofs=80,max_iterations=7,
                relative_linear_tolerance=1e-9,scaled_absolute_tolerance=2e-7))
        self.assertTrue(result.converged,result.status)
        np.testing.assert_allclose(result.mechanical[0],qtrue,atol=8e-5)
        np.testing.assert_allclose(result.electrical[0],ptrue,atol=8e-5)

    def test_nonlinear_two_way_coupling(self):
        q=np.full((1,7,4),.2)
        p=np.full_like(q,.1)
        result=solve_strong_nonlinear_equations(
            lambda q,p:(q+q**3+.1*p-1.,p+.2*q-.8),q,p,
            mechanical_state_scale=1.,electrical_state_scale=1.,
            mechanical_equation_scale=1.,electrical_equation_scale=1.,
            settings=StrongNewtonSettings(max_dofs=56,max_iterations=7))
        self.assertTrue(result.converged,result.status)

    def test_invalid_scales_and_size(self):
        q=np.zeros((1,7,4))
        with self.assertRaises(ValueError):
            solve_strong_nonlinear_equations(lambda q,p:(q,None),q,q,
                mechanical_state_scale=1.,mechanical_equation_scale=1.)
        with self.assertRaisesRegex(ValueError,"DOF_LIMIT"):
            solve_strong_nonlinear_equations(lambda q,p:(q,None),q,
                mechanical_state_scale=1.,mechanical_equation_scale=1.,
                settings=StrongNewtonSettings(max_dofs=10))

    def test_singular_tangent_rejected(self):
        q=np.zeros((1,7,4))
        result=solve_strong_nonlinear_equations(
            lambda q,p:(np.ones_like(q),None),q,
            mechanical_state_scale=1.,mechanical_equation_scale=1.,
            settings=StrongNewtonSettings(max_dofs=28,max_iterations=2))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"LINEAR_TANGENT_NOT_SOLVED")
