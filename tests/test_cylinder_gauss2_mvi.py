"""Manufactured fixed-rank Gauss2 X/Y collocation without shell simulations."""
import unittest
import numpy as np
from cylinder_gauss2_rk import Gauss2Settings,Gauss2State,GAUSS2_C
from cylinder_gauss2_mvi import (Gauss2MVISettings,solve_gauss2_mvi_step,
                                  _rank_factors, _profile_field)


def free_constraints(stage,t,field):
    return np.zeros((0,field.size)), np.empty(0)


class Gauss2MviTests(unittest.TestCase):
    def initial(self):
        x=np.array([1.,2.,3.,4.])
        y=np.array([2.,-1.,3.,1.])
        q=np.outer(x,y)[None,...]
        v=np.zeros_like(q)
        return Gauss2State(0.,q,v)

    def mvi(self,rank=1):
        return Gauss2MVISettings(rank=rank,max_sweeps=15,
            max_newton_iterations=10,absolute_tolerance=2.e-11,
            relative_tolerance=1.e-9,field_change_tolerance=1.e-10)

    def test_linear_oscillator_mvi_one_step_and_repeated_continuation(self):
        state=self.initial(); prev=None
        for step in range(2):
            result=solve_gauss2_mvi_step(
                state,Gauss2Settings(.05),mvi_settings=self.mvi(),
                equations=lambda q,v,a,p,t:(a+4*q,None),
                constraints=free_constraints,state_scales=1.,equation_scales=1.,
                previous=prev)
            self.assertTrue(result.converged)
            self.assertEqual(result.x_profiles.shape,(1,1,4))
            np.testing.assert_allclose(_profile_field(result.x_profiles,result.y_profiles)
                                       *result.scales,result.final.mechanical,atol=1e-10)
            time=result.final.time_s
            np.testing.assert_allclose(result.final.mechanical,
                self.initial().mechanical*np.cos(2*time),atol=5e-5)
            self.assertTrue(np.all(np.isfinite(result.final.velocity)))
            state,prev=result.final,result

    def test_coupled_electric_stages_and_endpoint(self):
        state=self.initial()
        p0=.4*state.mechanical
        state=Gauss2State(0.,state.mechanical,state.velocity,p0)
        count=[]
        def equations(q,v,a,p,t):
            count.append(t)
            return a+1.2*q+.5*p, p-.4*q
        result=solve_gauss2_mvi_step(
            state,Gauss2Settings(.07),mvi_settings=self.mvi(),
            equations=equations,constraints=free_constraints,
            state_scales=np.array([1.,1.]),equation_scales=1.,
            end_electrical_solver=lambda q,v,t,p,r:.4*q)
        self.assertEqual(result.x_profiles.shape,(2,1,4))
        for stage in result.stages:
            np.testing.assert_allclose(stage.electrical,.4*stage.mechanical,atol=1e-9)
        np.testing.assert_allclose(result.final.electrical,.4*result.final.mechanical,atol=1e-9)
        self.assertTrue(np.any(np.isclose(count,GAUSS2_C[0]*.07)))
        self.assertTrue(np.any(np.isclose(count,GAUSS2_C[1]*.07)))

    def test_initial_zero_stage_seeding_driven_by_spatial_force(self):
        z=np.zeros((1,4,4))
        spatial=np.outer(np.array([1.,2.,3.,4.]),np.array([4.,1.,3.,2.]))[None]
        result=solve_gauss2_mvi_step(Gauss2State(0.,z,z),Gauss2Settings(.05),
            mvi_settings=self.mvi(),
            equations=lambda q,v,a,p,t:(a+3*q-spatial,None),
            constraints=free_constraints,state_scales=1.,equation_scales=1.)
        self.assertGreater(np.max(np.abs(result.final.mechanical)),0.)
        self.assertEqual(np.linalg.matrix_rank(result.final.mechanical[0]),1)

    def test_unsupported_rank_in_endpoint_rejected(self):
        z=np.zeros((1,4,4))
        a=np.outer(np.array([1.,2.,3.,4.]),np.array([1.,1.,1.,1.]))
        b=np.outer(np.array([1.,1.,1.,1.]),np.array([1.,2.,3.,4.]))
        def stage_target(t):
            return a if t < .05 else b
        with self.assertRaisesRegex(ValueError,'ENDPOINT.*OUTSIDE_FIXED_RANK'):
            solve_gauss2_mvi_step(Gauss2State(0.,z,z),Gauss2Settings(.1),
                mvi_settings=self.mvi(),
                equations=lambda q,v,acc,p,t:(q-stage_target(t),None),
                constraints=free_constraints,state_scales=1.,equation_scales=1.)

    def test_all_ranks_exact_factorization_no_truncation(self):
        rng=np.random.default_rng(5)
        for rank in range(1,9):
            field=(rng.normal(size=(9,rank))@rng.normal(size=(rank,9)))[None]
            x,y=_rank_factors(field,rank,1e-11,'test')
            np.testing.assert_allclose(_profile_field(x,y),field,atol=1e-11)
            if rank<8:
                with self.assertRaisesRegex(ValueError,'OUTSIDE_FIXED_RANK'):
                    _rank_factors(field+np.eye(9)[None]*.1,rank,1e-11,'test')

    def test_electric_endpoint_mandatory(self):
        base=self.initial()
        state=Gauss2State(base.time_s,base.mechanical,base.velocity,
                           .2*base.mechanical)
        with self.assertRaisesRegex(ValueError,'ELECTRIC_ENDPOINT_SOLVER_REQUIRED'):
            solve_gauss2_mvi_step(state,Gauss2Settings(.05),
                mvi_settings=self.mvi(),
                equations=lambda q,v,a,p,t:(a+q,p-.2*q),
                constraints=free_constraints,state_scales=1.,equation_scales=1.)


if __name__=='__main__':unittest.main()
