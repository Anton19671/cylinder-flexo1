"""Guards for small-amplitude physical modes in fixed-rank Gauss MVI."""
import unittest
from types import SimpleNamespace
import numpy as np
from cylinder_gauss2_mvi import _rank_factors, _profile_field, Gauss2MVISettings, solve_gauss2_mvi_step
from cylinder_gauss2_rk import Gauss2Settings, Gauss2State

def free_constraints(stage,t,field):
    return np.zeros((0,field.size)),np.zeros(0)

class SmallModeGuards(unittest.TestCase):
    def test_weak_mode_outside_rank_refused(self):
        first=np.outer(np.array([1.,2.,3.,4.]),np.array([2.,1.,4.,3.]))
        second=np.outer(np.array([1.,-1.,1.,-1.]),np.array([1.,-1.,-1.,1.]))
        field=1e-12*(first+1e-4*second)
        with self.assertRaisesRegex(ValueError,"OUTSIDE_FIXED_RANK"):
            _rank_factors(field[None],1,2e-10,"weak")
        x,y=_rank_factors(field[None],2,2e-10,"weak")
        np.testing.assert_allclose(_profile_field(x,y),field[None],rtol=1e-10,atol=1e-25)

    def test_exact_tiny_rank_one_accepted(self):
        z=1e-24*np.outer(np.array([1.,2.,3.,4.]),np.array([2.,1.,3.,5.]))
        x,y=_rank_factors(z[None],1,2e-10,"tiny")
        np.testing.assert_allclose(_profile_field(x,y),z[None],rtol=1e-12,atol=1e-36)

    def test_continuation_checks_old_component_scales(self):
        x=np.ones((1,1,4));y=np.ones((1,1,4))
        q=np.ones((1,4,4));z=np.zeros_like(q)
        prev=SimpleNamespace(
            converged=True,final=Gauss2State(0.,q,z),
            x_profiles=x,y_profiles=y,
            scales=np.broadcast_to(np.arange(1.,5.)[:,None],(4,4))[None].copy())
        with self.assertRaisesRegex(ValueError,"INVALID_OLD_SCALES"):
            solve_gauss2_mvi_step(prev.final,Gauss2Settings(.1),
                mvi_settings=Gauss2MVISettings(rank=1),
                equations=lambda q,v,a,p,t:(a,None),
                constraints=free_constraints,state_scales=1.,equation_scales=1.,
                previous=prev)

if __name__=="__main__":
    unittest.main()
