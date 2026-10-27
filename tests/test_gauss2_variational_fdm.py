"""Small manufactured variational-shell adapters for Gauss IRK4 (no shell run)."""
import unittest
from types import SimpleNamespace
import numpy as np

from cylinder_boundary_spatial import affine_trace_space
from cylinder_gauss2_rk import Gauss2Settings,Gauss2State
from cylinder_gauss2_variational_fdm import solve_gauss2_variational_fdm_step


class ToyVariationalProblem:
    kind="DYNAMIC"
    def __init__(self,electric=False,moving=False):
        self.electric=electric
        self.moving=moving
        self.functional=SimpleNamespace(
            h=1.,W=np.ones((1,1)),theory="CUF_TE1",
            material=SimpleNamespace(young_pa=1.),
            electrical_on=electric,ops=SimpleNamespace(shape=(1,1)))
        self.seen=[]
    def assemble(self,q,p=None,*,time_s,dead_load=None,free_charge=None,
                 acceleration=None,density_kg_m3=None,velocity=None,**kwargs):
        self.seen.append((time_s,acceleration is not None))
        val=float(q[0,0,0])
        n=q.size+(0 if p is None else p.size)
        c=np.zeros((1,n));c[0,0]=1.
        d=np.array([time_s*time_s if self.moving else 0.])
        state=np.concatenate((q.ravel(),p.ravel())) if p is not None else q.ravel()
        space=affine_trace_space(c,d,state)
        m=np.zeros_like(q)
        if acceleration is not None:
            m=acceleration+q
            if p is not None:m=m+.25*p
        elec=None if p is None else p-.5*q
        full=np.concatenate((m.ravel(),elec.ravel())) if p is not None else m.ravel()
        return SimpleNamespace(space=space,full_residual=full)
    def evaluate_effective_residual(self,q,p=None,*,time_s,**kwargs):
        return (0.,np.concatenate((np.zeros(q.size), (p-.5*q).ravel())),None)


class Gauss2PhysicalAdapterTests(unittest.TestCase):
    def test_stationary_support_and_stage_physics(self):
        pr=ToyVariationalProblem()
        z=np.zeros((1,1,1))
        out=solve_gauss2_variational_fdm_step(pr,
            Gauss2State(0.,z,z),Gauss2Settings(.1,max_unknowns=5),
            density_kg_m3=1.,loads_at=lambda t:(None,None))
        np.testing.assert_allclose(out.final.mechanical,z,atol=1e-12)
        np.testing.assert_allclose(out.final.velocity,z,atol=1e-12)
        self.assertTrue(any(has_acc for _,has_acc in pr.seen))

    def test_moving_support_refused_before_stage_solve(self):
        pr=ToyVariationalProblem(moving=True)
        z=np.zeros((1,1,1))
        with self.assertRaisesRegex(ValueError,"MOVING_MECHANICAL_SUPPORT"):
            solve_gauss2_variational_fdm_step(pr,
                Gauss2State(0.,z,z),Gauss2Settings(.1,max_unknowns=5),
                density_kg_m3=1.,loads_at=lambda t:(None,None))

    def test_electric_mode_is_solved_at_stages_and_endpoint(self):
        pr=ToyVariationalProblem(electric=True)
        z=np.zeros((1,1,1))
        out=solve_gauss2_variational_fdm_step(pr,
            Gauss2State(0.,z,z,z),Gauss2Settings(.1,max_unknowns=6),
            density_kg_m3=1.,loads_at=lambda t:(None,None))
        self.assertIsNotNone(out.final.electrical)
        for stage in out.stages:
            np.testing.assert_allclose(stage.electrical,.5*stage.mechanical,atol=1e-10)
        np.testing.assert_allclose(out.final.electrical,.5*out.final.mechanical,atol=1e-10)

if __name__=="__main__":
    unittest.main()
