"""Dense Newton DOF limits apply to MVI substeps, not full grid."""
import unittest
from types import SimpleNamespace
import numpy as np
from cylinder_variational_dynamic import solve_dynamic_mvi,solve_dynamic_fdm
from cylinder_variational_fdm import StaticFDMSettings
from cylinder_variational_mvi import StaticMVISettings
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_functional_assembly import surface_trapezoid
from flexo_gradient_g3d import G3DMaterial

class QuadraticSurface:
    theory="CUF_TE1"
    n_mechanical=3
    n_derivatives=6
    R=2.
    h=.2
    electrical_on=True
    def __init__(self,n=7):
        self.ops=surface_trapezoid(n,n,1.2,2.)
        self.W=self.ops.area_weights
        self.material=G3DMaterial(100.,.25,.08,.01,-.02,1.5)
    def evaluate(self,q,p,*,dead_load=None,free_charge=None,**kwargs):
        force=np.zeros_like(q) if dead_load is None else dead_load
        charge=np.zeros_like(p) if free_charge is None else free_charge
        return SimpleNamespace(total_enthalpy_j=float(np.sum(self.W*(q*q-.5*q*force))),
            mechanical_residual=self.W*(2*q-force),
            electrical_residual=self.W*(-p-charge))
    def mass_action(self,a,*,density_kg_m3,mechanical=None,velocity=None):
        return self.W*float(density_kg_m3)*a

class GlobalVsReducedDofTests(unittest.TestCase):
    def setup_problem(self,n=7):
        f=QuadraticSurface(n)
        bc={"mechanical":{"left":{"support":"CLAMPED"}},
            "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
        problem=BoundarySpatialProblem(f,bc_spec=bc,calculation_kind="DYNAMIC")
        q=np.zeros((3,n,n));p=np.zeros((2,n,n))
        options=dict(q_predictor=q.copy(),velocity_predictor=q.copy(),time_s=.1,
            dt_s=.1,beta=.25,gamma=.5,density_kg_m3=1.,
            dead_load=np.broadcast_to(np.array([1.,0.,0.])[:,None,None],q.shape).copy(),
            settings=StaticFDMSettings(max_free_dofs=50))
        return problem,q,p,options
    def test_mvi_solves_reduced_space_above_full_grid_limit(self):
        problem,q,p,kw=self.setup_problem()
        a=problem.assemble(q,p,time_s=.1,acceleration=q,density_kg_m3=1.)
        self.assertGreater(a.space.basis.shape[1],kw["settings"].max_free_dofs)
        with self.assertRaisesRegex(ValueError,"free DOFs; limit=50"):
            solve_dynamic_fdm(problem,q,p,**kw)
        result=solve_dynamic_mvi(problem,q,p,**kw,
            mvi_settings=StaticMVISettings(rank=1,max_sweeps=5))
        self.assertTrue(result.converged,result.status)
        self.assertTrue(result.mvi_certificate["accepted"])
        self.assertEqual(result.mvi_certificate["fixed_rank"],1)
    def test_reduced_mvi_halfsteps_keep_dof_safety_gate(self):
        problem,q,p,kw=self.setup_problem()
        kw["settings"]=StaticFDMSettings(max_free_dofs=10)
        with self.assertRaisesRegex(ValueError,"free DOFs; limit=10"):
            solve_dynamic_mvi(problem,q,p,**kw,
                mvi_settings=StaticMVISettings(rank=1,max_sweeps=2))

if __name__=="__main__":unittest.main()
