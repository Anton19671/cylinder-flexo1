"""Regression of a0 from constrained mechanical mass equilibrium (audit V-1)."""
from types import SimpleNamespace
import numpy as np
import pytest
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_functional_assembly import surface_trapezoid
from cylinder_initial_acceleration import consistent_initial_acceleration
from cylinder_matrixfree_fdm import SparseTraceProjector, MatrixFreeNewtonSettings
from cylinder_variational_fdm import StaticFDMSettings, state_scales

class LinearMechanicalFunctional:
    theory="CUF_TE1"
    n_mechanical=3
    n_derivatives=10
    electrical_on=False
    R=2.
    h=.2
    def __init__(self):
        self.ops=surface_trapezoid(4,4,1.2,self.R)
        self.W=self.ops.area_weights
        self.material=SimpleNamespace(young_pa=100.)
    def evaluate(self,q,p=None,*,dead_load=None,free_charge=None,**kwargs):
        load=np.zeros_like(q) if dead_load is None else np.asarray(dead_load)
        return SimpleNamespace(total_enthalpy_j=float(np.sum(self.W*(q*q-load*q))),
            mechanical_residual=self.W*(2*q-load),electrical_residual=None)
    def mass_action(self,acceleration,*,density_kg_m3,mechanical=None,velocity=None):
        return self.W*float(density_kg_m3)*acceleration

def solve_initial(backend,force=1.,q_interior=0.):
    f=LinearMechanicalFunctional()
    bc={"mechanical":{"left":{"support":"CLAMPED"},
                      "right":{"support":"CLAMPED"}}}
    problem=BoundarySpatialProblem(f,bc_spec=bc,calculation_kind="DYNAMIC")
    q=np.zeros((3,4,4));q[0,1:3,:]=q_interior
    load=np.zeros_like(q);load[0]=force
    rho=2.5
    if backend=="DENSE_REFERENCE":
        settings=StaticFDMSettings()
        assembly=problem.assemble(q,None,dead_load=load,time_s=0.,
            acceleration=np.zeros_like(q),density_kg_m3=rho)
        residual=assembly.full_residual
        edge=assembly.edge;projector=None
    else:
        settings=MatrixFreeNewtonSettings();assembly=None
        projector=SparseTraceProjector.from_problem(problem,q,None,
            state_scales(f,q,None,settings),time_s=0.,
            max_rows=settings.max_constraint_rows)
        _,residual,edge=problem.evaluate_effective_residual(
            q,None,dead_load=load,time_s=0.,
            acceleration=np.zeros_like(q),density_kg_m3=rho)
    a0=consistent_initial_acceleration(problem,q,None,
        initial_residual=residual,initial_edge=edge,settings=settings,
        density_kg_m3=rho,backend=backend,
        full_assembly=assembly,sparse_projector=projector)
    return q,rho,a0

@pytest.mark.parametrize("backend",["DENSE_REFERENCE","MATRIX_FREE_GMRES"])
def test_constant_load(backend):
    q,rho,a0=solve_initial(backend)
    assert a0.shape==q.shape
    np.testing.assert_allclose(a0[:,0,:],0.,atol=1e-12)
    np.testing.assert_allclose(a0[:,-1,:],0.,atol=1e-12)
    np.testing.assert_allclose(a0[0,1:3,:],1./rho,rtol=2e-9,atol=1e-11)

@pytest.mark.parametrize("backend",["DENSE_REFERENCE","MATRIX_FREE_GMRES"])
def test_zero_load(backend):
    _,_,a0=solve_initial(backend,force=0.)
    assert np.count_nonzero(a0)==0

@pytest.mark.parametrize("backend",["DENSE_REFERENCE","MATRIX_FREE_GMRES"])
def test_internal_restoring_acceleration(backend):
    _,rho,a0=solve_initial(backend,force=0.,q_interior=.2)
    np.testing.assert_allclose(a0[0,1:3,:],-2*.2/rho,rtol=1e-9)

def test_static_problem_rejected():
    problem=BoundarySpatialProblem(LinearMechanicalFunctional(),calculation_kind="STATIC")
    with pytest.raises(ValueError,match="DYNAMIC"):
        consistent_initial_acceleration(problem,np.zeros((3,4,4)),None,
            initial_residual=np.zeros(48),
            initial_edge=SimpleNamespace(constraints=()),
            settings=StaticFDMSettings(),density_kg_m3=1.)
