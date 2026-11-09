"""Native MKR preparation, physical-law preservation and method isolation.

Only prescribed fields and zero-state solves are evaluated. Reference
comparisons verify the extraction, not independent physical convergence.
"""
import os
import subprocess
import sys
import numpy as np
import pytest

from cylinder_direct_strong_model import DirectStrongShell, THEORIES
from cylinder_direct_strong_fdm import DirectStrongGrid, direct_coefficient_derivative
from cylinder_mkr_field_jets import MKRFieldJets, MECHANICAL_JETS
from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from cylinder_cuf_parallel_ends import CUFParallelEndProblem
from cylinder_direct_strong_newton import StrongNewtonSettings
from flexo_gradient_g3d import G3DMaterial


def models(theory, gradient="G3D", geometry="GREEN_LAGRANGE", electric=True):
    grid=DirectStrongGrid(7,4,1.2,2.)
    mat=G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                    -.002 if electric else 0.,.8 if electric else 0.,gradient)
    native=DirectStrongShell(theory,mat,grid=grid,thickness_m=.3,
                             n_gauss=2,geometry_model=geometry)
    ref=CylinderFunctional(theory,mat,radius_m=grid.radius_m,thickness_m=.3,
        surface_operators=surface_trapezoid(grid.nx,grid.ns,grid.length_m,grid.radius_m),
        n_gauss=2,geometry_model=geometry)
    return native,ref


def forbidden(*args,**kwargs):
    raise AssertionError("native MKR called shared spatial preparation")


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","GZ"))
def test_all_theories_preserve_nonzero_static_and_dynamic_physics(theory,gradient,monkeypatch):
    model,ref=models(theory,gradient)
    rng=np.random.default_rng(716)
    q=rng.normal(0,2.e-5,model.shape)
    p=rng.normal(0,1.e-3,(2,model.grid.nx,model.grid.ns))
    v=rng.normal(0,1.e-4,model.shape)
    a=rng.normal(0,1.e-3,model.shape)
    kwargs=dict(velocity=v,acceleration=a,density_kg_m3=3.,
                eta_s=.02,mass_damping_rate_per_s=.15)
    ref_static=ref._dynamic_g3d_weak_jet_coefficients(q,p)
    ref_dynamic=ref._dynamic_g3d_weak_jet_coefficients(q,p,**kwargs)
    # Any accidental reference callback in the new path must fail.
    monkeypatch.setattr(CylinderFunctional,"_field_derivatives",forbidden)
    monkeypatch.setattr(CylinderFunctional,"_dynamic_g3d_weak_jet_coefficients",forbidden)
    import cylinder_kl_exact_inertia as kl
    monkeypatch.setattr(kl,"apply_axial",forbidden)
    monkeypatch.setattr(kl,"apply_periodic",forbidden)
    for expected,actual in zip(ref_static,model.coefficients(q,p)):
        np.testing.assert_allclose(actual,expected,rtol=3.e-8,atol=2.e-10)
    actual_dynamic=model.coefficients(q,p,**kwargs)
    for expected,actual in zip(ref_dynamic,actual_dynamic):
        np.testing.assert_allclose(actual,expected,rtol=3.e-8,atol=2.e-10)
    # Electricity has no added inertial/damping equations.
    np.testing.assert_allclose(actual_dynamic[1],ref_static[1],rtol=3.e-8,atol=2.e-10)


@pytest.mark.parametrize("theory",("KL","REDDY","CUF_TE1"))
@pytest.mark.parametrize("geometry",("LINEAR","VON_KARMAN"))
def test_reduced_geometry_uses_same_local_laws(theory,geometry):
    model,ref=models(theory,geometry=geometry)
    rng=np.random.default_rng(918)
    q=rng.normal(0,2.e-5,model.shape)
    p=rng.normal(0,1.e-3,(1,7,4))
    args=dict(velocity=rng.normal(0,1.e-4,model.shape),
              acceleration=rng.normal(0,1.e-3,model.shape),density_kg_m3=3.)
    expected=ref._dynamic_g3d_weak_jet_coefficients(q,p,**args)
    actual=model.coefficients(q,p,**args)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=3.e-8,atol=2.e-10)


def test_field_jets_have_exact_polynomial_and_fourier_symbols():
    g=DirectStrongGrid(9,8,2.,1.5)
    own=MKRFieldJets(g)
    x=(np.arange(g.nx)*g.dx)[None,:,None]
    theta=(2*np.pi*np.arange(g.ns)/g.ns)[None,None,:]
    # Real part of x^2*exp(i*m*theta). Axial quadratics are exact even at ends;
    # the periodic phase is the stated forward-difference symbol.
    for mode in (1,g.ns//2):
        field=np.broadcast_to(x*x*np.cos(mode*theta),(1,g.nx,g.ns)).copy()
        for name,ax,ss in MECHANICAL_JETS:
            polynomial=(x*x if ax==0 else 2*x if ax==1 else
                        np.full_like(x,2.) if ax==2 else np.zeros_like(x))
            symbol=((np.exp(2j*np.pi*mode/g.ns)-1)/g.ds)**ss
            exact=np.real(polynomial*np.exp(1j*mode*theta)*symbol)
            np.testing.assert_allclose(own.derivative(field,ax,ss),exact,atol=3.e-11)
        actual=own.prepare(field,MECHANICAL_JETS)
        for value,(_,ax,ss) in zip(actual,MECHANICAL_JETS):
            np.testing.assert_allclose(value,own.derivative(field,ax,ss),atol=1.e-12)
    # Paired forward potential gradient and backward electric flux derivative
    # retain the alternating mode on an even periodic grid.
    checker=np.broadcast_to((-1.)**np.arange(g.ns),(1,g.nx,g.ns))
    lap=direct_coefficient_derivative(own.derivative(checker,0,1),0,1,g)
    np.testing.assert_allclose(lap,-4*checker/g.ds**2)


@pytest.mark.parametrize("order",range(1,5))
def test_cuf_nodal_inertia_retains_even_grid_checkerboard(order):
    model,_=models(f"CUF_TE{order}",electric=False)
    q=np.zeros(model.shape)
    a=np.broadcast_to((-1.)**np.arange(4),model.shape).copy()
    mc,ec=model.coefficients(q,velocity=q,acceleration=a,density_kg_m3=3.)
    from cuf_g3d_weak import mass_matrix_te
    mass=mass_matrix_te(radius_m=2.,thickness_m=.3,rho_kg_m3=3.,te_order=order)
    exact=np.einsum("ij,cjkl->cikl",mass,a.reshape(3,order+1,7,4)).reshape(model.shape)
    np.testing.assert_allclose(mc[0],exact,atol=1.e-13)
    np.testing.assert_allclose(mc[1:],0.,atol=1.e-13)
    assert np.sum(a*mc[0])>0
    assert ec is None


def test_cuf_end_solve_runs_with_reference_and_mvi_imports_forbidden():
    script='''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith(("independent_mvi", "cylinder_variational_mvi", "cylinder_mvi")) or
                fullname in ("cylinder_functional_assembly", "cylinder_variational_fdm")):
            raise AssertionError("forbidden spatial backend: "+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from cylinder_direct_strong_model import DirectStrongShell, THEORIES
from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_direct_strong_newton import StrongNewtonSettings
from cylinder_cuf_parallel_ends import CUFParallelEndProblem
from flexo_gradient_g3d import G3DMaterial
model=DirectStrongShell("CUF_TE1",G3DMaterial(90.,.24,.06,.004,-.002,.8),
    grid=DirectStrongGrid(7,4,1.2,2.),thickness_m=.3,n_gauss=2)
problem=CUFParallelEndProblem(model,axial_orders=(2,)*6,potential_order=1)
q=np.zeros(model.shape);p=np.zeros((2,7,4))
answer=problem.solve(q,p,mechanical_state_scale=1.,mechanical_equation_scale=1.,
    electrical_state_scale=1.,electrical_equation_scale=1.,
    settings=StrongNewtonSettings(max_dofs=225))
assert answer.converged and answer.iterations==0
model.balance(q,p,velocity=q,acceleration=q,density_kg_m3=3.,eta_s=.02)
for theory in THEORIES:
    native=DirectStrongShell(theory,model.material,grid=model.grid,
        thickness_m=.3,n_gauss=2)
    zero=np.zeros(native.shape)
    native.balance(zero,p,velocity=zero,acceleration=zero,
        density_kg_m3=3.,mass_damping_rate_per_s=.1)
'''
    from cylinder_paths import activate
    activate()
    process_env=os.environ.copy()
    process_env["PYTHONPATH"]=os.pathsep.join(sys.path)
    result=subprocess.run([sys.executable,"-c",script],capture_output=True,
                          text=True,timeout=30,env=process_env)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize("theory",THEORIES)
def test_native_static_solver_all_theories_at_zero_state(theory,monkeypatch):
    model,_=models(theory,electric=False)
    monkeypatch.setattr(CylinderFunctional,"_dynamic_g3d_weak_jet_coefficients",forbidden)
    # Explicit declared end program only exercises native Newton wiring at
    # the exact unloaded state; it is NOT a physical order certificate.
    orders=(2,)*model.n_mechanical
    ends={side: [[{"kind":"ESSENTIAL","value":0.},
                  {"kind":"NATURAL","value":0.}] for _ in orders]
          for side in ("left","right")}
    answer=model.solve(np.zeros(model.shape),axial_orders=orders,
        mechanical_ends=ends,mechanical_state_scale=1.,mechanical_equation_scale=1.,
        settings=StrongNewtonSettings(max_dofs=500))
    assert answer.converged and answer.iterations==0


def test_native_validation_rejects_missing_physics():
    model,_=models("CUF_TE1")
    q=np.zeros(model.shape);p=np.zeros((1,7,4))
    with pytest.raises(ValueError,match="active potential"):
        model.balance(q)
    with pytest.raises(ValueError,match="together"):
        model.balance(q,p,velocity=q)
    with pytest.raises(ValueError,match="density"):
        model.balance(q,p,velocity=q,acceleration=q,density_kg_m3=0.)
    with pytest.raises(ValueError,match="nonnegative"):
        model.balance(q,p,eta_s=True)
    with pytest.raises(ValueError,match="total <=3"):
        model.field_jets.derivative(q,True,0)
