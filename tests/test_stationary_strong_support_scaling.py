"""Regression of false static acceptance caused by mixed residual units."""
import numpy as np
import pytest
from cylinder_direct_strong_boundary import stationary_essential_scales
from cylinder_direct_strong_fdm import DirectStrongGrid,direct_strong_jet_balance
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import StrongNewtonSettings
from cylinder_native_mkr_end_program import compile_native_mkr_end_program
from flexo_gradient_g3d import G3DMaterial


@pytest.mark.parametrize("scheme,nx",(("LEGACY_TRAPEZOID",7),
                                     ("MATCHED_QUARTIC_V1",21),
                                     ("MATCHED_QUARTIC_SAT_V1",21)))
def test_static_solver_does_not_accept_false_mechanical_or_electric_support(
        scheme,nx,monkeypatch):
    # Real native solver, physical end-row assembly, and actual Newton;
    # only the interior constitutive residual is manufactured to isolate
    # the scale bug without expensive physical Jacobian evaluations.
    grid=DirectStrongGrid(nx,4,1.2,2.,axial_scheme=scheme)
    mat=G3DMaterial(1.e9,.3,.002,.001,-.0005,1.e-8,"G3D")
    model=DirectStrongShell("FSDT",mat,grid=grid,thickness_m=.08,
                            n_gauss=2,geometry_model="LINEAR")
    q=np.zeros(model.shape);p=np.zeros((1,nx,4))
    q[:,[0,-1],:]=1.e-5;p[:,[0,-1],:]=2.e-5
    program=compile_native_mkr_end_program(
        model,axial_orders=(1,)*5,mechanical="IMMOBILE_PARALLEL_ENDS",
        electric="GROUNDED_BOTH",potential_order=0)
    def manufactured(q,p,**kwargs):
        mc=np.zeros((len(model.derivative_orders),*q.shape))
        ec=np.zeros((3,*p.shape))
        mc[0]=q;ec[0]=p
        return direct_strong_jet_balance(mc,model.derivative_orders,grid,ec)
    monkeypatch.setattr(model,"balance",manufactured)
    answer=model.solve(q,p,axial_orders=program.axial_orders,
        mechanical_ends=program.mechanical_ends,electrical_ends=program.electrical_ends,
        mechanical_state_scale=1.e-4,mechanical_equation_scale=1.e4,
        electrical_state_scale=1.,electrical_equation_scale=1.e4,
        settings=StrongNewtonSettings(max_dofs=1024,linear_solver="DENSE_LU",
                                     dense_lu_dof_limit=1024,max_iterations=3))
    assert answer.converged,answer.status
    assert answer.iterations>=1
    assert np.max(abs(answer.mechanical[:,[0,-1],:]))<=1.e-9
    assert np.max(abs(answer.electrical[:,[0,-1],:]))<=1.e-9


def test_scale_only_prescribed_rows_and_never_relax_user_scale():
    grid=DirectStrongGrid(11,4,1.2,2.)
    natural={"kind":"NATURAL","value":0.}
    fixed={"kind":"ESSENTIAL","value":0.}
    ends={"left":[[fixed,natural,fixed]],"right":[[natural,fixed,natural]]}
    base=np.full((1,11,4),1.e4);base[:,2]=1.e-3
    scale=stationary_essential_scales(grid,base.shape,ends,base,
                                     scaled_absolute_tolerance=1.e-8)
    expected=base.copy();expected[:,[0,9]]=0.1
    np.testing.assert_allclose(scale,expected)
    assert base[0,0,0]==1.e4
    eends={"left":[fixed],"right":[natural]}
    electric=stationary_essential_scales(grid,base.shape,eends,base,
                    scaled_absolute_tolerance=1.e-8,electric=True)
    expected=base.copy();expected[:,0]=.1
    np.testing.assert_allclose(electric,expected)
