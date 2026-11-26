"""Physical-kernel and dynamic integration contracts for matched SAT.

These tests use the real seven-theory kernels. The time-step test uses
manufactured generalized loads and an exact starting guess; it is NOT
a cold-start nonlinear pressure benchmark.
"""
import numpy as np
import pytest

from flexo_gradient_g3d import G3DMaterial
from cylinder_direct_strong_fdm import DirectStrongGrid,direct_strong_jet_balance
from cylinder_direct_strong_model import DirectStrongShell,THEORIES
from cylinder_direct_strong_boundary import assemble_physical_end_rows
from cylinder_mkr_field_jets import ELECTRIC_JETS
from cylinder_native_mkr_end_program import (
    compile_native_mkr_end_program,reference_axial_half_orders,
)
from cylinder_native_mkr_newmark import NativeStrongNewmark
from cylinder_native_mkr_initial import initialize_consistent_native_mkr


def _model(theory,gradient,geometry="GREEN_LAGRANGE"):
    grid=DirectStrongGrid(21,4,1.2,2.,axial_scheme="MATCHED_QUARTIC_SAT_V1")
    material=G3DMaterial(1.e9,.3,.002,.001,-.0005,1.e-8,gradient)
    return DirectStrongShell(theory,material,grid=grid,thickness_m=.08,
                             n_gauss=3,geometry_model=geometry)


def _program(model):
    return compile_native_mkr_end_program(
        model,axial_orders=reference_axial_half_orders(
            model.theory,model.material.gradient_model),
        mechanical="IMMOBILE_PARALLEL_ENDS",electric="GROUNDED_BOTH",
        potential_order=1)


def _fields(model):
    x=np.linspace(0.,1.,model.grid.nx)[None,:,None]
    theta=np.arange(model.grid.ns)[None,None,:]*2*np.pi/model.grid.ns
    q=1.e-7*np.arange(1,model.n_mechanical+1)[:,None,None]*np.sin(np.pi*x)*(
        1.+.2*np.cos(theta))
    p=.1*np.arange(1,3)[:,None,None]*np.sin(np.pi*x)*(1.+.1*np.sin(theta))
    q[:,[0,-1],:]=0.;p[:,[0,-1],:]=0.
    return q,p


def _rows(balance,q,p,model,program):
    return assemble_physical_end_rows(
        balance,q,model.grid,axial_orders=program.axial_orders,
        mechanical_ends=program.mechanical_ends,electrical=p,
        electrical_ends=program.electrical_ends)


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","Gz"))
def test_actual_coupled_dynamic_jet_work_is_preserved(theory,gradient):
    model=_model(theory,gradient);q,p=_fields(model);program=_program(model)
    # Nonzero displacement, electric field, velocity, acceleration and
    # mass damping; no prescribed physical solve or reference fallback.
    mc,ec=model.coefficients(q,p,velocity=.2*q,acceleration=-5*q,
        density_kg_m3=2500.,mass_damping_rate_per_s=.15)
    balance=direct_strong_jet_balance(mc,model.derivative_orders,model.grid,ec)
    rows=_rows(balance,q,p,model,program)
    rng=np.random.default_rng(117)
    dq=1.e-6*rng.normal(size=q.shape);dp=.1*rng.normal(size=p.shape)
    dq[:,[0,-1],:]=0.;dp[:,[0,-1],:]=0.
    weight=model.area_weights
    dm=np.stack(model.field_jets.prepare(dq,model.derivative_orders))
    de=np.stack(model.field_jets.prepare(dp,ELECTRIC_JETS))
    weak_m=mc*dm*weight[None,None,:,:];weak_e=ec*de*weight[None,None,:,:]
    strong=np.sum(weight*(np.sum(rows.mechanical*dq,axis=0)
                          +np.sum(rows.electrical*dp,axis=0)))
    weak=float(np.sum(weak_m)+np.sum(weak_e))
    scale=max(float(np.sum(abs(weak_m))+np.sum(abs(weak_e))),1.e-30)
    assert abs(strong-weak)/scale<3.e-11
    assert rows.status=="MATCHED_SAT_NATURAL_WORK_RESEARCH_NOT_BOUNDARY_CERTIFIED"


def test_sat_rejects_pointwise_higher_clamp_and_floating_electric():
    model=_model("CUF_TE1","G3D","LINEAR")
    with pytest.raises(ValueError,match="BOTH_ELECTRIC_ENDS_PRESCRIBED"):
        compile_native_mkr_end_program(model,axial_orders=(2,)*6,
            mechanical="IMMOBILE_PARALLEL_ENDS",electric="INSULATED_FLOATING",
            potential_order=1)
    q,p=_fields(model);program=_program(model)
    program.mechanical_ends["left"][0][1]={"kind":"ESSENTIAL","value":0.}
    with pytest.raises(ValueError,match="HIGHER_ESSENTIAL_TRACE_UNSUPPORTED"):
        _rows(model.balance(q,p),q,p,model,program)


def test_manufactured_native_sat_initialization_and_newmark():
    model=_model("CUF_TE1","G3D","LINEAR")
    q,p=_fields(model);program=_program(model);v=.2*q;a=-5*q;dt=1.e-4
    rho=2500.
    def loads_at(q,p,v,a):
        balance=model.balance(q,p,velocity=v,acceleration=a,density_kg_m3=rho)
        rows=_rows(balance,q,p,model,program)
        return rows.mechanical.copy(),-rows.electrical.copy()
    fm,fe=loads_at(q,p,v,a)
    engine=NativeStrongNewmark(model,axial_orders=program.axial_orders,
        mechanical_ends=program.mechanical_ends,electrical_ends=program.electrical_ends,
        density_kg_m3=rho,mechanical_load=fm,electrical_charge=fe)
    opts=dict(mechanical_state_scale=1.e-4,mechanical_equation_scale=1.e4,
              electrical_state_scale=1.,electrical_equation_scale=1.e-3)
    init=initialize_consistent_native_mkr(engine,q,v,p,acceleration_guess=a,**opts)
    assert init.state is not None,init.nonlinear.status
    np.testing.assert_allclose(init.state.acceleration,a,atol=1.e-9,rtol=1.e-8)
    target=q+dt*v+.5*dt**2*a
    target_v=v+dt*a;target_p=1.2*p
    fm1,fe1=loads_at(target,target_p,target_v,a)
    step=engine.advance(init.state,dt,mechanical_load=fm1,electrical_charge=fe1,
        mechanical_guess=target,electrical_guess=target_p,**opts)
    assert step.state is not None,step.nonlinear.status
    np.testing.assert_allclose(step.state.mechanical,target,atol=1.e-11,rtol=1.e-8)
    np.testing.assert_allclose(step.state.velocity,target_v,atol=1.e-9,rtol=1.e-8)
    np.testing.assert_allclose(step.state.acceleration,a,atol=1.e-8,rtol=1.e-6)
    np.testing.assert_allclose(step.state.electrical,target_p,atol=1.e-10,rtol=1.e-8)
