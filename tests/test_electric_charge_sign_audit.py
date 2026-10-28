"""Regression for audit V-2: positive physical electrode free charge."""
import numpy as np
import pytest
from cylinder_functional_assembly import CylinderFunctional,surface_trapezoid,surface_sbp21
from flexo_gradient_g3d import G3DMaterial
from cylinder_variational_boundary import combine_functional_and_boundary,compile_boundary_program
from cylinder_electric_face_flux import dielectric_face_flux_fdm
from cylinder_physical_fields import recover_thickness_fields
from cylinder_face_conditions import face_conditions_from_recovery
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_sbp_full_balance import audit_full_sbp_work

def model(mode="G3D",sbp=False):
    ops=surface_sbp21(4,5,1.2,2.) if sbp else surface_trapezoid(4,4,1.2,2.)
    f=CylinderFunctional("CUF_TE1",G3DMaterial(100.,.25,0.,0.,0.,1.,gradient_model=mode),
        radius_m=2.,thickness_m=.2,surface_operators=ops,n_gauss=3)
    return f,np.zeros((f.n_mechanical,*ops.shape))

@pytest.mark.parametrize("mode",["G3D","Gz"])
def test_positive_outer_electrode_charge_produces_positive_voltage(mode):
    f,q=model(mode)
    spec={"electrical":{"outer":{"mode":"CHARGE","value":1.},
                        "inner":{"mode":"POTENTIAL","value":0.}}}
    for u in (-.21,0.,.21):
        p=np.ones((2,*f.ops.shape))*(u/2.)
        _,_,re,edge=combine_functional_and_boundary(f,q,p,bc_spec=spec)
        derivative=float(np.sum((re[0]+re[1])*.5))
        if u==.21:assert abs(derivative)<2e-12
        else:assert derivative>0
        assert edge.constraint_max_defect(q,p)<1e-12

@pytest.mark.parametrize("mode",["G3D","Gz"])
def test_generalized_positive_charge_work(mode):
    f,q=model(mode)
    rng=np.random.default_rng(301)
    p=rng.normal(size=(2,*f.ops.shape))*.1
    charge=np.zeros_like(p)
    charge[0]=np.sin(2*np.pi*np.arange(f.ops.shape[1])/f.ops.shape[1])[None,:]
    a=f.evaluate(q,p,energy_breakdown=False)
    b=f.evaluate(q,p,free_charge=charge,energy_breakdown=False)
    assert b.total_enthalpy_j-a.total_enthalpy_j==pytest.approx(np.sum(f.W[None,...]*charge*p),abs=2e-13)
    np.testing.assert_allclose(b.electrical_residual-a.electrical_residual,f.W[None,...]*charge,atol=1e-12)

@pytest.mark.parametrize("side",["inner","outer","left","right"])
def test_charged_boundary_work_is_positive(side):
    f,q=model()
    p=np.random.default_rng(9).normal(0.,.05,(2,*f.ops.shape))
    ground="outer" if side in ("left","right") else "inner"
    if side=="inner":ground="outer"
    def bc(value):
        return {"electrical":{side:{"mode":"CHARGE","value":value},
                              ground:{"mode":"POTENTIAL","value":0.}}}
    a=compile_boundary_program(f,q,p,spec=bc(1.))
    b=compile_boundary_program(f,q,p,spec=bc(0.))
    source=a.electrical_residual_correction-b.electrical_residual_correction
    assert a.potential_correction_j-b.potential_correction_j==pytest.approx(float(np.sum(source*p)),abs=2e-13)

def test_recovered_outward_induction_is_negative_free_charge():
    f,q=model()
    p=np.ones((2,*f.ops.shape))*.1
    fields,report=face_conditions_from_recovery(f,recover_thickness_fields(f,q,p,n_thickness=3),
        {"electrical":{"outer":{"mode":"CHARGE","value":1.},
                       "inner":{"mode":"POTENTIAL","value":0.}}})
    outer=report["electric_faces"]["outer"]
    assert outer["prescribed_free_charge_c_per_m2"]==1.
    assert outer["prescribed_outward_induction_c_per_m2"]==-1.
    assert outer["normal_residual_abs_max_c_per_m2"]<1e-12
    np.testing.assert_allclose(fields["face_outer_electric_outward_induction_c_per_m2"],-1.,atol=1e-12)

def test_independent_flux_and_sbp_charge_signs():
    phi=np.zeros((4,5))
    charge=np.broadcast_to(np.sin(2*np.pi*np.arange(5)/5),phi.shape).copy()
    flux=dielectric_face_flux_fdm(phi,length_m=1.2,radius_m=2.,
        thickness_m=.2,kappa_f_per_m=1.,free_charge_c_per_m2=charge)
    dx=1.2/3
    wx=np.array([dx/2,dx,dx,dx/2])[:,None]
    np.testing.assert_allclose(flux.residual_c,wx*(4*np.pi/5)*charge,atol=2e-14)
    f,q=model(sbp=True)
    p=np.zeros((2,*f.ops.shape))
    problem=BoundarySpatialProblem(f,bc_spec={"electrical":{"inner":{"mode":"POTENTIAL","value":0.}}})
    ledger=audit_full_sbp_work(problem,q,p,free_charge=np.ones_like(p)*.03)
    np.testing.assert_allclose(ledger.electrical.reconstructed,ledger.electrical.independent_full,atol=2e-12,rtol=1e-9)
