"""Deep science audit of the *real* seven-theory local functional.

All checks here call material algebra, local/whole-surface energies,
analytical variations or dead reference-load projection. They never
invoke Newton, MVI sweeps, or a physical dynamic integration.
"""
import numpy as np
import pytest
from cylinder_paths import activate
activate()
from cylinder_functional_assembly import CylinderFunctional,surface_sbp21
from flexo_gradient_g3d import G3DMaterial,enthalpy_derivatives
from cylinder_variational_loads import project_radial_load

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
GRADS=("G3D","Gz")
OPS=surface_sbp21(4,5,1.2,2.0)


def make_functional(theory,model,enabled):
    material=G3DMaterial(
        100.,.25,.08,.001 if enabled else 0.,
        -.002 if enabled else 0.,
        1.5 if enabled else 0.,gradient_model=model)
    ng=2*int(theory[-1])+1 if theory.startswith("CUF_TE") else (
        7 if theory=="REDDY" else 3)
    return CylinderFunctional(
        theory,material,radius_m=2.,thickness_m=.2,
        surface_operators=OPS,n_gauss=ng,
        geometry_model="GREEN_LAGRANGE")


def electric_count(theory):
    return int(theory[-1])+1 if theory.startswith("CUF_TE") else 2


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("grad",GRADS)
@pytest.mark.parametrize("electric",[False,True])
def test_real_seven_theory_rigid_axial_translation_has_zero_internal_energy(
    theory,grad,electric
):
    f=make_functional(theory,grad,electric)
    q=np.zeros((f.n_mechanical,*OPS.shape))
    q[0]=3.4e-5
    p=None
    if electric:
        p=np.zeros((electric_count(theory),*OPS.shape))
        p[0]=.017
    item=f.evaluate(q,p,energy_breakdown=True,
                    check_charge_balance=False)
    assert abs(item.total_enthalpy_j)<3.e-15
    for name in ("classical","gradient","dielectric","flexoelectric"):
        assert abs(item.energy_parts_j[name])<3.e-15,(
            theory,grad,electric,name,item.energy_parts_j[name])
    assert np.max(abs(item.mechanical_residual))<4.e-8
    if p is not None:
        assert np.max(abs(item.electrical_residual))<4.e-8


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("grad",GRADS)
def test_electrical_constant_gauge_does_not_change_actual_seven_theory_physics(
    theory,grad
):
    f=make_functional(theory,grad,True)
    rng=np.random.default_rng(610+THEORIES.index(theory))
    q=rng.normal(0.,2.e-7,(f.n_mechanical,*OPS.shape))
    p=rng.normal(0.,3.e-5,(electric_count(theory),*OPS.shape))
    change=p.copy()
    change[0]+=.15
    base=f.evaluate(q,p,check_charge_balance=False)
    shifted=f.evaluate(q,change,check_charge_balance=False)
    np.testing.assert_allclose(
        shifted.total_enthalpy_j,base.total_enthalpy_j,
        rtol=1.e-8,atol=2.e-12)
    np.testing.assert_allclose(
        shifted.mechanical_residual,base.mechanical_residual,
        rtol=1.e-8,atol=4.e-10)
    np.testing.assert_allclose(
        shifted.electrical_residual,base.electrical_residual,
        rtol=1.e-8,atol=2.e-10)


@pytest.mark.parametrize("grad",GRADS)
def test_full_gradient_or_gz_flexoelectric_maxwell_reciprocity(grad):
    """Cross derivative of ONE electrical enthalpy must be reciprocal."""
    material=G3DMaterial(
        130.,.25,.08,.21,-.16,3.5,gradient_model=grad)
    rng=np.random.default_rng(20261107)
    E=rng.normal(0.,.04,6)
    G=rng.normal(0.,.15,(6,3))
    electric=rng.normal(0.,.15,3)
    dg=rng.normal(0.,.1,(6,3))
    de=rng.normal(0.,.1,3)
    _,hyper0,D0=enthalpy_derivatives(E,G,electric,material)
    _,hyperE,_=enthalpy_derivatives(E,G,electric+de,material)
    _,_,Dg=enthalpy_derivatives(E,G+dg,electric,material)
    # dH/dg = hyper, but dH/dE = -D.
    lhs=float(np.sum(dg*(hyperE-hyper0)))
    rhs=-float(np.dot(de,Dg-D0))
    np.testing.assert_allclose(lhs,rhs,rtol=5.e-12,atol=3.e-14)
    if grad=="Gz":
        np.testing.assert_array_equal(hyper0[:,:2],np.zeros((6,2)))
        np.testing.assert_array_equal(hyperE[:,:2],np.zeros((6,2)))


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("side",("INNER","OUTER"))
def test_surface_pressure_discrete_work_is_conjugate_to_real_thickness_fields(
    theory,side
):
    """A projected nominal dead face load does exactly the intended work."""
    if theory=="KL" and side!="OUTER":
        # KL permits MIDDLE only, never an offset-face dead nodal force.
        return
    f=make_functional(theory,"Gz",False)
    face="MIDDLE" if theory=="KL" else side
    pressure=3.1
    load=project_radial_load(f,dict(
        type="UNIFORM_RADIAL_FACE",face=face,kinematics="DEAD_REFERENCE",
        direction="RADIAL_INWARD",q=pressure,full_circumference=True))
    q=np.zeros((f.n_mechanical,*OPS.shape))
    rng=np.random.default_rng(100+THEORIES.index(theory))
    if theory.startswith("CUF_TE"):
        n=int(theory[-1])+1
        q[2*n:3*n]=rng.normal(0.,.001,(n,*OPS.shape))
        xi=1. if face=="OUTER" else -1.
        face_radial=sum(xi**k*q[2*n+k] for k in range(n))
    else:
        q[2]=rng.normal(0.,.001,OPS.shape)
        face_radial=q[2]
    H=1.+({"MIDDLE":0.,"INNER":-.1,"OUTER":.1}[face]/f.R)
    left=float(np.sum(OPS.area_weights[None,...]*load.generalized*q))
    right=float(np.sum(OPS.area_weights*H*load.face_pressure_pa*face_radial))
    np.testing.assert_allclose(left,right,rtol=5.e-14,atol=5.e-14)
    assert load.metadata["reference_face_area_factor"]==pytest.approx(H)
    assert load.metadata["load_is_follower"] is False


def test_nonlinear_kl_offset_face_pressure_is_not_silently_faked():
    f=make_functional("KL","G3D",False)
    for face in ("INNER","OUTER"):
        with pytest.raises(ValueError,match="offset-face"):
            project_radial_load(f,dict(type="UNIFORM_RADIAL_FACE",
                face=face,kinematics="DEAD_REFERENCE",
                direction="RADIAL_INWARD",q=1.,full_circumference=True))
