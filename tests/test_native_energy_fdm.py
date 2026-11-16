"""Independent native energy FDM: algebraic virtual-work and end-space contracts.

No nonlinear cylindrical shell time step or loaded physical campaign is run.
The tests independently form the actual native-FDM jet work and end traces.
"""
import numpy as np
import pytest

from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_mkr_field_jets import MKRFieldJets, ELECTRIC_JETS
from cylinder_native_energy_fdm import (
    NativeEnergyFDM, NativeEnergyEndSpace, NativeEnergyWork,
)
from cylinder_direct_strong_newton import (
    StrongElectricGauge,StrongNewtonSettings,
)
from flexo_gradient_g3d import G3DMaterial


def _setup(theory="KL",electric=True,nx=7,ns=5):
    grid=DirectStrongGrid(nx,ns,1.2,2.)
    material=G3DMaterial(90.,.24,.06,.004 if electric else 0.,
                         -.002 if electric else 0.,.8 if electric else 0.)
    model=DirectStrongShell(
        theory,material,grid=grid,thickness_m=.3,n_gauss=2)
    return NativeEnergyFDM(model)


@pytest.mark.parametrize("theory",[
    "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"])
@pytest.mark.parametrize("electric",[False,True])
def test_native_energy_jet_operator_exactly_matches_independent_virtual_work(
        theory,electric):
    engine=_setup(theory,electric)
    model=engine.model
    rng=np.random.default_rng(20261116)
    coeff=rng.normal(size=(len(model.derivative_orders),*model.shape))
    virtual=rng.normal(size=model.shape)
    f=rng.normal(size=model.shape)
    ec=(rng.normal(size=(3,2,model.grid.nx,model.grid.ns))
        if electric else None)
    pv=(rng.normal(size=(2,model.grid.nx,model.grid.ns))
        if electric else None)
    charge=(rng.normal(size=pv.shape) if electric else None)
    result=engine.jet_action(coeff,ec,
                             mechanical_load=f,electrical_charge=charge)
    jets=MKRFieldJets(model.grid)
    derivatives=jets.prepare(virtual,model.derivative_orders)
    expected_m=sum(float(np.sum(model.area_weights*coeff[k] *d))
                   for k,d in enumerate(derivatives))
    expected_m-=float(np.sum(model.area_weights*f*virtual))
    actual_m=float(np.sum(virtual*result.mechanical))
    np.testing.assert_allclose(actual_m,expected_m,rtol=5.e-12,atol=5.e-10)

    if electric:
        assert result.electrical is not None
        ed=jets.prepare(pv,ELECTRIC_JETS)
        expected_e=sum(float(np.sum(model.area_weights*ec[k]*d))
                       for k,d in enumerate(ed))
        expected_e+=float(np.sum(model.area_weights*charge*pv))
        actual_e=float(np.sum(pv*result.electrical))
        np.testing.assert_allclose(
            actual_e,expected_e,rtol=5.e-12,atol=5.e-10)
    else:
        assert result.electrical is None


@pytest.mark.parametrize("normal_order",[0,1,2])
def test_native_energy_end_constraints_are_exact_on_both_physical_ends(
        normal_order):
    from cylinder_direct_strong_boundary import native_kinematic_normal_trace
    engine=_setup("KL",False)
    model=engine.model
    counts=(normal_order+1,)*model.n_mechanical
    prescribed={}
    for side in ("left","right"):
        prescribed[side]=[
            [{"kind":"ESSENTIAL","value":float(c+1)*(.1+j*.025)}
             for j in range(normal_order+1)]
            for c in range(model.n_mechanical)]
    space=NativeEnergyEndSpace(
        engine,axial_orders=counts,mechanical_ends=prescribed)
    rng=np.random.default_rng(876)
    free=rng.normal(size=space.reduced_size)
    q,p=space.decode(free)
    assert p is None
    for side in ("left","right"):
        for c in range(model.n_mechanical):
            for order,entry in enumerate(prescribed[side][c]):
                got=native_kinematic_normal_trace(q,model.grid,c,order,side)
                np.testing.assert_allclose(
                    got,entry["value"],rtol=0.,atol=5.e-10)


def test_native_energy_natural_end_work_uses_true_normal_traces():
    from cylinder_direct_strong_boundary import native_kinematic_normal_trace
    engine=_setup("KL",False)
    model=engine.model
    ns=model.grid.ns
    ends={side:[
        [{"kind":"NATURAL","value":.1*(c+1)*(order+1)}
         for order in range(3)]
        for c in range(model.n_mechanical)]
        for side in ("left","right")}
    space=NativeEnergyEndSpace(
        engine,axial_orders=(3,)*model.n_mechanical,mechanical_ends=ends)
    rng=np.random.default_rng(872)
    virtual=rng.normal(size=model.shape)
    external=sum(
        model.grid.ds*np.sum(
            native_kinematic_normal_trace(virtual,model.grid,c,order,side)
            *entry["value"])
        for side in ("left","right")
        for c,items in enumerate(ends[side])
        for order,entry in enumerate(items))
    stored=sum(float(np.sum(virtual[c]*field.external_work))
               for c,field in enumerate(space.mechanical_spaces))
    np.testing.assert_allclose(stored,external,rtol=1.e-12,atol=1.e-12)


def test_native_energy_gauge_is_exact_and_projected_gradient_matches_work():
    engine=_setup("KL",True)
    model=engine.model
    ends={side:[
        [{"kind":"NATURAL","value":0.}]
        for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    electric_ends={side:[
        {"kind":"NATURAL","value":0.} for _ in range(2)]
        for side in ("left","right")}
    weights=np.zeros((2,model.grid.nx,model.grid.ns))
    weights[0]=1./(model.grid.nx*model.grid.ns)
    gauge=StrongElectricGauge(weights,target_v=.35)
    space=NativeEnergyEndSpace(
        engine,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=ends,electrical_ends=electric_ends,
        electric_gauge=gauge)
    rng=np.random.default_rng(781)
    y=rng.normal(size=space.reduced_size)
    q,p=space.decode(y)
    np.testing.assert_allclose(np.sum(p*weights),.35,rtol=0.,atol=3.e-12)
    # The reduced residual must be the Gateaux derivative along the
    # exact affine subspace, INCLUDING the compensated gauge direction.
    mq=rng.normal(size=q.shape)
    ep=rng.normal(size=p.shape)
    work=NativeEnergyWork(mq,ep)
    direction=rng.normal(size=y.shape)
    plus=space.decode(y+1.e-6*direction)
    minus=space.decode(y-1.e-6*direction)
    gradient=float(np.dot(space.residual(work),direction))
    independent=float(
        np.sum((plus[0]-minus[0])*.5e6*mq)
        +np.sum((plus[1]-minus[1])*.5e6*ep))
    np.testing.assert_allclose(gradient,independent,rtol=5.e-10,atol=5.e-9)


def test_native_energy_rejects_ungauged_fully_insulated_electric_problem():
    engine=_setup("KL",True)
    model=engine.model
    ends={side:[
        [{"kind":"NATURAL","value":0.}]
        for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    electrical={side:[{"kind":"NATURAL","value":0.}]
                for side in ("left","right")}
    with pytest.raises(ValueError,match="UNGAUGED"):
        NativeEnergyEndSpace(
            engine,axial_orders=(1,)*model.n_mechanical,
            mechanical_ends=ends,electrical_ends=electrical)


def test_native_energy_differs_from_uncertified_direct_strong_operator():
    engine=_setup("FSDT",False)
    model=engine.model
    rng=np.random.default_rng(73)
    coeff=rng.normal(size=(len(model.derivative_orders),*model.shape))
    energy=engine.jet_action(coeff).mechanical
    from cylinder_direct_strong_fdm import direct_strong_jet_balance
    strong=direct_strong_jet_balance(
        coeff,model.derivative_orders,model.grid).mechanical_bulk
    assert not np.allclose(
        energy,model.area_weights[None,:,:]*strong,rtol=1.e-8,atol=1.e-10)



def test_native_energy_electric_natural_work_uses_prescribed_outward_flux():
    """Target Q_n=value: the algebraic residual is weak work minus boundary work."""
    engine=_setup("KL",True)
    model=engine.model
    ends={side:[
        [{"kind":"NATURAL","value":0.}]
        for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    electric_ends={
        "left":[{"kind":"ESSENTIAL","value":0.},
                {"kind":"NATURAL","value":-.3}],
        "right":[{"kind":"ESSENTIAL","value":0.},
                 {"kind":"NATURAL","value":.6}],
    }
    space=NativeEnergyEndSpace(
        engine,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=ends,electrical_ends=electric_ends)
    virtual=np.zeros((2,model.grid.nx,model.grid.ns))
    virtual[1,0,:]=.2
    virtual[1,-1,:]=-.4
    external=space.electrical_spaces[1].external_work
    target=model.grid.ds*model.grid.ns*(.2*(-.3)-.4*.6)
    np.testing.assert_allclose(
        float(np.sum(virtual[1]*external)),
        target,rtol=1.e-12,atol=1.e-12)


def test_native_energy_static_reduced_solve_does_not_use_strong_bulk(monkeypatch):
    """One algebraic quadratic energy: a single exact Newton correction."""
    engine=_setup("KL",False,nx=7,ns=4)
    model=engine.model
    ends={side:[
        [{"kind":"ESSENTIAL","value":0.}]
        for _ in range(model.n_mechanical)]
        for side in ("left","right")}
    space=NativeEnergyEndSpace(
        engine,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=ends)
    rng=np.random.default_rng(197)
    target=rng.normal(size=space.reduced_size)*.01
    desired,_=space.decode(target)
    def algebraic_weak(q,p,**kwargs):
        assert p is None
        return NativeEnergyWork(q-desired,None)
    monkeypatch.setattr(engine,"balance",algebraic_weak)
    guess=np.zeros(model.shape)
    got=space.solve_static(
        guess,reduced_state_scale=1.,
        reduced_equation_scale=1.,
        settings=StrongNewtonSettings(
            max_dofs=200,max_iterations=5,
            scaled_absolute_tolerance=1.e-9))
    assert got.nonlinear.converged,got.nonlinear.status
    np.testing.assert_allclose(got.mechanical,desired,atol=1.e-7)
    assert got.electrical is None



@pytest.mark.parametrize("theory",("KL","FSDT","REDDY","CUF_TE2"))
@pytest.mark.parametrize("electric",(False,True))
def test_real_constitutive_enthalpy_gateaux_matches_native_energy_fdm(
        theory,electric):
    """Real G3D/coupled constitutive work, not randomized coefficient laws.

    Small-grid first-variation audit only. This never solves a loaded shell.
    Independent central energy differences are computed from the LOCAL
    material enthalpy rather than the FDM residual or its coefficients.
    """
    engine=_setup(theory,electric,nx=7,ns=4)
    model=engine.model
    rng=np.random.default_rng(2766)
    q=1.e-4*rng.normal(size=model.shape)
    dq=.1*rng.normal(size=model.shape)
    p=(1.e-4*rng.normal(size=(2,model.grid.nx,model.grid.ns))
       if electric else None)
    dp=(.1*rng.normal(size=p.shape) if electric else None)

    def direct_total_enthalpy(qstate,pstate):
        mder=model.field_jets.prepare(qstate,model.derivative_orders)
        efield=(pstate if electric else
                np.zeros((1,model.grid.nx,model.grid.ns)))
        eder=model.field_jets.prepare(efield,ELECTRIC_JETS)
        total=0.
        for i in range(model.grid.nx):
            for j in range(model.grid.ns):
                point=model._point_jets(mder,eder,i,j)
                local=model._local(point)
                total+=float(
                    model.area_weights[i,j]*local.energy_j_per_m2)
        return total

    work=engine.balance(q,p)
    derivative=float(np.sum(dq*work.mechanical))
    if electric:
        derivative+=float(np.sum(dp*work.electrical))
    step=2.e-6
    ep=direct_total_enthalpy(
        q+step*dq,None if p is None else p+step*dp)
    em=direct_total_enthalpy(
        q-step*dq,None if p is None else p-step*dp)
    independent=(ep-em)/(2*step)
    np.testing.assert_allclose(
        derivative,independent,rtol=7.e-5,atol=4.e-7)



@pytest.mark.parametrize("theory",[
    "KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"])
@pytest.mark.parametrize("electric",[False,True])
def test_native_energy_tangent_is_reciprocal_on_coupled_field_directions(
        theory,electric):
    """Real nonlinear electromechanical Hessian must satisfy reciprocity.

    This is stronger than a manufactured polynomial-jet test: both
    independent perturbations pass through the complete native thickness
    constitutive kernels. The electric tangent need not be positive,
    because electric enthalpy is a saddle functional, but must be
    reciprocal as a Hessian of one differentiable potential.
    No nonlinear shell solve, boundary certificate or time step is run.
    """
    fdm=_setup(theory,electric,nx=7,ns=4)
    model=fdm.model
    rng=np.random.default_rng(2026111607)
    q=1.e-5*rng.standard_normal(model.shape)
    p=(1.e-5*rng.standard_normal((2,model.grid.nx,model.grid.ns))
       if electric else None)
    dq1=rng.standard_normal(model.shape)
    dq2=rng.standard_normal(model.shape)
    dp1=(rng.standard_normal(p.shape) if electric else None)
    dp2=(rng.standard_normal(p.shape) if electric else None)

    def directional_tangent(dq,dp):
        h=1.e-6
        plus=fdm.balance(q+h*dq,None if p is None else p+h*dp)
        minus=fdm.balance(q-h*dq,None if p is None else p-h*dp)
        return (plus.mechanical-minus.mechanical)/(2*h),(
            None if not electric else
            (plus.electrical-minus.electrical)/(2*h))

    r2_m,r2_e=directional_tangent(dq2,dp2)
    r1_m,r1_e=directional_tangent(dq1,dp1)
    left=float(np.sum(dq1*r2_m))
    right=float(np.sum(dq2*r1_m))
    if electric:
        left+=float(np.sum(dp1*r2_e))
        right+=float(np.sum(dp2*r1_e))
    np.testing.assert_allclose(left,right,rtol=3.e-4,atol=3.e-6)



@pytest.mark.parametrize("nx,ns",[(7,5),(17,19),(33,13)])
def test_bounded_stencil_probes_match_full_native_jet_actions(nx,ns):
    """Batching the exact native jets changes memory use, NOT physics."""
    fdm=_setup("REDDY",False,nx=nx,ns=ns)
    jets=MKRFieldJets(fdm.model.grid)
    rng=np.random.default_rng(20261116)
    field=rng.standard_normal((2,nx,ns))
    for _name,ax,az in fdm.model.derivative_orders:
        actual=jets.derivative(field,ax,az)
        reconstructed=np.einsum(
            "ij,cjk,lk->cil",fdm.axial[ax],field,fdm.angular[az],
            optimize=True)
        np.testing.assert_allclose(
            reconstructed,actual,rtol=2.e-12,atol=1.e-10)
    assert all(not m.flags.writeable for m in (
        *fdm.axial,*fdm.angular))


def test_insulated_source_compatibility_includes_prescribed_axial_end_flux():
    engine=_setup("KL",True)
    model=engine.model
    ends={side:[[{"kind":"ESSENTIAL","value":0.}]
                for _ in range(model.n_mechanical)] for side in ("left","right")}
    electric={"left":[{"kind":"NATURAL","value":0.}],
              "right":[{"kind":"NATURAL","value":model.grid.length_m}]}
    weights=(model.area_weights/np.sum(model.area_weights))[None,:,:]
    space=NativeEnergyEndSpace(engine,axial_orders=(1,)*model.n_mechanical,
        mechanical_ends=ends,electrical_ends=electric,
        electric_gauge=StrongElectricGauge(weights))
    charge=np.ones((1,*model.shape[1:]))
    space.check_electric_source(charge)
    with pytest.raises(ValueError,match="INCOMPATIBLE_INSULATED_CHARGE"):
        space.check_electric_source(None)
    with pytest.raises(ValueError,match="INCOMPATIBLE_INSULATED_CHARGE"):
        space.solve_static(np.zeros(model.shape),np.zeros_like(charge),
            electrical_charge=2.*charge,reduced_state_scale=1.,
            reduced_equation_scale=1.)
