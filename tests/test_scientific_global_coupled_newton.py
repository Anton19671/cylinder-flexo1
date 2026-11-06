"""Publication-gate algebra for GLOBAL coupled shell enthalpy and Newton blocks.

Tests run the existing seven-theory variational assembler with its real
mechanical/electrical boundary work, trace constraints, and Newmark effective
operator. No physical shell PDE solve, long simulation, or GUI interaction.

Source: Krysko-Zhong corrected 24-Oct-2026 CUF TE1-TE4 manuscript;
local material identities are checked separately. This suite verifies
DISCRETE global Gateaux work and mixed Jacobian reciprocity only.
"""
from __future__ import annotations

import numpy as np
import pytest

from cylinder_paths import activate
activate()

from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_functional_assembly import CylinderFunctional, surface_sbp21
from flexo_gradient_g3d import G3DMaterial

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")
MODES=("G3D","Gz")


def _problem(theory,mode,*,geometry="GREEN_LAGRANGE",kind="STATIC",bc=None):
    ops=surface_sbp21(4,5,1.3,2.)
    material=G3DMaterial(125.,.27,.075,.018,-.009,2.6,
                         gradient_model=mode)
    # Three points deliberately test *consistency of one discrete functional*,
    # not full through-thickness integration accuracy for higher TE orders.
    functional=CylinderFunctional(
        theory,material,radius_m=2.,thickness_m=.18,
        surface_operators=ops,n_gauss=3,geometry_model=geometry)
    if bc is None:
        bc={"mechanical":{
                "left":{"support":"CLAMPED"},
                "right":{"traction":[.005]*functional.n_mechanical,
                         "spring":[.04]*functional.n_mechanical}},
            "electrical":{"inner":{"mode":"POTENTIAL","value":0.},
                          "outer":{"mode":"CHARGE","value":.013}}}
    return BoundarySpatialProblem(functional,bc_spec=bc,
                                  calculation_kind=kind)


def _fields(problem,seed):
    rng=np.random.default_rng(seed)
    f=problem.functional
    nx,ns=f.ops.shape
    q=rng.normal(0.,.00017,(f.n_mechanical,nx,ns))
    p=rng.normal(0.,.004,(2,nx,ns))
    dq=rng.normal(0.,.008,q.shape)
    dp=rng.normal(0.,.09,p.shape)
    return q,p,dq,dp


def _work(residual,dq,dp):
    return float(np.dot(residual,np.concatenate((dq.ravel(),dp.ravel()))))


def _base_residual(problem,q,p,**kwargs):
    energy,residual,_=problem.evaluate_effective_residual(q,p,**kwargs)
    return energy,residual


@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("mode",MODES)
def test_all_seven_theories_full_boundary_energy_and_mixed_tangent(theory,mode):
    problem=_problem(theory,mode)
    q,p,dq,dp=_fields(problem,3040+THEORIES.index(theory))
    energy,residual=_base_residual(problem,q,p)
    h=2.e-4
    ep=_base_residual(problem,q+h*dq,p+h*dp)[0]
    em=_base_residual(problem,q-h*dq,p-h*dp)[0]
    analytic=_work(residual,dq,dp)
    np.testing.assert_allclose((ep-em)/(2*h),analytic,
                               rtol=5.e-6,atol=2.e-7)
    # Independent mixed blocks: dR_mech/dphi and dR_elec/dq.
    # No numerical symmetry is enforced or substituted into the solver.
    qsize=q.size
    rp=_base_residual(problem,q,p+h*dp)[1][:qsize]
    rm=_base_residual(problem,q,p-h*dp)[1][:qsize]
    direct=float(dq.ravel()@(rp-rm)/(2*h))
    rp=_base_residual(problem,q+h*dq,p)[1][qsize:]
    rm=_base_residual(problem,q-h*dq,p)[1][qsize:]
    converse=float(dp.ravel()@(rp-rm)/(2*h))
    np.testing.assert_allclose(direct,converse,
                               rtol=3.e-4,atol=3.e-7)
    assert np.isfinite((energy,analytic,direct,converse)).all()


@pytest.mark.parametrize("theory",("KL","REDDY","CUF_TE4"))
@pytest.mark.parametrize("mode",MODES)
def test_constrained_newton_tangent_is_reciprocal_in_admissible_space(
    theory,mode
):
    # Fully insulated electrical boundary: one global electric gauge,
    # NOT one gauge for every thickness coefficient.
    spec={"mechanical":{"left":{"support":"CLAMPED"}}}
    problem=_problem(theory,mode,bc=spec)
    q,p,_,_=_fields(problem,601+THEORIES.index(theory))
    assembly=problem.assemble(q,p)
    space=assembly.space
    assert assembly.edge.gauge_required
    assert "electric volume-mean gauge" in space.labels
    np.testing.assert_allclose(space.matrix@space.basis,0.,
                               rtol=0.,atol=2.e-11)
    seed=np.random.default_rng(234+THEORIES.index(theory))
    u=space.origin+space.basis@seed.normal(0,7.e-5,space.basis.shape[1])
    d1=space.basis@seed.normal(0,.004,space.basis.shape[1])
    d2=space.basis@seed.normal(0,.006,space.basis.shape[1])
    np.testing.assert_allclose(space.matrix@d1,0.,atol=3.e-11)
    np.testing.assert_allclose(space.matrix@d2,0.,atol=3.e-11)
    h=1.e-3
    def residual(state):
        qm=state[:q.size].reshape(q.shape)
        pm=state[q.size:].reshape(p.shape)
        return _base_residual(problem,qm,pm)[1]
    derivative_12=float(d1@(residual(u+h*d2)-residual(u-h*d2))/(2*h))
    derivative_21=float(d2@(residual(u+h*d1)-residual(u-h*d1))/(2*h))
    np.testing.assert_allclose(derivative_12,derivative_21,
                               rtol=4.e-4,atol=4.e-7)


@pytest.mark.parametrize("theory",("KL","REDDY","CUF_TE4"))
@pytest.mark.parametrize("mode",MODES)
def test_effective_newmark_mixed_tangent_and_zero_electric_inertia(theory,mode):
    problem=_problem(theory,mode,kind="DYNAMIC")
    q,p,dq,dp=_fields(problem,901+THEORIES.index(theory))
    rng=np.random.default_rng(928)
    density=850.
    dt=.018
    beta=.25
    gamma=.5
    predictor=q+rng.normal(0.,1.e-5,q.shape)
    velocity_predictor=rng.normal(0.,2.e-4,q.shape)
    def effective(qm,pm):
        acceleration=(qm-predictor)/(beta*dt**2)
        velocity=velocity_predictor+gamma*dt*acceleration
        return _base_residual(problem,qm,pm,time_s=.04,
                              acceleration=acceleration,velocity=velocity,
                              density_kg_m3=density)
    h=2.e-4
    qsize=q.size
    static_phi=_base_residual(problem,q,p,time_s=.04,
                             acceleration=np.zeros_like(q),
                             density_kg_m3=density)[1][qsize:]
    dynamic_phi=effective(q,p)[1][qsize:]
    np.testing.assert_allclose(dynamic_phi,static_phi,rtol=0.,atol=1.e-9)
    mech_plus=effective(q,p+h*dp)[1][:qsize]
    mech_minus=effective(q,p-h*dp)[1][:qsize]
    elec_plus=effective(q+h*dq,p)[1][qsize:]
    elec_minus=effective(q-h*dq,p)[1][qsize:]
    direct=float(dq.ravel()@(mech_plus-mech_minus)/(2*h))
    converse=float(dp.ravel()@(elec_plus-elec_minus)/(2*h))
    np.testing.assert_allclose(direct,converse,
                               rtol=4.e-4,atol=3.e-7)


@pytest.mark.parametrize("geometry",("LINEAR","VON_KARMAN","GREEN_LAGRANGE"))
@pytest.mark.parametrize("mode",MODES)
def test_global_cuf_te4_geometry_switch_preserves_coupled_work(geometry,mode):
    problem=_problem("CUF_TE4",mode,geometry=geometry)
    q,p,dq,dp=_fields(problem,1200+len(geometry))
    h=1.e-4
    _,r=_base_residual(problem,q,p)
    ep=_base_residual(problem,q+h*dq,p+h*dp)[0]
    em=_base_residual(problem,q-h*dq,p-h*dp)[0]
    np.testing.assert_allclose((ep-em)/(2*h),_work(r,dq,dp),
                               rtol=6.e-6,atol=3.e-7)
