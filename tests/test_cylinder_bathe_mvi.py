"""Independent Bathe gamma=2-sqrt(2) predictor and variational-step tests."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
from unittest.mock import patch

from cylinder_bathe_mvi import (
    BATHE_GAMMA,bathe_coefficients,first_predictors,second_predictors,
    audit_bathe_kinematics,solve_bathe_variational_step,
)


def test_equal_effective_mass_coefficient():
    for h in (.002,.001,.0005,.0001):
        c=bathe_coefficients(h)
        assert math.isclose(c.gamma,2-math.sqrt(2),rel_tol=0,abs_tol=1e-15)
        assert math.isclose(c.kappa,c.c3*c.c3,rel_tol=3e-15)
        assert math.isclose(c.c3*h,2+math.sqrt(2),rel_tol=3e-15)
        assert math.isclose(c.c1+c.c2+c.c3,0.,abs_tol=3e-12/h)


def test_trapezoidal_and_three_point_identities_on_arbitrary_fields():
    rng=np.random.default_rng(27)
    q=rng.normal(size=(3,5,7))
    v=rng.normal(size=q.shape)
    a=rng.normal(size=q.shape)
    c=bathe_coefficients(.002)
    qp,vp=first_predictors(q,v,a,c)
    qg=qp+np.ones_like(q)*3.e-5
    ag=4.*(qg-qp)/(c.delta*c.delta)
    vg=vp+.5*c.delta*ag
    expected_q=q+.5*c.delta*(v+vg)
    expected_v=v+.5*c.delta*(a+ag)
    np.testing.assert_allclose(qg,expected_q,atol=1e-12,rtol=2e-11)
    np.testing.assert_allclose(vg,expected_v,atol=1e-12,rtol=2e-11)
    qpred,vpred=second_predictors(q,v,qg,vg,c)
    qe=np.array(qg+.02*rng.normal(size=q.shape))
    ae=c.kappa*(qe-qpred)
    ve=vpred+.5*c.delta*ae
    np.testing.assert_allclose(ve,c.c1*q+c.c2*qg+c.c3*qe,
                               atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(ae,c.c1*v+c.c2*vg+c.c3*ve,
                               atol=2e-8,rtol=2e-10)
    r1=SimpleNamespace(mechanical=qg,velocity=vg,acceleration=ag)
    r2=SimpleNamespace(mechanical=qe,velocity=ve,acceleration=ae)
    errors=audit_bathe_kinematics(q,v,a,r1,r2,c)
    assert all(entry["relative"]<1.e-9 for entry in errors.values())


def test_bathe_zero_step_rejected():
    for dt in (0.,-1.e-3,np.nan,np.inf):
        with pytest.raises(ValueError,match="BATHE_TIME_STEP_MUST_BE_POSITIVE"):
            bathe_coefficients(dt)


def test_two_actual_variational_calls_separate_and_preserve_full_velocity():
    rng=np.random.default_rng(91)
    q=np.zeros((3,5,7))
    v=rng.normal(size=q.shape)*1e-5
    a=rng.normal(size=q.shape)*1e-4
    dt=.002
    c=bathe_coefficients(dt)
    calls=[]
    def physical_mock(problem, mechanical, electrical, **kwargs):
        calls.append(kwargs)
        qp=kwargs["q_predictor"]
        vp=kwargs["velocity_predictor"]
        if len(calls)==1:
            qnew=qp+1.e-7*np.ones_like(qp)
        else:
            qnew=qp-1.e-7*np.ones_like(qp)
        anew=(qnew-qp)*c.kappa
        vnew=vp+anew/c.c3
        return SimpleNamespace(mechanical=qnew,electrical=electrical,
             velocity=vnew,acceleration=anew,converged=True,
             status="CONVERGED_FULL",x_profiles=np.zeros((3,1,5)),
             y_profiles=np.zeros((3,1,7)))
    with patch("cylinder_bathe_mvi.solve_dynamic_mvi",side_effect=physical_mock):
        out=solve_bathe_variational_step(
            SimpleNamespace(kind="DYNAMIC"),q,None,v,a,time_s=.03,h=dt,
            density_kg_m3=7800.,
            loads_at=lambda t:(np.full_like(q,t),None),
            method="MVI",mvi_settings=SimpleNamespace(rank=1))
    assert len(calls)==2
    assert out.final.status=="CONVERGED_FULL"
    assert math.isclose(calls[0]["time_s"],.03+c.delta)
    assert math.isclose(calls[1]["time_s"],.032)
    assert calls[0]["dt_s"]==calls[1]["dt_s"]==c.delta
    assert calls[0]["beta"]==calls[1]["beta"]==.25
    assert calls[0]["gamma"]==calls[1]["gamma"]==.5
    assert calls[0]["dead_load"].shape==q.shape
    assert calls[1]["dead_load"].shape==q.shape
    assert calls[1]["previous_mvi_result"] is not None


def test_free_oscillator_bathe_is_second_order_and_damps_high_frequency():
    omega=40.
    def evolve(h,final_time):
        c=bathe_coefficients(h)
        q,v,a=1.,0.,-omega*omega
        for i in range(round(final_time/h)):
            qp=q+c.delta*v+.25*c.delta**2*a
            # q_pred already contains +delta^2*a_n/4; subtracting a_n
            # here would count the initial acceleration twice.
            qg=c.kappa*qp/(c.kappa+omega*omega)
            ag=c.kappa*(qg-qp)
            vg=v+.5*c.delta*(a+ag)
            qpred,vpred=second_predictors(
                np.array([[[q]]]),np.array([[[v]]]),
                np.array([[[qg]]]),np.array([[[vg]]]),c)
            qe=float(c.kappa*qpred[0,0,0]/(c.kappa+omega*omega))
            ae=c.kappa*(qe-qpred[0,0,0])
            ve=vpred[0,0,0]+ae/c.c3
            q,v,a=qe,ve,ae
        return q,v
    t=.2
    e=[]
    for h in (.01,.005,.0025,.00125):
        q,v=evolve(h,t)
        e.append(abs(q-math.cos(omega*t)))
    assert e[2]<e[1]<e[0]
    assert e[3]<e[2]
    assert e[2]/e[3]>3.
    # For an underresolved frequency Bathe must damp, unlike trapezoidal.
    q,v=evolve(.04,.2)
    assert (q*q+(v/omega)**2)<1.
