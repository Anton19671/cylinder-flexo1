"""Analytic modal temporal verification, no full shell dynamics."""
import numpy as np
import pytest
from cylinder_gauss2_rk import Gauss2Settings, Gauss2State, gauss2_step
from cylinder_modal_runge_kutta_comparison import compare_modal_integrators, frequency_from_npz


def test_temporal_orders_and_undamped_energy():
    data = {n: compare_modal_integrators(269.50842161549195,
                                         steps_per_period=n, periods=10)["integrators"]
            for n in (50,100,200)}
    for method, order in (("NEWMARK",2),("RK4",4),("GAUSS_IRK4",4)):
        for large,small in ((50,100),(100,200)):
            quotient = (abs(data[large][method]["frequency_error_percent"]) /
                        abs(data[small][method]["frequency_error_percent"]))
            assert 2**(order-0.05) < quotient < 2**(order+0.05)
    assert abs(data[100]["NEWMARK"]["energy_change_percent"]) < 1.e-10
    assert abs(data[100]["GAUSS_IRK4"]["energy_change_percent"]) < 1.e-10
    assert data[100]["RK4"]["energy_change_percent"] < 0.


def test_gauss_irk4_uses_same_tableau_as_existing_stage_integrator():
    f=269.50842161549195
    omega=2*np.pi*f
    dt=1/(100*f)
    initial=Gauss2State(0.,np.array([[[1.]]]),np.array([[[0.]]]))
    result=gauss2_step(
        initial,Gauss2Settings(dt,max_unknowns=2),
        equations=lambda q,v,a,p,t:(a+omega**2*q,None),
        mechanical_state_scale=1.,mechanical_equation_scale=omega**2,
        acceleration_guess=np.full((2,1,1,1),-omega**2))
    comparison=compare_modal_integrators(f,periods=1,include_histories=True)
    state=comparison["histories"]["GAUSS_IRK4"][1]
    np.testing.assert_allclose(
        [result.final.mechanical.item(),result.final.velocity.item()/omega],
        state,rtol=1.e-12,atol=1.e-12)


def test_npz_reader_and_invalid_input(tmp_path):
    path=tmp_path/"modes.npz"
    np.savez(path,frequencies=np.array([269.5,280.]))
    assert frequency_from_npz(path,2)==280.
    with pytest.raises(ValueError):
        frequency_from_npz(path,3)
    with pytest.raises(ValueError):
        compare_modal_integrators(0.)
    with pytest.raises(ValueError):
        compare_modal_integrators(269.,steps_per_period=4)
