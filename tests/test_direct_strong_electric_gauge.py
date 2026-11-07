"""Algebraic gauge closure and analytical cylindrical CUF volume moments."""
import numpy as np
import pytest
from cylinder_direct_strong_newton import (
    StrongElectricGauge, StrongNewtonSettings, cuf_mean_potential_gauge,
    solve_strong_nonlinear_equations, solve_direct_strong_reference,
)


@pytest.mark.parametrize('order',range(5))
def test_cuf_gauge_matches_independent_thickness_quadrature(order):
    area=np.arange(1,13,dtype=float).reshape(3,4)
    R,h=2.,.4
    gauge=cuf_mean_potential_gauge(area,radius_m=R,thickness_m=h,
                                  potential_order=order)
    xi,wi=np.polynomial.legendre.leggauss(12)
    for k in range(order+1):
        moment=np.sum(wi*(1+h*xi/(2*R))*xi**k)/2
        np.testing.assert_allclose(gauge.weights[k],area/area.sum()*moment,
                                   atol=1.e-15)


def solve_gauge_case(charge_shift=0.):
    q=np.zeros((1,3,4));p=np.full_like(q,7.)
    target=np.arange(p.size,dtype=float).reshape(p.shape)
    target-=target.mean()
    def equations(q,p):
        # A singular electrical operator whose sole null vector is constant.
        return q-1.,p-p.mean()-target+charge_shift
    return solve_strong_nonlinear_equations(
        equations,q,p,mechanical_state_scale=1.,mechanical_equation_scale=1.,
        electrical_state_scale=1.,electrical_equation_scale=1.,
        electric_gauge=StrongElectricGauge(np.full_like(p,1./p.size),target_v=2.),
        settings=StrongNewtonSettings(max_dofs=25,scaled_absolute_tolerance=1.e-7)),target


def test_one_gauge_preserves_all_electrical_equations():
    result,target=solve_gauge_case()
    assert result.converged,result.status
    np.testing.assert_allclose(result.mechanical,1.,atol=1.e-7)
    np.testing.assert_allclose(result.electrical,target+2.,atol=1.e-7)
    assert abs(result.gauge_multiplier_scaled)<1.e-7
    assert result.gauge_defect_scaled<1.e-7


def test_multiplier_cannot_hide_incompatible_electrical_source():
    result,_=solve_gauge_case(charge_shift=.1)
    assert not result.converged
    assert result.status=='ELECTRIC_GAUGE_COMPATIBILITY_FAILED'
    assert abs(result.gauge_multiplier_scaled)>1.


def test_grounded_flag_does_not_replace_an_equation():
    with pytest.raises(ValueError,match='flag does not impose'):
        solve_direct_strong_reference(None,np.zeros((1,7,4)),
            length_m=1.,axial_orders=(1,),mechanical_ends={},
            grounded_electric_elsewhere=True,
            mechanical_state_scale=1.,mechanical_equation_scale=1.)
