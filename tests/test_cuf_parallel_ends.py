"""Source end program and coupled manufactured equations; no shell load run."""
import numpy as np
import pytest

from cylinder_direct_strong_boundary import (
    CUFParallelEndConditions, assemble_physical_end_rows,
)
from cylinder_direct_strong_fdm import (
    DirectStrongGrid, direct_coefficient_derivative, direct_strong_jet_balance,
    direct_strong_from_functional,
)
from cylinder_direct_strong_newton import (
    StrongElectricGauge, StrongNewtonSettings, solve_strong_nonlinear_equations,
)
from cylinder_cuf_parallel_ends import CUFParallelEndProblem
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from flexo_gradient_g3d import G3DMaterial


def functional(order=1, electric=True, reference=False):
    material = G3DMaterial(80., .23, .1, .001 if electric else 0.,
                          .003 if electric else 0., .8 if electric else 0.)
    if reference:
        return CylinderFunctional(f"CUF_TE{order}", material, radius_m=2.,
            thickness_m=.2, surface_operators=surface_trapezoid(7, 4, 1., 2.),
            n_gauss=6)
    return DirectStrongShell(f"CUF_TE{order}",material,
        grid=DirectStrongGrid(7,4,1.,2.),thickness_m=.2,n_gauss=6)


@pytest.mark.parametrize("order", range(1, 5))
def test_all_cuf_coefficients_fixed_but_higher_traces_microfree(order):
    n = 3*(order+1)
    ends = CUFParallelEndConditions(order, (2,)*n, order)
    for side in ("left", "right"):
        assert len(ends.mechanical_ends[side]) == n
        for row in ends.mechanical_ends[side]:
            assert row == [{"kind": "ESSENTIAL", "value": 0.},
                           {"kind": "NATURAL", "value": 0.}]
        assert len(ends.electrical_ends[side]) == order+1
        assert all(row["kind"] == "NATURAL" for row in ends.electrical_ends[side])
    copy = ends.mechanical_ends
    copy["left"][0][1]["value"] = 9.
    assert copy["right"][0][1]["value"] == 0.
    assert ends.mechanical_ends["left"][0][1]["value"] == 0.


@pytest.mark.parametrize("order", range(1, 5))
def test_real_cuf_zero_state_and_constant_potential(order):
    f = functional(order)
    problem = CUFParallelEndProblem(f, length_m=1.,
        axial_orders=(2,)*f.n_mechanical, potential_order=order, mean_potential_v=3.)
    q = np.zeros(problem.mechanical_shape)
    p = np.zeros(problem.electrical_shape)
    p[0] = 3.
    result = problem.evaluate(q, p)
    np.testing.assert_allclose(result.mechanical, 0., atol=1.e-11)
    np.testing.assert_allclose(result.electrical, 0., atol=1.e-11)
    assert abs(result.mean_potential_v-3.) < 1.e-12
    assert abs(result.mean_potential_error_v) < 1.e-12
    # This checks the gauge-aware solve wiring at an exact zero-load state;
    # it does not launch a loaded-shell Newton iteration.
    solved = problem.solve(q, p, mechanical_state_scale=1.,
        mechanical_equation_scale=1., electrical_state_scale=1.,
        electrical_equation_scale=1., settings=StrongNewtonSettings(max_dofs=600))
    assert solved.converged and solved.iterations == 0
    assert solved.gauge_multiplier_scaled == 0.


def test_off_has_no_electrical_unknown_or_gauge():
    f = functional(electric=False)
    problem = CUFParallelEndProblem(f, length_m=1., axial_orders=(2,)*6)
    result = problem.evaluate(np.zeros(problem.mechanical_shape))
    assert result.electrical is None
    assert result.mean_potential_v is None
    assert problem.electric_gauge is None
    with pytest.raises(ValueError, match="must be None"):
        problem.evaluate(np.zeros(problem.mechanical_shape), np.zeros((1,7,4)))


def test_reject_wrong_orders_electric_state_and_quadrature():
    f = functional()
    for orders in ((2,)*5, (3,)*6, (True,)*6):
        with pytest.raises(ValueError, match="half-order"):
            CUFParallelEndProblem(f, length_m=1., axial_orders=orders, potential_order=1)
    with pytest.raises(ValueError, match="ON/OFF"):
        CUFParallelEndProblem(f, length_m=1., axial_orders=(2,)*6)
    with pytest.raises(ValueError, match="quadrature"):
        CUFParallelEndProblem(f, length_m=2., axial_orders=(2,)*6, potential_order=1)
    problem = CUFParallelEndProblem(f, length_m=1., axial_orders=(2,)*6, potential_order=1)
    with pytest.raises(ValueError, match="potential state"):
        problem.evaluate(np.zeros(problem.mechanical_shape), np.zeros((1,7,4)))
    # Mixed known orders remain explicit; no hidden second constraint for
    # a field whose model has only a second-order axial equation.
    mixed = CUFParallelEndConditions(1, (1,2,1,2,1,2))
    assert [len(x) for x in mixed.mechanical_ends["left"]] == [1,2,1,2,1,2]


def test_insulated_net_charge_rejected_but_higher_modes_not_individually_constrained():
    f = functional()
    charge = np.zeros((2, 7, 4))
    charge[0] = 1.
    with pytest.raises(ValueError, match="nonzero net charge"):
        CUFParallelEndProblem(f, length_m=1., axial_orders=(2,)*6,
                             potential_order=1, electrical_charge=charge)
    charge[0] = (np.arange(7)/6.-.5)[:,None]
    charge[1] = 1.
    problem = CUFParallelEndProblem(f, length_m=1., axial_orders=(2,)*6,
                                   potential_order=1, electrical_charge=charge)
    charge[:] = 99.
    assert np.all(problem.electrical_charge[1] == 1.)
    assert not problem.electrical_charge.flags.writeable


def test_physical_free_charge_has_same_sign_as_enthalpy_first_variation():
    f = functional(reference=True)
    q = np.zeros((6, 7, 4)); p = np.zeros((2, 7, 4))
    charge = np.ones_like(p)
    zero = f.evaluate(q, p, energy_breakdown=False)
    charged = f.evaluate(q, p, free_charge=charge, check_charge_balance=False,
                         energy_breakdown=False)
    expected = (charged.electrical_residual-zero.electrical_residual)/f.W
    direct = direct_strong_from_functional(f,q,p,length_m=1.,electrical_charge=charge)
    formal = f.formal_strong_balance(q,p,
        derivative=lambda a,ax,az: np.zeros_like(a), electrical_charge=charge)
    np.testing.assert_allclose(direct.electrical_bulk, expected, atol=1.e-13)
    np.testing.assert_allclose(formal.electrical, expected, atol=1.e-13)
    # Positive physical rho is balanced by div(D)>0, not div(D)<0.
    g = DirectStrongGrid(7,4,1.,2.)
    ec = np.zeros((3,1,7,4)); ec[1,0] = (np.arange(7)*g.dx)[:,None]
    out = direct_strong_jet_balance(np.zeros((1,1,7,4)), (("q",0,0),), g, ec,
                                    electrical_charge=np.ones((1,7,4)))
    np.testing.assert_allclose(out.electrical_bulk, 0., atol=1.e-12)


def test_polynomial_coupled_fourth_order_with_free_slopes_and_single_gauge():
    """Solve a tiny manufactured coupled 4th/2nd order system.

    W=0.5*q_xx**2 - 0.5*(p_x**2+p_s**2)+c*q_xx*p_x.
    The circumferential electric contribution is ESSENTIAL: otherwise
    each theta node carries its own independent constant-potential mode
    and a single global mean gauge cannot fix a unique electric state.
    Exact polynomials have q=0, q_xx=p_x=p_s=0 at both ends, while
    q_x=(1,-1); slopes remain physically free.
    """
    g = DirectStrongGrid(9, 4, 1., 2.)
    x = np.arange(g.nx)*g.dx
    shape = (1, g.nx, g.ns)
    field = lambda a: np.broadcast_to(a[None,:,None], shape).copy()
    qt = field(x**4-2*x**3+x)
    pt = field(x*x*(1-x)**2)
    wx = np.full(g.nx, g.dx); wx[[0,-1]] *= .5
    weights = np.broadcast_to(wx[None,:,None]/g.ns, shape).copy()
    pt += 2.-np.sum(weights*pt)
    c = .07
    load = field(np.full_like(x, 24.)+c*(-12.+24.*x))
    charge = field(-2.+12.*x-12.*x*x+c*(24.*x-12.))
    mechanical_ends = {side: [[{"kind":"ESSENTIAL", "value":0.},
                              {"kind":"NATURAL", "value":0.}]]
                       for side in ("left", "right")}
    electrical_ends = {side: [{"kind":"NATURAL", "value":0.}]
                       for side in ("left", "right")}
    def equations(q, p):
        qxx = direct_coefficient_derivative(q,2,0,g)
        px = direct_coefficient_derivative(p,1,0,g)
        ps = direct_coefficient_derivative(p,0,1,g)
        mc = np.stack((np.zeros_like(q), qxx+c*px))
        ec = np.stack((np.zeros_like(p), -px+c*qxx, -ps))
        balance = direct_strong_jet_balance(mc, (("q",0,0),("qxx",2,0)),
            g, ec, mechanical_load=load, electrical_charge=charge)
        rows = assemble_physical_end_rows(balance, q, g, axial_orders=(2,),
            mechanical_ends=mechanical_ends, electrical=p,
            electrical_ends=electrical_ends, gauge_closed_elsewhere=True)
        return rows.mechanical, rows.electrical
    for values in equations(qt, pt):
        np.testing.assert_allclose(values, 0., atol=2.e-9)
    zero = np.zeros(shape)
    result = solve_strong_nonlinear_equations(equations, zero, zero,
        mechanical_state_scale=1., mechanical_equation_scale=1.,
        electrical_state_scale=1., electrical_equation_scale=1.,
        electric_gauge=StrongElectricGauge(weights, target_v=2.),
        settings=StrongNewtonSettings(max_dofs=80, max_iterations=5,
            scaled_absolute_tolerance=2.e-6, relative_linear_tolerance=1.e-9))
    assert result.converged, result.status
    np.testing.assert_allclose(result.mechanical, qt, atol=1.e-6)
    np.testing.assert_allclose(result.electrical, pt, atol=1.e-6)
    slopes = direct_coefficient_derivative(result.mechanical,1,0,g)
    np.testing.assert_allclose(slopes[0,[0,-1],0], [1.,-1.], atol=1.e-5)
