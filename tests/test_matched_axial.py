import math
import numpy as np
import pytest
from cylinder_matched_axial import matched_axial_differences, matched_axial_green_defects


@pytest.mark.parametrize('nx',(15,16,21,31,51,81))
def test_green_polynomial_kernel(nx):
    op=matched_axial_differences(nx)
    x=np.linspace(0.,1.,nx); h=1./(nx-1)
    assert np.all(op.weights>0)
    assert abs(h*sum(op.weights)-1.)<1.e-14
    assert max(matched_axial_green_defects(nx))<3.e-14
    for order in (1,2,3):
        for matrix in (op.kinematic[order],op.coefficient[order]):
            for power in range(4):
                exact=(math.prod(range(power-order+1,power+1))*x**(power-order)
                       if power>=order else np.zeros_like(x))
                np.testing.assert_allclose(matrix@x**power/h**order,exact,
                                           atol=4.e-9,rtol=4.e-9)
            assert nx-np.linalg.matrix_rank(matrix,tol=1.e-10)==order


@pytest.mark.parametrize('nx',(41,42,81))
def test_no_axial_checkerboard_and_curvature_sign(nx):
    op=matched_axial_differences(nx)
    q=(-1.)**np.arange(nx)
    # Three compositions can extend the boundary closure by 2*4 nodes.
    interior=slice(14,-14)
    for order,amplitude in enumerate((-2.,-4.,8.),1):
        np.testing.assert_allclose((op.kinematic[order]@q)[interior],
                                   amplitude*q[interior],atol=2.e-13)


def test_refinement_including_both_boundaries():
    errors=[]
    for nx in (21,41,81,161):
        op=matched_axial_differences(nx)
        u=np.exp(np.linspace(0.,1.,nx));h=1./(nx-1)
        errors.append([max(np.max(np.abs(a@u/h**k-u)) for a in
                       (op.kinematic[k],op.coefficient[k])) for k in (1,2,3)])
    observed=np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert np.all(observed>np.array([2.8,1.8,.85]))


def test_size_and_immutable_cache():
    for n in (14,258,True):
        with pytest.raises(ValueError):
            matched_axial_differences(n)
    o=matched_axial_differences(21)
    for a in (*o.kinematic,*o.coefficient,o.weights,o.boundary,
              o.second_displacement_flux,o.third_displacement_flux,o.third_slope_flux):
        assert not a.flags.writeable
