"""Manufactured continuum boundary-work identities for the Krysko-Zhong shell.

Physical domain x in [-1,1], periodic s=R*theta. The manufactured
coefficients of virtual derivatives A_ab are polynomials in x times
low Fourier harmonics in s. All weak and interior integrals use
independent closed-form derivatives evaluated by Gaussian quadrature;
the axial boundary operator uses polynomial-exact / spectral
differentiation matrices.

This proves integration-by-parts flux algebra (including mixed q_xs,
q_xxs and q_xss) for up to third surface derivatives. It does NOT prove
a stable or convergent discrete edge treatment for the full shell PDE.
No physical computation or time marching is performed.
"""
import numpy as np
import pytest
from numpy.polynomial import Polynomial

from cylinder_paths import activate
activate()

from cylinder_g3d_natural_edges import axial_natural_end_tractions
from cylinder_functional_assembly import MECHANICAL_DERIVATIVES

RADIUS=1.7
NS=13
NX=9
X=np.linspace(-1.,1.,NX)
THETA=2*np.pi*np.arange(NS)/NS
AREA_S=2*np.pi*RADIUS/NS


def axial_derivative_matrix():
    # Construct exact nodal derivative on polynomials of order up to NX-1.
    v=np.vander(X,N=NX,increasing=True)
    dv=np.column_stack([
        np.zeros_like(X) if k==0 else k*X**(k-1)
        for k in range(NX)
    ])
    return np.linalg.solve(v.T,dv.T).T


def periodic_spectral_derivative():
    # Independent DFT construction, not the solver's SBP/Fourier builder.
    omega=np.fft.fftfreq(NS,d=1/NS)
    eye=np.eye(NS)
    return np.fft.ifft(
        1j*omega[:,None]*np.fft.fft(eye,axis=0),axis=0
    ).real/RADIUS


def coefficient_angular(derivative):
    return ((1/RADIUS)**derivative
            *np.sin(THETA+derivative*np.pi/2)
            +.23*(2/RADIUS)**derivative
            *np.cos(2*THETA+derivative*np.pi/2))


def virtual_angular(derivative):
    return ((1/RADIUS)**derivative
            *np.cos(THETA+derivative*np.pi/2)
            +.34*(2/RADIUS)**derivative
            *np.sin(2*THETA+derivative*np.pi/2))


VIRTUAL_POLY=Polynomial([1.12,-.31,.19,.066])


def coefficient_poly(index):
    return Polynomial([.47+.03*index,.19-.016*index,
                       .035*(index+1),-.008+.0002*index])


def _integrate(values, axial_weights):
    return float(np.sum(values*axial_weights[:,None])*AREA_S)


@pytest.mark.parametrize("highest_order",(6,10))
@pytest.mark.parametrize("selected_slot",(-1,0,1,2,3,4,5,6,7,8,9))
def test_complete_mechanical_integration_by_parts(highest_order,selected_slot):
    if selected_slot>=highest_order:
        pytest.skip("derivative is not present in this theory")
    orders=MECHANICAL_DERIVATIVES[:highest_order]
    a=np.zeros((highest_order,1,NX,NS))
    for i in range(highest_order):
        if selected_slot!=-1 and selected_slot!=i:
            continue
        a[i,0]=coefficient_poly(i)(X)[:,None]*coefficient_angular(0)[None,:]
    end=axial_natural_end_tractions(a,orders,
                                   axial_derivative_matrix(),
                                   periodic_spectral_derivative())

    # Independent continuum weak/bulk integrations, not the boundary formula.
    quadrature_x,quadrature_weights=np.polynomial.legendre.leggauss(10)
    weak=np.zeros((len(quadrature_x),NS))
    bulk=np.zeros_like(weak)
    for i,(_,ax,az) in enumerate(orders):
        if selected_slot!=-1 and selected_slot!=i:
            continue
        p=coefficient_poly(i)
        a_value=p(quadrature_x)[:,None]*coefficient_angular(0)[None,:]
        virtual_derivative=(VIRTUAL_POLY.deriv(ax)(quadrature_x)[:,None]
                            *virtual_angular(az)[None,:])
        weak+=a_value*virtual_derivative
        bulk+=((-1)**(ax+az))*(
            p.deriv(ax)(quadrature_x)[:,None]
            *coefficient_angular(az)[None,:]
            *VIRTUAL_POLY(quadrature_x)[:,None]
            *virtual_angular(0)[None,:]
        )

    # Physical normal derivatives: d_n = -d_x at the left,
    # +d_x at right; d_nn = d_xx at both ends.
    def end_work(end_,location,n_x):
        v0=VIRTUAL_POLY(location)*virtual_angular(0)
        vn=n_x*VIRTUAL_POLY.deriv(1)(location)*virtual_angular(0)
        vnn=VIRTUAL_POLY.deriv(2)(location)*virtual_angular(0)
        return (end_.traction[0]*v0
                +end_.normal_double_traction[0]*vn
                +end_.normal_triple_traction[0]*vnn)
    boundary=AREA_S*np.sum(
        end_work(end.right,1.,1.)
        +end_work(end.left,-1.,-1.))
    np.testing.assert_allclose(
        _integrate(weak,quadrature_weights),
        _integrate(bulk,quadrature_weights)+boundary,
        rtol=2e-10,atol=2e-9
    )
    if highest_order==6:
        np.testing.assert_array_equal(end.left.normal_triple_traction,0.)
        np.testing.assert_array_equal(end.right.normal_triple_traction,0.)


def test_electric_end_flux_is_exact_variation_of_axial_potential_work():
    dx=axial_derivative_matrix()
    ds=periodic_spectral_derivative()
    a=np.zeros((3,1,NX,NS))
    electric=np.zeros((3,1,NX,NS))
    electric_poly=Polynomial([.8,.16,-.04])
    potential_poly=Polynomial([.35,.18,.21])
    electric[1,0]=electric_poly(X)[:,None]*coefficient_angular(0)[None,:]
    end=axial_natural_end_tractions(a,MECHANICAL_DERIVATIVES[:3],dx,ds,electric)
    np.testing.assert_allclose(
        end.left.electric_charge_flux[0],
        -electric[1,0,0,:],atol=1e-14,rtol=0)
    np.testing.assert_allclose(
        end.right.electric_charge_flux[0],
        electric[1,0,-1,:],atol=1e-14,rtol=0)

    x,w=np.polynomial.legendre.leggauss(10)
    local=(
        electric_poly(x)[:,None]*coefficient_angular(0)[None,:]
        *potential_poly.deriv()(x)[:,None]*virtual_angular(0)[None,:])
    interior=-(
        electric_poly.deriv()(x)[:,None]*coefficient_angular(0)[None,:]
        *potential_poly(x)[:,None]*virtual_angular(0)[None,:])
    boundary=AREA_S*np.sum(
        end.left.electric_charge_flux[0]
        *potential_poly(-1.)*virtual_angular(0)
        +end.right.electric_charge_flux[0]
        *potential_poly(1.)*virtual_angular(0))
    np.testing.assert_allclose(_integrate(local,w),
                               _integrate(interior,w)+boundary,
                               rtol=2e-10,atol=2e-9)
