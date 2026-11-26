"""Actual strong FDM assembly, manufactured BVPs and algebra only."""
import math
import numpy as np
import pytest
from scipy.linalg import solve

from cylinder_direct_strong_fdm import (
    DirectStrongGrid,direct_strong_jet_balance,
)
from cylinder_direct_strong_boundary import (
    assemble_physical_end_rows,native_kinematic_normal_trace,
)
from cylinder_mkr_field_jets import MKRFieldJets
from cylinder_strong_axial_green_audit import audit_native_higher_axial_green


def grid(n):
    return DirectStrongGrid(n,4,1.,2.,axial_scheme="MATCHED_QUARTIC_V1")


@pytest.mark.parametrize("n",(20,21,41,61))
def test_quartic_exactness_and_actual_green_identity(n):
    g=grid(n);op=g.matched_axial;x=np.linspace(0.,1.,n);h=g.dx
    assert np.all(g.axial_weights>0)
    np.testing.assert_allclose(sum(g.axial_weights),1.,atol=1.e-14)
    for order in (1,2,3):
        for a in (op.kinematic[order],op.coefficient[order]):
            assert n-np.linalg.matrix_rank(a,tol=1.e-10)==order
            for p in range(5):
                exact=(math.prod(range(p-order+1,p+1))*x**(p-order)
                       if p>=order else np.zeros_like(x))
                np.testing.assert_allclose(a@x**p/h**order,exact,
                                           atol=3.e-8,rtol=3.e-8)
    audit=audit_native_higher_axial_green(g)
    assert max(audit.relative_defect)<1.e-12


@pytest.mark.parametrize("n",(41,42,61))
def test_quartic_nyquist(n):
    g=grid(n);q=(-1.)**np.arange(n)
    for order,amplitude in enumerate((-2.,-4.,8.),1):
        matrix=g.matched_axial.kinematic[order]
        np.testing.assert_allclose((matrix@q)[18:-18],amplitude*q[18:-18],
                                   atol=2.e-13)


def _bvp_error(n,m,essential,variable):
    g=grid(n);x=np.linspace(0.,1.,n)
    q=np.broadcast_to(np.eye(n)[:,:,None],(n,n,g.ns)).copy()
    a=1.+.3*x if variable else np.ones(n)
    reaction=np.pi**(2*m)
    dm=MKRFieldJets(g).derivative(q,m,0)
    mc=np.stack((reaction*q,a[None,:,None]*dm))
    balance=direct_strong_jet_balance(mc,(("q",0,0),("q_m",m,0)),g)
    ends={side:[[{"kind":"ESSENTIAL" if k==0 or essential else "NATURAL",
                  "value":0.} for k in range(m)] for i in range(n)]
          for side in ("left","right")}
    system=assemble_physical_end_rows(balance,q,g,axial_orders=(m,)*n,
                                      mechanical_ends=ends)
    L=system.mechanical[:,:,0].T.copy()
    d=lambda k:np.exp(x)+2.**k*np.sin(2*x+k*np.pi/2)
    flux=lambda k:a*d(m+k)+(k*.3*d(m+k-1) if variable else 0.)
    f=(-1)**m*flux(m)+reaction*d(0)
    for node,normal in ((0,-1),(-1,1)):
        for k in range(m):
            row=k if node==0 else -1-k
            if k==0 or essential:
                f[row]=normal**k*d(k)[node]
            else:
                f[row]=(-1)**(m-k-1)*normal**(k+1)*flux(m-k-1)[node]
    row_scale=np.linalg.norm(L,axis=1)
    y=solve(L/row_scale[:,None],f/row_scale)
    return np.max(np.abs(y-d(0)))


@pytest.mark.parametrize("m",(1,2,3))
@pytest.mark.parametrize("essential",(False,True))
@pytest.mark.parametrize("variable",(False,True))
def test_manufactured_strong_boundary_value_convergence(m,essential,variable):
    errors=[_bvp_error(n,m,essential,variable) for n in (21,41,81)]
    assert errors[1]<.65*errors[0]
    assert errors[2]<.65*errors[1]


def test_scheme_bounds_and_legacy_default():
    assert DirectStrongGrid(7,4,1.,2.).axial_scheme=="LEGACY_TRAPEZOID"
    for n in (19,258):
        with pytest.raises(ValueError,match="20_TO_257"):
            grid(n)
    with pytest.raises(ValueError,match="UNKNOWN_AXIAL_SCHEME"):
        DirectStrongGrid(21,4,1.,2.,axial_scheme="unknown")
