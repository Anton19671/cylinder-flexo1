"""Research candidate gates, not a full coupled BVP certification."""
import math
import numpy as np
import pytest
from scipy.linalg import eigh
from independent_mvi.coupled_axial_sbp84_pair import (
    coupled_mvi_sbp84_pair,SCHEME)
from independent_mvi.coupled_operators import coupled_mvi_sbp42
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,NativeCoupledMVISurface,CoupledMVIWork,CoupledMVIFactors)
from independent_mvi.coupled_discrete_green import coupled_mvi_discrete_green_identity
from independent_mvi.coupled_dynamic_green import audit_mvi_dynamic_discrete_green
from independent_mvi.coupled_inertia import NativeCoupledMVIInertia
from independent_mvi.coupled_viscosity import NativeCoupledMVIViscosity
from flexo_gradient_g3d import G3DMaterial

THEORIES=("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")

@pytest.mark.parametrize("n",(21,31,41,81))
def test_pair_moments_and_green(n):
    P,M,H=coupled_mvi_sbp84_pair(n,2.7)
    B=np.zeros((n,n));B[0,0]=-1;B[-1,-1]=1
    np.testing.assert_allclose(P.T*H[None,:],
        B-H[:,None]*M,rtol=0.,atol=2e-15)
    x=np.linspace(-1.,1.,n)
    for k in range(5):
        exact=np.zeros(n) if k==0 else k*x**(k-1)/1.35
        np.testing.assert_allclose(P@x**k,exact,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(M@x**k,exact,rtol=0.,atol=1e-12)
    assert np.all(H>0)
    assert not P.flags.writeable

def test_legacy_is_bitwise_identical():
    g=CoupledMVISurfaceGrid(13,5,2.,1.)
    D,H=coupled_mvi_sbp42(13,2.)
    assert np.array_equal(g.dx,D) and np.array_equal(g.wx,H)
    assert g.axial_scheme=="SBP42"
    s=NativeCoupledMVISurface("CUF_TE1",G3DMaterial(10.,0.,.01,0.,0.,0.),
        grid=g,thickness_m=.1,n_gauss=2,geometry_model="LINEAR")
    for k in range(4):
        assert np.array_equal(s._dx_power[k],np.linalg.matrix_power(D,k))
    with pytest.raises(ValueError):
        CoupledMVISurfaceGrid(19,5,2.,1.,axial_scheme=SCHEME)

def _scalar(n,ell):
    P,M,H=coupled_mvi_sbp84_pair(n,1.)
    B=P[:,1:-1];C=(M@P)[:,1:-1];W=H[1:-1]
    K=B.T@(H[:,None]*B)+ell**2*C.T@(H[:,None]*C)
    A=K/np.sqrt(W[:,None]*W[None,:])
    return A,W

def test_continuum_initial_acceleration_and_static_converge():
    errors=[];statics=[]
    for n in (21,41,81,161):
        A,W=_scalar(n,.2);x=np.linspace(0.,1.,n)[1:-1]
        u=np.sin(np.pi*x);z=np.sqrt(W)*u
        lam=np.pi**2+.04*np.pi**4
        errors.append(np.linalg.norm(A@z/np.sqrt(W)-lam*u)/np.linalg.norm(lam*u))
        q=np.linalg.solve(A,lam*z)/np.sqrt(W)
        statics.append(np.linalg.norm(q-u)/np.linalg.norm(u))
    assert errors[0]<.1 and errors[-1]<.005
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert all(b<a for a,b in zip(statics,statics[1:]))
    assert statics[0]<1.e-4

@pytest.mark.parametrize("n",(41,81))
@pytest.mark.parametrize("ell",(0.,.001,.02,.2,1.))
def test_first_four_axial_modes_not_contaminated(n,ell):
    A,W=_scalar(n,ell)
    ev,V=eigh(A,subset_by_index=(0,3))
    k=np.pi*np.arange(1,5)
    np.testing.assert_allclose(np.sqrt(ev/(k*k+ell**2*k**4)),1.,rtol=5.e-4,atol=0.)
    x=np.linspace(0.,1.,n)[1:-1]
    for j in range(4):
        target=np.sqrt(W)*np.sin(k[j]*x)
        overlap=abs(V[:,j]@target)/np.linalg.norm(target)
        assert overlap>.99

@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","GZ"))
def test_actual_coupled_gl_dynamic_green_all_models(theory,gradient):
    g=CoupledMVISurfaceGrid(21,5,1.2,2.,axial_scheme=SCHEME)
    s=NativeCoupledMVISurface(theory,
        G3DMaterial(1.e6,.22,.02,1.e-4,-5.e-5,1.e-6,gradient),
        grid=g,thickness_m=.08,n_gauss=2,potential_order=1,
        geometry_model="GREEN_LAGRANGE")
    rng=np.random.default_rng(1201)
    x=rng.normal(size=(s.n_fields,2,g.nx))*1.e-6
    x[:,:,0]=0.;x[:,:,-1]=0.
    y=rng.normal(size=(s.n_fields,2,g.ns))
    f=CoupledMVIFactors(x,y)
    q=s.reconstruct(f)[:s.n_mechanical]
    v=.05*q;a=-.03*q
    inertia=NativeCoupledMVIInertia(s,density_kg_m3=1000.)
    viscosity=NativeCoupledMVIViscosity(s,eta_s=.01)
    out=audit_mvi_dynamic_discrete_green(s,f,v,a,inertia=inertia,viscosity=viscosity)
    assert out.total.identity_relative_error<2.e-9
    assert out.electric_nodal_work_error==0.
    assert out.total.status=="MVI_EXACT_DISCRETE_PAIRED_SBP84_FOURIER_GREEN_IDENTITY"
