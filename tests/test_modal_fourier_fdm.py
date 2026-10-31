"""No-shell-run checks of complete Fourier-sector eigenvalue selection."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from cylinder_modal_fourier_fdm import can_use_fourier_modal_fdm, solve_modal_fourier_fdm


def _fixture():
    nx=ns=9
    k=np.fft.fftfreq(ns,d=1/ns)
    d=np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(ns),axis=0),axis=0).real
    angular=-d@d
    omega=SimpleNamespace(Ds=d,ws=np.ones(ns),
        wx=np.array([17/48,1.,1.,1.,1.,1.,1.,1.,17/48]))
    f=SimpleNamespace(geometry_model='LINEAR',
        material=SimpleNamespace(ell_m=0.),R=1.,ops=omega,
        theory='SYNTHETIC_FOURIER')
    q=np.zeros((1,nx,ns),dtype=float)
    fixed=np.array([np.ravel_multi_index((0,i,j),q.shape)
                    for i in (0,nx-1) for j in range(ns)])
    c=sparse.csr_matrix((np.ones(len(fixed)),
        (np.arange(len(fixed)),fixed)),shape=(len(fixed),q.size))
    projector=SimpleNamespace(c=c,count=len(fixed),size=q.size,
        target=np.zeros(len(fixed)),lift=np.zeros(q.size))
    prepared=SimpleNamespace(method='FDM',fdm_backend='MATRIX_FREE_GMRES',
        mechanical=q,electrical=None,
        problem=SimpleNamespace(functional=f))
    a=10+5*np.arange(nx)
    mask=(c.getnnz(axis=0)==0).reshape(q.shape)
    def action(x,stiffness):
        field=np.asarray(x).reshape(q.shape)
        out=np.zeros_like(field)
        if stiffness:
            out[0,1:-1,:]=(a[1:-1,None]*field[0,1:-1,:]
                              +field[0,1:-1,:]@angular)
        else:
            out[mask]=field[mask]
        return out.ravel()
    def op(stiffness):
        return LinearOperator((q.size,q.size),dtype=float,
            matvec=lambda x:action(x,stiffness),
            matmat=lambda x:np.column_stack(
                [action(x[:,j],stiffness) for j in range(x.shape[1])]))
    return prepared,projector,op(True),op(False)


def test_all_fourier_sectors_produce_exact_lowest_modes():
    prepared,projector,k,m=_fixture()
    assert can_use_fourier_modal_fdm(prepared,projector)
    result=solve_modal_fourier_fdm(prepared,1.,8,projector,k,m,
        np.ones(projector.size),1.,1.)
    expected=np.sqrt(np.array([15,16,16,19,19,20,21,21]))/(2*np.pi)
    np.testing.assert_allclose(result['frequencies_hz'],expected,rtol=1e-10)
    assert result['circumferential_wavenumbers']==[0,1,1,2,2,0,1,1]
    assert result['global_spectrum_completeness_verified'] is True
    vectors=result['mechanical_modes'].reshape(8,-1)
    np.testing.assert_allclose(vectors@vectors.T,np.eye(8),atol=1e-10)


def test_circumferential_nonuniformity_disables_shortcut():
    prepared,projector,k,m=_fixture()
    prepared.problem.functional.ops.ws[3]=2.
    assert not can_use_fourier_modal_fdm(prepared,projector)
