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


def test_cross_component_circumferential_derivative_needs_both_phases():
    """An analytic symmetric K with a skew angular u-v coupling.

    The off-diagonal K_uv=beta*D and K_vu=-beta*D is SELF-adjoint for
    D.T=-D. Cosine-only Fourier assembly loses the i*n cross-component
    coupling entirely and therefore gives the wrong first doublet.
    This regression checks the actual full Fourier-block eigen solver,
    without using any shell equations or an FDM-generated target mode.
    """
    nx,ns=9,13
    ktheta=np.fft.fftfreq(ns,d=1/ns)
    derivative=np.fft.ifft(
        1j*ktheta[:,None]*np.fft.fft(np.eye(ns),axis=0),
        axis=0).real
    angular=-(derivative@derivative)
    base=10.+10.*np.abs(np.arange(nx)-1)
    beta=1.5
    q=np.zeros((2,nx,ns))
    fixed=np.array([
        np.ravel_multi_index((component,x,s),q.shape)
        for component in range(2)
        for x in (0,nx-1) for s in range(ns)],dtype=int)
    constraints=sparse.csr_matrix((
        np.ones(len(fixed)),
        (np.arange(len(fixed)),fixed)),shape=(len(fixed),q.size))
    projector=SimpleNamespace(
        c=constraints,count=len(fixed),size=q.size,
        target=np.zeros(len(fixed)),lift=np.zeros(q.size))
    functional=SimpleNamespace(
        geometry_model="LINEAR",material=SimpleNamespace(ell_m=0.),
        R=1.,theory="SYNTHETIC_ANGULAR_CROSS_COUPLED",
        ops=SimpleNamespace(
            Ds=derivative,ws=np.ones(ns),
            wx=np.array([17/48,*([1.]*7),17/48])))
    prepared=SimpleNamespace(
        method="FDM",fdm_backend="MATRIX_FREE_GMRES",
        mechanical=q,electrical=None,
        problem=SimpleNamespace(functional=functional))

    def action(vector,stiffness):
        state=np.asarray(vector,float).reshape(q.shape).copy()
        state[:,[0,-1],:]=0.
        if stiffness:
            output=np.empty_like(state)
            output[0]=(
                base[:,None]*state[0]+state[0]@angular.T+
                beta*(state[1]@derivative.T))
            output[1]=(
                base[:,None]*state[1]+state[1]@angular.T-
                beta*(state[0]@derivative.T))
        else:
            output=state
        output[:,[0,-1],:]=0.
        return output.ravel()

    def op(stiffness):
        return LinearOperator((q.size,q.size),dtype=float,
            matvec=lambda vec:action(vec,stiffness),
            matmat=lambda mat:np.column_stack([
                action(mat[:,j],stiffness) for j in range(mat.shape[1])]))

    assert can_use_fourier_modal_fdm(prepared,projector)
    result=solve_modal_fourier_fdm(
        prepared,1.,4,projector,op(True),op(False),
        np.ones(projector.size),1.,1.)
    # At axial x=1 the lowest K/M eigenvalue is
    # base(1)+n**2-beta*n = 10+1-1.5 = 9.5 for n=1.
    # The next (n=0) pair has eigenvalue 10.
    reference=np.sqrt([9.5,9.5,10.,10.])/(2.*np.pi)
    np.testing.assert_allclose(
        result["frequencies_hz"],reference,rtol=1.e-9,atol=1.e-11)
    assert result["circumferential_wavenumbers"]==[1,1,0,0]
    fields=result["mechanical_modes"].reshape(4,-1)
    np.testing.assert_allclose(
        fields@fields.T,np.eye(4),rtol=1.e-9,atol=1.e-9)
    assert result["global_spectrum_completeness_verified"] is True
