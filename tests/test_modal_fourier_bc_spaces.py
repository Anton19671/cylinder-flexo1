"""Regression of Fourier modal boundary subspaces and safe fallback.

The synthetic operator uses exact known angular sectors, not a shell benchmark.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from types import SimpleNamespace
from cylinder_matrixfree_fdm import SparseTraceProjector
from cylinder_modal_fourier_fdm import (
    _axial_constraint_basis, can_use_fourier_modal_fdm, solve_modal_fourier_fdm)


def _fixture(kind):
    nx = ns = 9
    d = np.fft.ifft(1j*np.fft.fftfreq(ns,d=1/ns)[:,None]*
                    np.fft.fft(np.eye(ns),axis=0),axis=0).real
    angular = -d@d
    q = np.zeros((1,nx,ns))
    f = SimpleNamespace(geometry_model="LINEAR",
        material=SimpleNamespace(ell_m=0.),R=1.,theory="SYNTHETIC",
        ops=SimpleNamespace(Ds=d,ws=np.ones(ns),
                            wx=np.array([17/48,1,1,1,1,1,1,1,17/48])))
    prepared=SimpleNamespace(method="FDM",fdm_backend="MATRIX_FREE_GMRES",
        electrical=None,mechanical=q,problem=SimpleNamespace(functional=f))
    rows=[]
    for j in range(ns):
        if kind in ("SLOPE","NONUNIFORM"):
            if kind=="NONUNIFORM" and j!=0:
                continue
            row=np.zeros(q.size)
            for i,v in enumerate((1.,-3.,3.,-1.)):
                row[i*ns+j]=v
            rows.append(row)
        if kind=="SLOPE":
            for i in (0,nx-1):
                row=np.zeros(q.size);row[i*ns+j]=1.;rows.append(row)
    pr=SparseTraceProjector(sparse.csr_matrix(
        np.array(rows).reshape(-1,q.size)),np.zeros(len(rows)),max_rows=500)
    a=10.+np.arange(nx)
    def action(x,stiff):
        field=pr.project_null(np.asarray(x)).reshape(q.shape)
        out=np.zeros_like(field)
        out[0]=a[:,None]*field[0]+field[0]@angular if stiff else field[0]
        return pr.project_null(out.ravel())
    def op(stiff):
        return LinearOperator((q.size,q.size),dtype=float,
            matvec=lambda x:action(x,stiff),
            matmat=lambda x:np.column_stack([
                action(x[:,j],stiff) for j in range(x.shape[1])]))
    return prepared,pr,op(True),op(False)


@pytest.mark.parametrize("kind,expected", [("SLOPE",6),("FREE",9)])
def test_uniform_multipoint_and_free_end_modal_eigenpairs(kind,expected):
    prepared,pr,k,m=_fixture(kind)
    basis=_axial_constraint_basis(prepared,pr)
    assert basis.shape==(9,expected)
    assert can_use_fourier_modal_fdm(prepared,pr)
    result=solve_modal_fourier_fdm(prepared,1.,6,pr,k,m,np.ones(pr.size),1.,1.)
    freq=result["frequencies_hz"]
    assert np.isfinite(freq).all() and np.all(np.diff(freq)>=-1e-9)
    assert result["global_spectrum_completeness_verified"]
    for mode in result["mechanical_modes"]:
        np.testing.assert_allclose(pr.c@mode.ravel(),0.,atol=1e-10)


def test_asymmetric_circumferential_constraint_refuses_fourier_route():
    prepared,pr,_,_=_fixture("NONUNIFORM")
    assert _axial_constraint_basis(prepared,pr) is None
    assert not can_use_fourier_modal_fdm(prepared,pr)
