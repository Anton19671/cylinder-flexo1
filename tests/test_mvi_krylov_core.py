"""Rank-fixed MVI operator checks; no full-grid Newton corrections."""
import numpy as np
from scipy import sparse
from cylinder_mvi_direction_space import SeparatedMVIDirections, SparseHalfstepTraceProjector
from cylinder_mvi_krylov_core import separated_krylov_sweeps

def _pair(nx=5,ns=7):
    x=np.zeros((1,1,nx));y=np.ones((1,1,ns))/np.sqrt(ns)
    return x,y

def test_x_and_y_adjoint():
    rng=np.random.default_rng(7)
    x,y=_pair()
    for axis in ("X","Y"):
        op=SeparatedMVIDirections(x,y,axis)
        a=rng.normal(size=op.n_parameters);b=rng.normal(size=op.n)
        np.testing.assert_allclose(op.push(a),op.sparse_matrix()@a,atol=1e-13)
        np.testing.assert_allclose(op.pull(b),op.sparse_matrix().T@b,atol=1e-13)

def test_trace_projection_is_idempotent():
    x,y=_pair()
    op=SeparatedMVIDirections(x,y,"X")
    ns=y.shape[-1]
    C=sparse.csr_matrix((np.ones(ns),(np.arange(ns),np.arange(ns))),
                        shape=(ns,op.n))
    projector=SparseHalfstepTraceProjector(C,op)
    v=np.arange(op.n_parameters,dtype=float)+1.
    u=projector.project(v)
    np.testing.assert_allclose(C@op.push(u),0.,atol=1e-12)
    np.testing.assert_allclose(projector.project(u),u,atol=1e-12)

def test_nonaxisymmetric_rank_one_without_svd():
    x,y=_pair()
    nx,ns=x.shape[-1],y.shape[-1]
    target=np.outer(np.arange(nx)/(nx-1),
                    np.sin(2*np.pi*np.arange(ns)/ns))[None]
    def evaluator(v):
        g=v-target.ravel()
        return .5*float(g@g),g
    result=separated_krylov_sweeps(
        x,y,np.zeros_like(target),sparse.csr_matrix((0,target.size)),
        evaluator,absolute_tolerance=1e-11,relative_tolerance=1e-8,
        max_halfstep_iterations=5,gmres_rtol=1e-5)
    assert result.converged,result.status
    np.testing.assert_allclose(result.state_scaled,target.ravel(),atol=1e-8)
