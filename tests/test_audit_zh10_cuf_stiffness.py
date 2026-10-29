"""CUF auxiliary ILU: exact polynomial derivative scaling and coupled action."""
import numpy as np
import pytest
from cylinder_krylov_block_preconditioner import _cuf_thickness_penalty, PhysicsBlockILU
from cylinder_functional_assembly import CylinderFunctional, surface_sbp21_csr
from cylinder_matrixfree_fdm import SparseTraceProjector
from flexo_gradient_g3d import G3DMaterial

@pytest.mark.parametrize('k,expected_factor',[(0,0.),(1,4.),(2,16/3.),(3,36/5.),(4,64/7.)])
def test_cuf_penalty_scales_as_E_over_h(k,expected_factor):
    e,h=200.,.1
    assert _cuf_thickness_penalty(e,h,k)==pytest.approx(expected_factor*e/h)
    assert _cuf_thickness_penalty(2*e,h,k)==pytest.approx(2*_cuf_thickness_penalty(e,h,k))
    assert _cuf_thickness_penalty(e,2*h,k)==pytest.approx(.5*_cuf_thickness_penalty(e,h,k))

def test_cuf_te4_preconditioner_acts_on_both_blocks():
    ops=surface_sbp21_csr(4,5,1.2,2.)
    material=G3DMaterial(200.,.25,.08,.001,-.002,1.5)
    f=CylinderFunctional('CUF_TE4',material,radius_m=2.,thickness_m=.1,surface_operators=ops,n_gauss=9)
    n=20
    size=(f.n_mechanical+2)*n
    c=np.zeros((1,size));c[0,0]=1.
    projector=SparseTraceProjector(c,[0.],max_rows=2)
    pre=PhysicsBlockILU(f,np.ones(size),200.,projector,drop_tolerance=1e-3,fill_factor=5.,max_surface_nodes=25)
    v=np.random.default_rng(27).normal(size=size)
    y=pre.apply(v)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(projector.c@y,projector.c@v,atol=1e-10)
