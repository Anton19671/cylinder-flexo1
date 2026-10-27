"""Optional *local material-kernel* audits: all seven theories x G3D/Gz.

Enable explicitly with CYLINDER_LOCAL_SYMBOL_AUDIT=1. This audit does NOT
solve any static/dynamic shell case, perform MVI, or march Newmark in time.
The default test suite checks synthetic symbol algebra without evaluating
slow full local shell kernels. The explicit audit can reveal incorrect
entirely local direct/inverse flexoelectric first variations.
"""
import os
import numpy as np
import pytest
from cylinder_coupled_principal_symbol import THEORIES, local_shell_symbol


@pytest.mark.skipif(os.environ.get("CYLINDER_LOCAL_SYMBOL_AUDIT")!="1",
                    reason="Enable local 7-theory constitutive symbol audit explicitly")
@pytest.mark.parametrize("theory",THEORIES)
@pytest.mark.parametrize("gradient",("G3D","GZ"))
def test_actual_shell_thickness_kernels_are_reciprocal(theory,gradient):
    from flexo_gradient_g3d import G3DMaterial
    # A small, geometrically admissible undeformed reference state.
    material=G3DMaterial(young_pa=100.,poisson=.25,ell_m=.05,
                        mu1_c_per_m=.001,mu2_c_per_m=-.002,
                        kappa_f_per_m=1.5,gradient_model=gradient)
    symbol=local_shell_symbol(theory,material,radius_m=2.,thickness_m=.2,
                              potential_order=1,n_gauss=3,relative_step=1e-5,
                              reciprocity_tolerance=5e-4)
    assert symbol.layout.n_electric==2
    for direction in ((1.,0.),(0.,1.),(1.,1.)):
        matrix=symbol.evaluate(*direction)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix,matrix.conj().T,rtol=5e-4,atol=1e-8)
        report=symbol.directional_coupling_audit()
        assert report["status"]=="MIXED_COUPLED_SYMBOL_ORDER_NOT_CERTIFIED"
