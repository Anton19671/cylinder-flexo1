"""Deferred O(nx+ns) SBP-CSR physics tests for the coupled shell path.

Run with the integrated solver test suite, not during mathematics-first
assembly. Compare exact *physical* operators with the dense reference.
"""
import unittest
import numpy as np
from cylinder_functional_assembly import (
    CylinderFunctional, surface_sbp21, surface_sbp21_csr,
)
from cylinder_sbp_derivatives import require_green_identity
from cylinder_surface_linear_algebra import is_sparse_matrix
from flexo_gradient_g3d import G3DMaterial
from cylinder_boundary_spatial import BoundarySpatialProblem


class SparseSurfaceOperatorTests(unittest.TestCase):
    def setUp(self):
        self.dense=surface_sbp21(6,7,1.2,2.)
        self.csr=surface_sbp21_csr(6,7,1.2,2.)
        self.rng=np.random.default_rng(611)
        self.mat=G3DMaterial(100.,.25,.08,.001,-.002,1.5)

    def test_sbp_discrete_green_and_odd_grid_safety(self):
        self.assertTrue(is_sparse_matrix(self.csr.Dx))
        self.assertTrue(is_sparse_matrix(self.csr.Ds))
        self.assertLess(require_green_identity(
            self.csr.Dx,self.csr.Ds,self.csr.wx,self.csr.ws
        )["relative_max"],1.e-12)
        self.assertLess(self.csr.Dx.nnz,4*len(self.csr.wx))
        self.assertLess(self.csr.Ds.nnz,4*len(self.csr.ws))
        with self.assertRaisesRegex(ValueError,"odd ns"):
            surface_sbp21_csr(6,8,1.2,2.)

    def test_reference_derivatives_and_adjoint_all_orders(self):
        from cylinder_surface_linear_algebra import (
            apply_biaxial,apply_weighted_adjoint,operator_power)
        field=self.rng.normal(size=(3,*self.csr.shape))
        coeff=self.rng.normal(size=field.shape)
        for ax,az in ((0,0),(1,0),(0,1),(2,0),(1,1),
                      (0,2),(3,0),(2,1),(1,2),(0,3)):
            dx=operator_power(self.dense.Dx,ax)
            ds=operator_power(self.dense.Ds,az)
            sx=operator_power(self.csr.Dx,ax)
            ss=operator_power(self.csr.Ds,az)
            np.testing.assert_allclose(
                apply_biaxial(field,sx,ss),
                apply_biaxial(field,dx,ds),rtol=1.e-11,atol=1.e-10)
            np.testing.assert_allclose(
                apply_weighted_adjoint(coeff,self.csr.area_weights,sx,ss),
                apply_weighted_adjoint(coeff,self.dense.area_weights,dx,ds),
                rtol=1.e-11,atol=1.e-10)

    def test_green_lagrange_flexo_and_viscous_newmark_all_models(self):
        from cylinder_g3d_natural_edges import axial_natural_end_tractions
        for theory in ("KL","FSDT","REDDY","CUF_TE1",
                       "CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                ngauss=(2*int(theory[-1])+1
                        if theory.startswith("CUF_TE") else 7)
                f1=CylinderFunctional(theory,self.mat,radius_m=2.,
                    thickness_m=.2,surface_operators=self.dense,n_gauss=ngauss)
                f2=CylinderFunctional(theory,self.mat,radius_m=2.,
                    thickness_m=.2,surface_operators=self.csr,n_gauss=ngauss)
                q=self.rng.normal(0,1.e-5,(f1.n_mechanical,*self.csr.shape))
                phi=self.rng.normal(0,1.e-4,(2,*self.csr.shape))
                v=self.rng.normal(0,1.e-5,q.shape)
                a=self.rng.normal(0,1.e-4,q.shape)
                # Shared Green-Lagrange/Flexo enthalpy and exact adjoint.
                left=f1.evaluate(q,phi,energy_breakdown=False)
                right=f2.evaluate(q,phi,energy_breakdown=False)
                self.assertAlmostEqual(
                    left.total_enthalpy_j,right.total_enthalpy_j,places=9)
                np.testing.assert_allclose(
                    left.mechanical_residual,right.mechanical_residual,
                    rtol=2.e-7,atol=1.e-8)
                np.testing.assert_allclose(
                    left.electrical_residual,right.electrical_residual,
                    rtol=2.e-7,atol=1.e-8)
                for f in (f1,f2):
                    # Dynamic forces use identical physical geometry.
                    force=f.mass_action(a,density_kg_m3=1300.,
                        mechanical=q,velocity=v)
                    self.assertEqual(force.shape,q.shape)
                    kv=f.kelvin_voigt_action(q,v,eta_s=.002)
                    self.assertGreaterEqual(kv.power_w,0.)
                np.testing.assert_allclose(
                    f1.mass_action(a,density_kg_m3=1300.,mechanical=q,velocity=v),
                    f2.mass_action(a,density_kg_m3=1300.,mechanical=q,velocity=v),
                    rtol=3.e-7,atol=1.e-7)
                np.testing.assert_allclose(
                    f1.kelvin_voigt_action(q,v,eta_s=.002).mechanical_force,
                    f2.kelvin_voigt_action(q,v,eta_s=.002).mechanical_force,
                    rtol=3.e-7,atol=1.e-8)
                bc={"mechanical":{"left":{"support":"CLAMPED"},
                                  "right":{"support":"FREE"}}}
                e1=BoundarySpatialProblem(f1,bc_spec=bc,calculation_kind="DYNAMIC")
                e2=BoundarySpatialProblem(f2,bc_spec=bc,calculation_kind="DYNAMIC")
                kwargs=dict(time_s=.01,velocity=v,acceleration=a,
                    density_kg_m3=1300.,eta_s=.002)
                _,r1,_=e1.evaluate_effective_residual(q,phi,**kwargs)
                _,r2,_=e2.evaluate_effective_residual(q,phi,**kwargs)
                np.testing.assert_allclose(r1,r2,rtol=3.e-7,atol=1.e-7)

    def test_sparse_green_split_matches_dense_reference(self):
        from cylinder_functional_assembly import MECHANICAL_DERIVATIVES
        from cylinder_sbp_green_reduction import split_coupled_sbp_residual
        q=self.rng.normal(size=(10,3,*self.csr.shape))
        e=self.rng.normal(size=(3,2,*self.csr.shape))
        a=split_coupled_sbp_residual(
            self.dense,q,MECHANICAL_DERIVATIVES,e)
        b=split_coupled_sbp_residual(
            self.csr,q,MECHANICAL_DERIVATIVES,e)
        np.testing.assert_allclose(
            a.mechanical.reconstructed,b.mechanical.reconstructed,
            atol=1.e-8,rtol=1.e-10)
        np.testing.assert_allclose(
            a.electrical.reconstructed,b.electrical.reconstructed,
            atol=1.e-8,rtol=1.e-10)


if __name__=="__main__":
    unittest.main()
