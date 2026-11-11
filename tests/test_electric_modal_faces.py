"""Manufactured tests for every electrical thickness mode, even ns, and P-flux."""
import unittest
import numpy as np

from cylinder_electric_modal_faces import (
    FacePolarizationSamples, modal_electric_face_weak)
from cylinder_staggered_surface import StaggeredCylinderSurface


class ModalFaceFluxTests(unittest.TestCase):
    PARAMS=dict(length_m=1.2,radius_m=2.,thickness_m=.2,
                kappa_f_per_m=1.5)

    def run_case(self, p, **extras):
        return modal_electric_face_weak(p,**self.PARAMS,**extras)

    def test_te0_matches_independent_scalar_face_flux(self):
        from cylinder_electric_face_flux import dielectric_face_flux_fdm
        rng=np.random.default_rng(81)
        p=rng.normal(size=(1,5,8))
        result=self.run_case(p,n_gauss=24)
        expected=dielectric_face_flux_fdm(p[0],**self.PARAMS)
        np.testing.assert_allclose(result.electrical_residual_c[0],
                                   expected.residual_c,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(result.axial_mode_flux_c_per_m[0],
                                   expected.axial_face_flux_c_per_m,
                                   rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(result.circumferential_mode_flux_c_per_m[0],
                                   expected.circumferential_face_flux_c_per_m,
                                   rtol=1e-13,atol=1e-13)
        self.assertAlmostEqual(result.electric_potential_enthalpy_j,
                               expected.potential_energy_j,places=12)

    def test_energy_gateaux_all_five_modes_no_polarization(self):
        rng=np.random.default_rng(33)
        for nm in (1,2,3,4,5):
            with self.subTest(nm=nm):
                p=.08*rng.normal(size=(nm,5,6))
                v=.3*rng.normal(size=p.shape)
                result=self.run_case(p,n_gauss=12)
                eps=2e-6
                plus=self.run_case(p+eps*v,n_gauss=12).electric_potential_enthalpy_j
                minus=self.run_case(p-eps*v,n_gauss=12).electric_potential_enthalpy_j
                numerical=(plus-minus)/(2*eps)
                analytic=float(np.sum(result.electrical_residual_c*v))
                self.assertAlmostEqual(numerical,analytic,delta=1e-8*max(1,abs(analytic)))

    def test_pol_variational_work_includes_all_three_directions(self):
        rng=np.random.default_rng(252)
        n_gauss=11
        p=.1*rng.normal(size=(5,5,8))
        dp=rng.normal(size=p.shape)
        px=.02*rng.normal(size=(n_gauss,4,8))
        ps=.02*rng.normal(size=(n_gauss,5,8))
        pz=.02*rng.normal(size=(n_gauss,5,8))
        pol=FacePolarizationSamples(px,ps,pz)
        def solve(x):return self.run_case(x,n_gauss=n_gauss,polarization=pol)
        result=solve(p)
        eps=1e-6
        fd=(solve(p+eps*dp).electric_potential_enthalpy_j-
            solve(p-eps*dp).electric_potential_enthalpy_j)/(2*eps)
        exact=float(np.sum(result.electrical_residual_c*dp))
        self.assertAlmostEqual(fd,exact,delta=5e-8*max(1.,abs(exact)))
        self.assertIsNone(result.inner_outward_induction_c_per_m2)
        self.assertIsNone(result.outer_outward_induction_c_per_m2)

    def test_constant_p0_is_global_gauge_only(self):
        rng=np.random.default_rng(6)
        p=rng.normal(size=(5,5,6))*.4
        old=self.run_case(p)
        shifted=p.copy();shifted[0]+=33.0
        new=self.run_case(shifted)
        np.testing.assert_allclose(old.electrical_residual_c,new.electrical_residual_c,
                                   atol=5e-13,rtol=0)
        self.assertAlmostEqual(old.electric_potential_enthalpy_j,
                               new.electric_potential_enthalpy_j,places=11)
        self.assertAlmostEqual(old.gauge_mode_net_c,0.,places=10)
        self.assertAlmostEqual(new.gauge_mode_net_c,0.,places=10)

    def test_linear_thickness_mode_exact_curved_geometry(self):
        V=.3;L,R,h,k=1.2,2.,.2,1.5
        p=np.zeros((2,5,6));p[1]=V
        result=self.run_case(p,n_gauss=6)
        area=2*np.pi*R*L
        expected=-2.*k*V*V/h*area
        self.assertAlmostEqual(result.electric_potential_enthalpy_j,expected,places=11)
        grid=StaggeredCylinderSurface(5,6,L,R)
        np.testing.assert_allclose(result.electrical_residual_c[0],0.,atol=2e-14)
        np.testing.assert_allclose(result.electrical_residual_c[1],
                                   (-4*k*V/h)*grid.wx[:,None]*grid.ds*np.ones((1,6)),
                                   rtol=0,atol=8e-14)
        np.testing.assert_allclose(result.inner_outward_induction_c_per_m2,
                                   np.ones((5,6))*2*k*V/h)
        np.testing.assert_allclose(result.outer_outward_induction_c_per_m2,
                                   -np.ones((5,6))*2*k*V/h)

    def test_even_periodic_nyquist_not_killed(self):
        for ns in (4,6,8,12):
            with self.subTest(ns=ns):
                p=np.zeros((5,5,ns))
                p[0]=(-1.)**np.arange(ns)
                out=self.run_case(p)
                self.assertGreater(np.max(np.abs(out.electrical_residual_c[0])),.01)
                self.assertAlmostEqual(out.gauge_mode_net_c,0.,places=10)

    def test_transverse_polarization_produces_higher_mode_weak_force(self):
        n_gauss=8
        p=np.zeros((2,5,6))
        Z=.2
        pol=FacePolarizationSamples(
            np.zeros((n_gauss,4,6)),np.zeros((n_gauss,5,6)),
            Z*np.ones((n_gauss,5,6)),
            Z*np.ones((5,6)),Z*np.ones((5,6)))
        out=self.run_case(p,n_gauss=n_gauss,polarization=pol)
        grid=StaggeredCylinderSurface(5,6,1.2,2.)
        np.testing.assert_allclose(out.electrical_residual_c[0],0.,atol=1e-13)
        np.testing.assert_allclose(out.electrical_residual_c[1],
                                   2*Z*grid.wx[:,None]*grid.ds*np.ones((1,6)),atol=1e-13)
        np.testing.assert_allclose(out.inner_outward_induction_c_per_m2,-Z,atol=1e-14)
        np.testing.assert_allclose(out.outer_outward_induction_c_per_m2,Z,atol=1e-14)
        self.assertEqual(out.electric_potential_enthalpy_j,0.)

    def test_incomplete_face_polarization_rejected(self):
        z=6
        pol=FacePolarizationSamples(np.zeros((z,4,6)),np.zeros((z,5,6)),
                                    np.zeros((z,5,6)),np.zeros((5,6)),None)
        with self.assertRaisesRegex(ValueError,"both inner and outer"):
            self.run_case(np.zeros((2,5,6)),n_gauss=z,polarization=pol)

    def test_insufficient_quadrature_rejected(self):
        with self.assertRaisesRegex(ValueError,"n_gauss"):
            self.run_case(np.zeros((5,4,4)),n_gauss=4)


if __name__=="__main__":unittest.main()


def test_cylindrical_collocation_recovers_constant_cartesian_vector():
    from cylinder_electric_modal_faces import collocate_cylindrical_polarization
    for ns in (4,5,6,9,12,32):
        theta=2*np.pi*np.arange(ns)/ns
        # Any uniform in-plane Cartesian polarization rotates between
        # the cylindrical local s/z components as theta changes.
        px=3.1
        cartesian_y=2.7
        cartesian_z=-1.9
        nodal=np.empty((3,5,ns,3))
        nodal[...,0]=px
        nodal[...,1]=cartesian_y*np.cos(theta)-cartesian_z*np.sin(theta)
        nodal[...,2]=cartesian_y*np.sin(theta)+cartesian_z*np.cos(theta)
        axial,circ,transverse=collocate_cylindrical_polarization(nodal)
        np.testing.assert_allclose(axial,px,rtol=0,atol=1e-14)
        face_theta=theta+np.pi/ns
        expected=(cartesian_y*np.cos(face_theta)-
                  cartesian_z*np.sin(face_theta))
        np.testing.assert_allclose(circ,np.broadcast_to(expected,circ.shape),
                                   rtol=0,atol=2e-14)
        np.testing.assert_allclose(transverse,nodal[...,2])


def test_collocation_rejects_invalid_vector_shape():
    from cylinder_electric_modal_faces import collocate_cylindrical_polarization
    with np.testing.assert_raises_regex(ValueError,'polarization'):
        collocate_cylindrical_polarization(np.zeros((2,5,6,2)))
