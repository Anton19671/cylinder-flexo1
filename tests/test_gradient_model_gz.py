"""Constitutive and seven-model Gz/G3D selection regressions.

Gz is a REAL reduction of the enthalpy: only the six covariant
through-thickness strain-gradient components survive. The spatial
discretization and time integrator remain those of each shell theory.
"""
import unittest
import numpy as np

from flexo_gradient_g3d import (G3DMaterial, active_strain_gradients,
    enthalpy_parts, enthalpy_derivatives, gradient_invariants)
from cylinder_functional_assembly import CylinderFunctional, surface_sbp21
from cylinder_boundary_spatial import BoundarySpatialProblem
from flexo_scale_policy import micro_material_from_case


class GzMaterialTests(unittest.TestCase):
    def material(self,mode):
        return G3DMaterial(170.,.28,.2,1.5,-.7,3.,gradient_model=mode)

    def test_6_vs_18_components_and_default(self):
        g=np.arange(18,dtype=float).reshape(6,3)*.023
        self.assertEqual(self.material("Gz").gradient_model,"GZ")
        self.assertEqual(self.material("Gz").active_gradient_components,6)
        self.assertEqual(self.material("G3D").active_gradient_components,18)
        self.assertEqual(G3DMaterial(170.,.28,.2,1.5,-.7,3.).gradient_model,"G3D")
        np.testing.assert_array_equal(active_strain_gradients(g,"Gz")[:,:2],0.)
        np.testing.assert_array_equal(active_strain_gradients(g,"Gz")[:,2],g[:,2])
        np.testing.assert_array_equal(active_strain_gradients(g,"G3D"),g)
        with self.assertRaisesRegex(ValueError,"gradient_model"):
            self.material("G2D")

    def test_thickness_gradient_27_enthalpy_derivatives(self):
        rng=np.random.default_rng(793)
        E=rng.normal(0.,.07,6)
        g=rng.normal(0.,.12,(6,3))
        ef=rng.normal(0.,.15,3)
        h=1.e-6
        for mode in ("G3D","Gz"):
            material=self.material(mode)
            stress,hyper,D=enthalpy_derivatives(E,g,ef,material)
            energy=lambda e,grad,field:enthalpy_parts(
                e,grad,field,material)["total"]
            if mode=="Gz":
                np.testing.assert_array_equal(hyper[:,:2],0.)
            for j in range(6):
                dx=np.zeros(6);dx[j]=h
                np.testing.assert_allclose(
                    (energy(E+dx,g,ef)-energy(E-dx,g,ef))/(2*h),
                    stress[j],rtol=3.e-7,atol=1.e-8)
            for j in range(6):
                for k in range(3):
                    dx=np.zeros((6,3));dx[j,k]=h
                    np.testing.assert_allclose(
                        (energy(E,g+dx,ef)-energy(E,g-dx,ef))/(2*h),
                        hyper[j,k],rtol=3.e-7,atol=1.e-8)
            for k in range(3):
                dx=np.zeros(3);dx[k]=h
                np.testing.assert_allclose(
                    (energy(E,g,ef+dx)-energy(E,g,ef-dx))/(2*h),
                    -D[k],rtol=3.e-7,atol=1.e-8)

    def test_z_only_gradient_energy_and_flexo_are_not_disabled(self):
        E=np.zeros(6)
        g=np.zeros((6,3))
        g[:,2]=np.array([.12,-.05,.03,.04,.11,-.09])
        ef=np.array([.3,-.2,.5])
        gz=self.material("Gz")
        whole=self.material("G3D")
        Wz=enthalpy_parts(E,g,ef,gz)
        W3d=enthalpy_parts(E,g,ef,whole)
        self.assertGreater(Wz["gradient"],0.)
        self.assertNotEqual(Wz["flexoelectric"],0.)
        np.testing.assert_allclose(Wz["total"],W3d["total"])
        _,mz,Dz=enthalpy_derivatives(E,g,ef,gz)
        _,m3,D3=enthalpy_derivatives(E,g,ef,whole)
        np.testing.assert_allclose(mz[:,2],m3[:,2])
        np.testing.assert_allclose(Dz,D3)
        self.assertTrue(np.any(mz[:,2]!=0))

    def test_in_plane_covariant_curvature_terms_disappear_from_Gz_energy(self):
        E=np.array([.03,-.07,.05,.06,-.02,.12])
        g=np.zeros((6,3))
        g[:,1]=np.array([0.,.05,-.05,.13,-.08,.04])
        g[:,0]=.03
        ef=np.array([.2,-.13,.09])
        gz=self.material("Gz")
        parts=enthalpy_parts(E,g,ef,gz)
        self.assertEqual(parts["gradient"],0.)
        self.assertEqual(parts["flexoelectric"],0.)
        _,M,D=enthalpy_derivatives(E,g,ef,gz)
        # Zero flexoelectric energy at zero z-gradient does not imply zero
        # first variation: the flexoelectric enthalpy is linear in g[:, 2].
        np.testing.assert_array_equal(M[:,:2],0.)
        self.assertGreater(np.max(np.abs(M[:,2])),0.)
        np.testing.assert_allclose(D,gz.kappa_f_per_m*ef)
        delta=1.e-6
        for j in range(6):
            gp=g.copy();gm=g.copy();gp[j,2]+=delta;gm[j,2]-=delta
            slope=(enthalpy_parts(E,gp,ef,gz)['total']-
                   enthalpy_parts(E,gm,ef,gz)['total'])/(2*delta)
            np.testing.assert_allclose(slope,M[j,2],rtol=2.e-7,atol=1.e-8)

    def test_micro_CASE_accepts_both_and_rejects_unknown(self):
        template=dict(scale_model="MICRO",geometry_model="GREEN_LAGRANGE",
            units={"system":"SI_N_M_PA"},material={"E0":170.,"nu0":.28},
            micro={"gradient_model":"Gz","enable_flexoelectric":True,
                   "ell_m":.2,"mu1_c_per_m":1.5,"mu2_c_per_m":-.7,
                   "kappa_f_per_m":3.,"rho_kg_m3":1300.,"eta_s":0.,
                   "potential_order":1})
        self.assertEqual(micro_material_from_case(template).gradient_model,"GZ")
        template["micro"]["gradient_model"]="G3D"
        self.assertEqual(micro_material_from_case(template).gradient_model,"G3D")
        template["micro"]["gradient_model"]="Gx"
        with self.assertRaisesRegex(ValueError,"gradient_model"):
            micro_material_from_case(template)


class SevenTheoryGradientSwitchTests(unittest.TestCase):
    def test_all_theories_static_dynamic_same_selected_constitutive_law(self):
        # All terms other than gradient elasticity / flexoelectricity
        # must be the same for Gz/G3D on identical q,phi,qdot,qddot.
        rng=np.random.default_rng(20261025)
        surface=surface_sbp21(4,5,1.2,2.)
        for theory in ("KL","FSDT","REDDY","CUF_TE1",
                       "CUF_TE2","CUF_TE3","CUF_TE4"):
            n_gauss=(2*int(theory[-1])+1
                     if theory.startswith("CUF_TE") else 7)
            f3=CylinderFunctional(theory,
                G3DMaterial(170.,.28,.2,.015,-.007,3.,"G3D"),
                radius_m=2.,thickness_m=.2,surface_operators=surface,
                n_gauss=n_gauss)
            fz=CylinderFunctional(theory,
                G3DMaterial(170.,.28,.2,.015,-.007,3.,"Gz"),
                radius_m=2.,thickness_m=.2,surface_operators=surface,
                n_gauss=n_gauss)
            q=rng.normal(0,1.e-4,(f3.n_mechanical,*surface.shape))
            phi=rng.normal(0,.001,(2,*surface.shape))
            velocity=rng.normal(0,1.e-3,q.shape)
            acceleration=rng.normal(0,1.e-3,q.shape)
            with self.subTest(theory=theory):
                pure3=f3.evaluate(q,phi,energy_breakdown=True,
                                  check_charge_balance=False)
                purez=fz.evaluate(q,phi,energy_breakdown=True,
                                  check_charge_balance=False)
                np.testing.assert_allclose(
                    pure3.energy_parts_j["classical"],
                    purez.energy_parts_j["classical"])
                np.testing.assert_allclose(
                    pure3.energy_parts_j["dielectric"],
                    purez.energy_parts_j["dielectric"])
                dynamic=dict(time_s=.01,velocity=velocity,
                             acceleration=acceleration,
                             density_kg_m3=1300.,eta_s=.01,
                             mass_damping_rate_per_s=.03)
                p3=BoundarySpatialProblem(f3,calculation_kind="DYNAMIC")
                pz=BoundarySpatialProblem(fz,calculation_kind="DYNAMIC")
                _,dr3,_=p3.evaluate_effective_residual(q,phi,**dynamic)
                _,drz,_=pz.evaluate_effective_residual(q,phi,**dynamic)
                _,sr3,_=p3.evaluate_effective_residual(q,phi,time_s=.01,
                    acceleration=np.zeros_like(q),density_kg_m3=1300.)
                _,srz,_=pz.evaluate_effective_residual(q,phi,time_s=.01,
                    acceleration=np.zeros_like(q),density_kg_m3=1300.)
                np.testing.assert_allclose(
                    dr3-drz,sr3-srz,rtol=1.e-7,atol=2.e-8)
                self.assertEqual(dr3.shape,drz.shape)


if __name__=="__main__":
    unittest.main()
