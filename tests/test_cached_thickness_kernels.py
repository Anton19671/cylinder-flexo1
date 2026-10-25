"""Regression specifications for cached nonlinear G3D/flexo assembly.

These checks are staged for the integrated scientific verification phase.
They must compare *identical* constitutive weak forms, not just runtime.
"""
import unittest
import numpy as np

from cuf_g3d_weak import (
    CUFSurfaceJets, MECHANICAL_VARIATIONS, _unit_variation,
    _kinematic_matrix_jets, _cuf_virtual_kinematic_library,
    _dimensionless_gauss_rule,gauss_thickness)
from cylinder_director_g3d import (
    FIELDS,ELECTRICAL_FIELDS,DirectorSurfaceJets,
    _unit,_F_jets,_displacement,
    _director_virtual_f_jets,_director_virtual_velocity)
from cylinder_functional_assembly import (
    CylinderFunctional,surface_sbp21_csr)
from flexo_gradient_g3d import G3DMaterial


class CacheEquivalence(unittest.TestCase):
    def test_cuf_kinematic_library_matches_fresh_unit_variations(self):
        for te in (1,2,3,4):
            for z in (-.073,0.,.051):
                q=np.zeros((3,te+1))
                e=np.zeros(2)
                state=CUFSurfaceJets(*(q.copy() for _ in MECHANICAL_VARIATIONS),
                                     *(e.copy() for _ in ("psi","psix","psis")))
                cached=_cuf_virtual_kinematic_library(te,z,2.,.2)
                self.assertFalse(cached.flags.writeable)
                for a in range(3):
                    for i in range(te+1):
                        for k,name in enumerate(MECHANICAL_VARIATIONS):
                            fresh=_kinematic_matrix_jets(
                                _unit_variation(state,a,i,name),z,2.,.2,
                                variation=True)
                            np.testing.assert_allclose(
                                cached[a,i,k],np.stack(fresh),rtol=0.,atol=1e-14)

    def test_director_libraries_match_fresh_unit_variations(self):
        for theory in ("FSDT","REDDY"):
            q=np.zeros(5);e=np.zeros(2)
            state=DirectorSurfaceJets(*(q.copy() for _ in FIELDS),
                                      *(e.copy() for _ in ELECTRICAL_FIELDS))
            for z in (-.073,.051):
                f=_director_virtual_f_jets(theory,z,2.,.2)
                a=_director_virtual_velocity(theory,z,2.,.2)
                self.assertFalse(f.flags.writeable)
                for k,name in enumerate(FIELDS):
                    for comp in range(5):
                        unit=_unit(state,name,comp)
                        np.testing.assert_allclose(
                            f[k,comp],np.stack(_F_jets(
                                unit,theory,z,2.,.2,virtual=True)),
                            rtol=0.,atol=1e-13)
                        if k<3:
                            np.testing.assert_allclose(
                                a[k,comp],_displacement(unit,theory,z,2.,.2),
                                rtol=0.,atol=1e-13)

    def test_gauss_rule_immutable_and_scaled(self):
        x,w=_dimensionless_gauss_rule(9)
        self.assertFalse(x.flags.writeable)
        self.assertFalse(w.flags.writeable)
        z,weight=gauss_thickness(.2,9)
        np.testing.assert_allclose(z,x*.1)
        np.testing.assert_allclose(weight,w*.1)
        self.assertAlmostEqual(float(np.sum(weight)),.2,places=14)

    def test_newton_sparse_capacity_is_configurable(self):
        from cylinder_matrixfree_fdm import MatrixFreeNewtonSettings
        cfg=MatrixFreeNewtonSettings(max_surface_nodes=12000,
                                     max_total_dofs=130000)
        self.assertEqual(cfg.max_surface_nodes,12000)
        with self.assertRaises(ValueError):
            MatrixFreeNewtonSettings(max_surface_nodes=0)


if __name__=="__main__":
    unittest.main()
