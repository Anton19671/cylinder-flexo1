"""Queued integrated checks for physical radial load weak forms.

These tests are intentionally NOT automatically run during assembly stage.
They check the external work and physical-face metric for all theories.
"""
import unittest
import numpy as np
from cylinder_variational_loads import project_radial_load
from cylinder_functional_assembly import surface_trapezoid


class _Reference:
    R=2.
    h=.2

    def __init__(self,theory,nx=6,ns=12):
        self.theory=theory
        self.ops=surface_trapezoid(nx,ns,1.2,self.R)
        self.n_mechanical=(3 if theory=="KL" else
                           5 if theory in ("FSDT","REDDY") else
                           3*(int(theory[-1])+1))


class VariationalRadialLoadsTest(unittest.TestCase):
    def definition(self,**extra):
        return {"type":"UNIFORM_RADIAL_FACE",
                "q":7.,"face":"MIDDLE",
                "kinematics":"DEAD_REFERENCE",**extra}

    def test_all_theories_middle_face_virtual_work(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                f=_Reference(theory)
                res=project_radial_load(f,self.definition())
                self.assertEqual(res.generalized.shape,
                                 (f.n_mechanical,*f.ops.shape))
                radial=2 if not theory.startswith("CUF") else (
                    2*(int(theory[-1])+1))
                np.testing.assert_allclose(res.generalized[radial],7.)
                direction=np.random.default_rng(42).normal(
                    size=res.generalized.shape)
                weights=f.ops.area_weights[None,...]
                eps=1.e-6
                potential=lambda state:-float(np.sum(
                    weights*res.generalized*state))
                finite_difference=(potential(eps*direction)-
                                   potential(-eps*direction))/(2*eps)
                analytical=-float(np.sum(weights*res.generalized*direction))
                self.assertAlmostEqual(finite_difference,analytical,places=8)

    def test_cuf_inner_and_outer_face_moments(self):
        f=_Reference("CUF_TE4")
        for face,xi in (("OUTER",1.),("INNER",-1.)):
            with self.subTest(face=face):
                d=self.definition();d["face"]=face
                r=project_radial_load(f,d)
                H=1.+xi*f.h/(2*f.R)
                for k in range(5):
                    np.testing.assert_allclose(
                        r.generalized[10+k],7.*H*xi**k)

    def test_fsd_and_reddy_face_metric(self):
        for theory in ("FSDT","REDDY"):
            for face in ("OUTER","INNER"):
                with self.subTest(theory=theory,face=face):
                    f=_Reference(theory)
                    d=self.definition();d["face"]=face
                    r=project_radial_load(f,d)
                    z=f.h/2 if face=="OUTER" else -f.h/2
                    np.testing.assert_allclose(
                        r.generalized[2],7.*(1+z/f.R))

    def test_kl_offset_requires_director_virtual_work(self):
        d=self.definition();d["face"]="OUTER"
        with self.assertRaisesRegex(ValueError,"KL offset-face"):
            project_radial_load(_Reference("KL"),d)

    def test_generator_force_normalized_with_face_area(self):
        f=_Reference("CUF_TE2",nx=8,ns=24)
        spec={"type":"LINE_RADIAL_GENERATOR","face":"INNER",
              "kinematics":"DEAD_REFERENCE","p_line":9.,
              "c0":0.,"band_width":2.*np.pi*f.R/3.,
              "x1":0.,"x2":1.2}
        r=project_radial_load(f,spec)
        self.assertAlmostEqual(r.metadata["face_force_radial_n"],9.*1.2,places=10)

    def test_regularized_point_force_not_mesh_scaled(self):
        f=_Reference("CUF_TE1",nx=8,ns=24)
        spec={"type":"POINT_RADIAL_FORCE","face":"OUTER",
              "kinematics":"DEAD_REFERENCE","P":15.,
              "x0":.6,"theta0":1.1,
              "radius_x_m":.35,"radius_s_m":1.5}
        r=project_radial_load(f,spec)
        self.assertAlmostEqual(r.metadata["face_force_radial_n"],15.,places=10)

    def test_rejects_follower_and_singular_forces(self):
        f=_Reference("CUF_TE1")
        d=self.definition();d["kinematics"]="FOLLOWER"
        with self.assertRaisesRegex(ValueError,"DEAD_REFERENCE"):
            project_radial_load(f,d)
        with self.assertRaisesRegex(ValueError,"regularization"):
            project_radial_load(f,{"type":"POINT_RADIAL_FORCE","P":5.,
                "face":"MIDDLE","kinematics":"DEAD_REFERENCE",
                "x0":.5,"theta0":0.,
                "radius_x_m":1.e-6,"radius_s_m":1.e-6})


if __name__=="__main__":
    unittest.main()
