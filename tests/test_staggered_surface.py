"""Discrete Green pairing and even-periodic-mode checks for §40 faces."""
import unittest
import numpy as np

from cylinder_staggered_surface import StaggeredCylinderSurface


class StaggeredSurfaceTests(unittest.TestCase):
    def test_face_action_is_exact_weighted_gradient_adjoint(self):
        grid=StaggeredCylinderSurface(5,6,1.2,2.)
        rng=np.random.default_rng(26)
        variation=rng.normal(size=(3,grid.nx,grid.ns))
        fx=rng.normal(size=(3,grid.nx-1,grid.ns))
        fs=rng.normal(size=(3,grid.nx,grid.ns))
        gx,gs=grid.gradients(variation)
        virtual_work=(grid.dx*grid.ds*np.sum(fx*gx)
                      +grid.ds*np.sum(grid.wx[None,:,None]*fs*gs))
        nodal_work=np.sum(grid.weak_face_action(fx,fs)*variation)
        self.assertAlmostEqual(virtual_work,nodal_work,places=12)
        np.testing.assert_allclose(np.sum(grid.weak_face_action(fx,fs)),0.,atol=1.e-13)

    def test_even_periodic_checkerboard_has_nonzero_gradient(self):
        grid=StaggeredCylinderSurface(4,8,1.,2.)
        field=np.broadcast_to((-1.)**np.arange(grid.ns),(grid.nx,grid.ns))
        gx,gs=grid.gradients(field)
        np.testing.assert_array_equal(gx,np.zeros_like(gx))
        np.testing.assert_allclose(gs,-2.*field/grid.ds)
        zero_x,zero_s=grid.gradients(np.ones_like(field))
        np.testing.assert_array_equal(zero_x,np.zeros_like(zero_x))
        np.testing.assert_array_equal(zero_s,np.zeros_like(zero_s))


if __name__=="__main__":unittest.main()
