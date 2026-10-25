"""Independent stored-state Newmark audit regression."""
import unittest
import numpy as np
from cylinder_newmark_audit import audit_newmark_step


class NewmarkAuditTests(unittest.TestCase):
    def make_step(self, shape=(2,3,4)):
        rng = np.random.default_rng(42)
        q0 = rng.normal(0, .01, shape)
        v0 = rng.normal(0, .02, shape)
        a0 = rng.normal(0, .4, shape)
        a1 = rng.normal(0, .3, shape)
        dt, beta, gamma = .04, .25, .5
        q1 = q0+dt*v0+dt**2*((.5-beta)*a0+beta*a1)
        v1 = v0+dt*((1-gamma)*a0+gamma*a1)
        return (q0,v0,a0,q1,v1,a1), dict(dt_s=dt,beta=beta,gamma=gamma)

    def test_exact_discrete_step_passes(self):
        arrays, args = self.make_step()
        defects = audit_newmark_step(*arrays, **args)
        self.assertLess(defects.displacement_abs_max,1e-15)
        self.assertLess(defects.velocity_abs_max,1e-15)
        self.assertIn("displacement_tolerance", defects.as_dict())

    def test_zero_motion_passes(self):
        z = np.zeros((6,4,4))
        self.assertEqual(audit_newmark_step(z,z,z,z,z,z,dt_s=.01,beta=.25,gamma=.5).velocity_abs_max,0.)

    def test_bad_displacement_rejected(self):
        a,k = self.make_step()
        modified = list(a)
        modified[3] = a[3].copy(); modified[3][0,1,2] += .0001
        with self.assertRaisesRegex(ValueError,"NEWMARK_KINEMATICS_DRIFT"):
            audit_newmark_step(*modified,**k)

    def test_bad_velocity_rejected(self):
        a,k = self.make_step()
        modified = list(a)
        modified[4] = a[4].copy();modified[4][1,2,3] += .002
        with self.assertRaisesRegex(ValueError,"NEWMARK_KINEMATICS_DRIFT"):
            audit_newmark_step(*modified,**k)

    def test_nonfinite_or_shape_mismatch_rejected(self):
        a,k=self.make_step()
        bad=list(a);bad[5] = np.zeros((4,2))
        with self.assertRaisesRegex(ValueError,"shape"):
            audit_newmark_step(*bad,**k)
        bad=list(a);bad[2] = a[2].copy();bad[2][0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            audit_newmark_step(*bad,**k)

    def test_invalid_parameters_rejected(self):
        a,k=self.make_step()
        for key,value in (("dt_s",0),("dt_s",float("nan")),("gamma",-1),
                          ("beta",False),("relative_tolerance",-1)):
            with self.subTest(key=key,value=value):
                with self.assertRaises(ValueError):
                    audit_newmark_step(*a,**{**k,key:value})

    def test_extremely_small_signal_does_not_divide_by_zero(self):
        z=np.zeros((1,4,4)); q1=z.copy();q1[0,1,2] = 5e-16
        audit_newmark_step(z,z,z,q1,z,z,dt_s=.1,beta=.25,gamma=.5)


if __name__=="__main__":unittest.main()
