"""Manufactured boundary traces for the independent direct strong route."""
import unittest
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid,DirectStrongBalance
from cylinder_direct_strong_boundary import assemble_physical_end_rows

def bc(kind,value=0.):
    return {"kind":kind,"value":value}

class StrongEndClosureTests(unittest.TestCase):
    def setUp(self):
        g=self.g=DirectStrongGrid(11,8,2.,1.)
        x=(np.arange(g.nx)*g.dx)
        self.q=np.zeros((3,g.nx,g.ns))
        self.q[0]=(3+2*x)[:,None]
        self.q[1]=x[:,None]**2
        self.q[2]=x[:,None]**3
        self.b=DirectStrongBalance(
            np.ones_like(self.q),
            np.ones((1,g.nx,g.ns)),
            np.broadcast_to(np.array([-4.,4.])[:,None,None,None],
                            (2,3,g.ns)).copy() if False else np.stack(
                            (np.full((3,g.ns),-4.),np.full((3,g.ns),4.))),
            np.stack((np.full((3,g.ns),7.),np.full((3,g.ns),7.))),
            np.stack((np.full((3,g.ns),-5.),np.full((3,g.ns),5.))),
            np.stack((np.full((1,g.ns),-2.),np.full((1,g.ns),2.))),
            tuple(range(3,g.nx-3)),3)

    def test_natural_slope_has_same_sign_on_both_ends(self):
        m=(1,2,3)
        conditions={"left":[[bc("NATURAL",-4.)],
                            [bc("ESSENTIAL"),bc("NATURAL",7.)],
                            [bc("ESSENTIAL"),bc("NATURAL",7.),bc("NATURAL",-5.)]],
                    "right":[[bc("NATURAL",4.)],
                             [bc("ESSENTIAL",4.),bc("NATURAL",7.)],
                             [bc("ESSENTIAL",8.),bc("NATURAL",7.),bc("NATURAL",5.)]]}
        out=assemble_physical_end_rows(
            self.b,self.q,self.g,axial_orders=m,mechanical_ends=conditions,
            electrical=np.zeros((1,self.g.nx,self.g.ns)),
            electrical_ends={"left":[bc("ESSENTIAL")],"right":[bc("NATURAL",2.)]})
        for i,order in enumerate(m):
            np.testing.assert_allclose(out.mechanical[i,:order],0.)
            np.testing.assert_allclose(out.mechanical[i,-order:],0.)
            np.testing.assert_allclose(out.mechanical[i,order:-order],1.)
        np.testing.assert_allclose(out.electrical[:,0],0.)
        np.testing.assert_allclose(out.electrical[:,-1],0.)

    def test_electric_gauge_not_silently_invented(self):
        ends={"left":[[bc("ESSENTIAL")]]*3,"right":[[bc("ESSENTIAL")]]*3}
        # Expand to match declared orders.
        ends={"left":[[bc("ESSENTIAL")]*m for m in (1,2,3)],
              "right":[[bc("ESSENTIAL")]*m for m in (1,2,3)]}
        with self.assertRaisesRegex(ValueError,"electrode or gauge"):
            assemble_physical_end_rows(
                self.b,self.q,self.g,axial_orders=(1,2,3),
                mechanical_ends=ends,
                electrical=np.zeros((1,self.g.nx,self.g.ns)),
                electrical_ends={"left":[bc("NATURAL")],
                                 "right":[bc("NATURAL")]})

    def test_reject_missing_field_specific_rows(self):
        ends={"left":[[bc("ESSENTIAL")]*m for m in (1,2,3)],
              "right":[[bc("ESSENTIAL")]*m for m in (1,2,3)]}
        ends["left"][2]=[bc("ESSENTIAL")]
        with self.assertRaisesRegex(ValueError,"exactly m"):
            assemble_physical_end_rows(self.b,self.q,self.g,
                  axial_orders=(1,2,3),mechanical_ends=ends)
        with self.assertRaisesRegex(ValueError,"certified axial orders"):
            assemble_physical_end_rows(self.b,self.q,self.g,
                  axial_orders=(1,2,4),mechanical_ends=ends)

if __name__=="__main__":
    unittest.main()
