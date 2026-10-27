import unittest
import numpy as np
from types import SimpleNamespace
from cylinder_gauss2_rk import Gauss2Settings,Gauss2State
from cylinder_gauss2_mvi import Gauss2MVISettings,_profile_field
from cylinder_gauss2_variational_mvi import solve_gauss2_variational_mvi_step

class ToyProblem:
    kind='DYNAMIC'
    def __init__(self,electrical=True,moving=False):
        self.electric=electrical;self.moving=moving;self.seen=[]
        self.functional=SimpleNamespace(electrical_on=electrical,theory='CUF_TE1',
            material=SimpleNamespace(young_pa=1.),h=1.,W=np.ones((4,4)),
            ops=SimpleNamespace(shape=(4,4)))
    def assemble(self,q,p=None,*,time_s,dead_load=None,free_charge=None,**kw):
        self.seen.append(('assemble',time_s))
        n=q.size+(0 if p is None else p.size)
        cc=np.zeros((8 if p is not None else 4,n))
        for s in range(4):
            cc[s,s]=1.
        if p is not None:
            for s in range(4):cc[4+s,q.size+s]=1.
        dd=np.zeros(len(cc))
        if self.moving:dd[:4]=time_s**2
        return SimpleNamespace(space=SimpleNamespace(matrix=cc,target=dd))
    def evaluate_effective_residual(self,q,p=None,*,time_s,acceleration=None,
                                     density_kg_m3=None,velocity=None,**kw):
        self.seen.append(('physics',time_s))
        a=np.zeros_like(q) if acceleration is None else acceleration
        rm=a+2*q+(0 if p is None else .2*p)
        electric=None if p is None else p-.5*q
        v=rm.ravel() if electric is None else np.concatenate((rm.ravel(),electric.ravel()))
        return 0.,v,None

class GaussPhysicalMviTests(unittest.TestCase):
    def mvi(self):
        return Gauss2MVISettings(rank=1,absolute_tolerance=1e-10,
          field_change_tolerance=1e-9,relative_tolerance=1e-9)
    def state(self,electric=True):
        x=np.array([0.,1.,2.,1.]);y=np.array([1.,2.,1.,3.]);q=np.outer(x,y)[None,...]
        return Gauss2State(0.,q,np.zeros_like(q),.5*q if electric else None)
    def test_physical_two_gauss_stages_and_electric_endpoint(self):
        pr=ToyProblem()
        result=solve_gauss2_variational_mvi_step(
            pr,self.state(),Gauss2Settings(.05),mvi_settings=self.mvi(),
            density_kg_m3=1.,loads_at=lambda t:(None,None))
        self.assertTrue(result.converged)
        for s in result.stages:
            np.testing.assert_allclose(s.electrical,.5*s.mechanical,atol=1e-9)
            np.testing.assert_allclose(s.mechanical[0,0],0.,atol=1e-12)
        np.testing.assert_allclose(result.final.electrical,.5*result.final.mechanical,atol=1e-9)
        np.testing.assert_allclose(result.final.mechanical[0,0],0.,atol=1e-10)
        np.testing.assert_allclose(_profile_field(result.x_profiles,result.y_profiles)*result.scales,
                                   np.concatenate((result.final.mechanical,result.final.electrical)),atol=1e-10)
        self.assertTrue(any(k=='physics' and 0<t<.05 for k,t in pr.seen))
    def test_moving_support_rejected(self):
        pr=ToyProblem(moving=True)
        with self.assertRaisesRegex(ValueError,'MOVING_MECHANICAL_SUPPORT'):
            solve_gauss2_variational_mvi_step(
                pr,self.state(),Gauss2Settings(.05),mvi_settings=self.mvi(),
                density_kg_m3=1.,loads_at=lambda t:(None,None))
    def test_without_electricity(self):
        pr=ToyProblem(electrical=False)
        result=solve_gauss2_variational_mvi_step(
            pr,self.state(False),Gauss2Settings(.05),mvi_settings=self.mvi(),
            density_kg_m3=1.,loads_at=lambda t:(None,None))
        self.assertIsNone(result.final.electrical)

if __name__=='__main__': unittest.main()
