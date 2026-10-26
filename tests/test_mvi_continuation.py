import numpy as np
from types import SimpleNamespace
from cylinder_mvi_continuation import continue_accepted_mvi_profiles
def test_no_additive_field_in_continuation():
 x=np.ones((1,1,4));y=np.ones((1,1,5));scale=np.ones((1,4,5))
 field=np.einsum("crx,crs->cxs",x,y)
 prev=SimpleNamespace(converged=True,mvi_certificate={"accepted":True},
  x_profiles=x,y_profiles=y,scales=scale)
 class Space:
  def defect(self,u):return u-2.
 xx,yy,d=continue_accepted_mvi_profiles(prev,field,None,
   new_scales=scale,full_space=Space(),rank=1)
 np.testing.assert_allclose(np.einsum("crx,crs->cxs",xx,yy),field)
 assert d["new_trace_defect_before_halfstep_scaled"]==1.
