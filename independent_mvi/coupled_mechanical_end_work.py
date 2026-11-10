"""Explicit generalized mechanical end work for native MVI.

End q traces are essential (fixed), while normal slope and, where the
surface jet order permits, normal curvature remain admissible. Applied
double/triple actions enter the external potential by exact SBP
transposes, not strong boundary rows. Physical microfree face closure
is NOT certified by these user-prescribed generalized loads.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MechanicalEndWork:
    surface: object
    left: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        s=self.surface
        expected=(s.n_mechanical,max(a for a,b in s.mechanical_multi),
                  s.grid.ns)
        for name in ("left","right"):
            value=np.array(getattr(self,name),dtype=float,copy=True)
            if value.shape!=expected or not np.isfinite(value).all():
                raise ValueError("MVI_END_WORK_INVALID_SHAPE")
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    def nodal_gradient(self):
        s=self.surface
        result=np.zeros(s.shape)
        for node,coeff in ((0,self.left),(-1,self.right)):
            for order in range(coeff.shape[1]):
                trace=np.zeros((s.n_mechanical,s.grid.nx,s.grid.ns))
                trace[:,node,:]=-coeff[:,order,:]*s.grid.ws[None,:]
                result[:s.n_mechanical]+=np.einsum(
                    "ip,cij->cpj",s._dx_power[order],trace,optimize=True)
        return result

    def potential(self,factors):
        s=self.surface
        q=s.reconstruct(factors)[:s.n_mechanical]
        value=0.
        for node,coeff in ((0,self.left),(-1,self.right)):
            for order in range(coeff.shape[1]):
                trace=np.einsum("ip,cpj->cij",s._dx_power[order],q,
                                optimize=True)[:,node,:]
                value-=float(np.sum(coeff[:,order,:]*trace*
                                    s.grid.ws[None,:]))
        return value


def compile_mechanical_end_work(surface,spec):
    """Strict left/right axial or outward-normal generalized actions.

    Accept scalar or (n_mechanical,ns) arrays. q-traction must be zero
    because every mechanical q is prescribed at the axial ends.
    """
    if spec is None:
        return None
    if not isinstance(spec,dict) or set(spec)!={"left","right"}:
        raise ValueError("MVI_END_WORK_REQUIRES_LEFT_RIGHT")
    keys={"traction","double_traction","normal_double_traction",
          "triple_traction","normal_triple_traction"}
    count=max(a for a,b in surface.mechanical_multi)
    arrays=[]
    for side,normal in (("left",-1),("right",1)):
        data=spec[side]
        if data is None:
            data={}
        if not isinstance(data,dict) or set(data)-keys:
            raise ValueError("MVI_END_WORK_UNKNOWN_ACTION")
        if ({"double_traction","normal_double_traction"}<=set(data) or
            {"triple_traction","normal_triple_traction"}<=set(data)):
            raise ValueError("MVI_END_WORK_AMBIGUOUS_ORIENTATION")
        out=np.zeros((surface.n_mechanical,count,surface.grid.ns))
        for key,raw in data.items():
            if raw is None or isinstance(raw,(bool,np.bool_)):
                raise ValueError("MVI_END_WORK_NONFINITE_PROFILE")
            try:
                profile=np.asarray(raw,dtype=float)
            except (TypeError,ValueError,OverflowError) as exc:
                raise ValueError("MVI_END_WORK_NONFINITE_PROFILE") from exc
            if profile.ndim==0:
                profile=np.full((surface.n_mechanical,surface.grid.ns),
                                float(profile))
            if (profile.shape!=(surface.n_mechanical,surface.grid.ns)
                    or not np.isfinite(profile).all()):
                raise ValueError("MVI_END_WORK_INVALID_PROFILE_SHAPE")
            if key=="traction":
                if np.any(profile):
                    raise ValueError("MVI_END_FIXED_Q_TRACTION_IS_REACTION")
                continue
            order=1 if "double" in key else 2
            if order>=count:
                raise ValueError("MVI_END_WORK_TRACE_ORDER_UNAVAILABLE")
            out[:,order,:]=(normal if key=="normal_double_traction"
                            else 1.)*profile
        arrays.append(out)
    return MechanicalEndWork(surface,*arrays)


__all__=("MechanicalEndWork","compile_mechanical_end_work")
