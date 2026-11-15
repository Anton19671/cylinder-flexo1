"""Nominal radial traction work on a physical KL reference face.

Local continuum law only. Each numerical method supplies its own q, q_x,
q_s and assembles these unweighted coefficients with its own operators.
Signed traction is positive along the reference outward radial basis.
"""
from dataclasses import dataclass
import math
import numpy as np
from cylinder_kinematic_models import normalized_model


def _normal(q,qx,qs,radius_m,geometry_model):
    shape=np.asarray(q).shape
    if len(shape)!=3 or shape[0]!=3:
        raise ValueError("KL_FACE_REQUIRES_THREE_MECHANICAL_FIELDS")
    arrays=tuple(np.asarray(a,float) for a in (q,qx,qs))
    if any(a.shape!=shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("KL_FACE_REQUIRES_FINITE_MECHANICAL_JETS")
    q,qx,qs=arrays
    if normalized_model(geometry_model)!="GREEN_LAGRANGE":
        return arrays,None
    A=np.moveaxis(qx,0,-1).copy()
    B=np.moveaxis(qs,0,-1).copy()
    A[...,0]+=1.
    B[...,1]+=1.+q[2]/radius_m
    B[...,2]-=q[1]/radius_m
    N=np.cross(A,B)
    magnitude=np.linalg.norm(N,axis=-1)
    if not np.isfinite(magnitude).all() or np.any(magnitude<=1.e-10):
        raise ValueError("INVALID_KL_MIDSURFACE_NORMAL: degenerate metric")
    return arrays,(A,B,N/magnitude[...,None],magnitude)


def kl_radial_displacement(q,qx,qs,*,radius_m,z_m,geometry_model):
    arrays,normal=_normal(q,qx,qs,radius_m,geometry_model)
    radial=arrays[0][2].copy()
    if normal is not None:
        radial+=z_m*(normal[2][...,2]-1.)
    return radial


@dataclass(frozen=True)
class KLNominalRadialFaceLoad:
    traction_pa: np.ndarray
    radius_m: float
    z_m: float
    geometry_model: str

    def local_work(self,q,qx,qs):
        arrays,normal=_normal(q,qx,qs,self.radius_m,self.geometry_model)
        q=arrays[0]
        factor=self.traction_pa*(1.+self.z_m/self.radius_m)
        coefficients=np.zeros((3,3,*q.shape[1:]))
        radial=q[2].copy()
        coefficients[2,0]=factor
        if normal is not None:
            A,B,n,magnitude=normal
            radial+=self.z_m*(n[...,2]-1.)
            direction=-n*n[...,2,None]
            direction[...,2]+=1.
            direction/=magnitude[...,None]
            jx=self.z_m*np.cross(B,direction)
            js=self.z_m*np.cross(direction,A)
            coefficients[:,1]=np.moveaxis(jx*factor[...,None],-1,0)
            coefficients[:,2]=np.moveaxis(js*factor[...,None],-1,0)
            coefficients[1,0]-=factor*js[...,2]/self.radius_m
            coefficients[2,0]+=factor*js[...,1]/self.radius_m
        return factor*radial,coefficients


def compile_kl_face_load(spec,*,theory,shape,radius_m,thickness_m,geometry_model):
    if spec is None:
        return None
    if theory!="KL" or not isinstance(spec,dict) or set(spec)!={"face","traction_pa"}:
        raise ValueError("KL_FACE_LOAD_REQUIRES_EXPLICIT_KL_FACE_AND_TRACTION")
    R,h=float(radius_m),float(thickness_m)
    if not (math.isfinite(R) and math.isfinite(h) and R>h/2>0):
        raise ValueError("KL_FACE_LOAD_REQUIRES_R_GREATER_THAN_HALF_THICKNESS")
    if spec["face"] not in ("INNER","MIDDLE","OUTER"):
        raise ValueError("KL_FACE_LOAD_UNKNOWN_FACE")
    value=np.asarray(spec["traction_pa"],dtype=float)
    if value.shape==():
        value=np.full(shape,float(value))
    if value.shape!=tuple(shape) or not np.isfinite(value).all():
        raise ValueError("KL_FACE_LOAD_TRACTION_SHAPE_OR_FINITE")
    value=value.copy()
    value.setflags(write=False)
    z={"INNER":-h/2,"MIDDLE":0.,"OUTER":h/2}[spec["face"]]
    return KLNominalRadialFaceLoad(value,R,z,normalized_model(geometry_model))
