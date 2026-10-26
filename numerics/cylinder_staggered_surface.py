"""Compatible first-derivative face pair on a periodic cylindrical surface.

Node fields have axial and circumferential axes last. Axial gradients live
on the nx-1 interior faces; periodic gradients live on ns circumferential
faces. The weak face action is exactly the weighted transpose of those
gradients, with one physical value at each periodic face. Consequently an
even-grid alternating nodal field is visible to the face gradient.

These operators are the first-order building block of §40, not the complete
G3D mechanical strong operator: higher mixed jets and constitutive fluxes
must be collocated consistently before a direct strong solver is dispatched.
"""
from __future__ import annotations

from dataclasses import dataclass,field
import math
import numpy as np


@dataclass(frozen=True)
class StaggeredCylinderSurface:
    nx: int
    ns: int
    length_m: float
    radius_m: float
    dx: float=field(init=False)
    ds: float=field(init=False)
    wx: np.ndarray=field(init=False,repr=False)

    def __post_init__(self):
        for name in ("nx","ns"):
            count=getattr(self,name)
            if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count<4:
                raise ValueError(name+" must be an integer >= 4")
        for name in ("length_m","radius_m"):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)):
                raise ValueError(name+" must be positive finite")
            try:value=float(value)
            except (TypeError,ValueError) as exc:
                raise ValueError(name+" must be positive finite") from exc
            if not math.isfinite(value) or value<=0:
                raise ValueError(name+" must be positive finite")
            object.__setattr__(self,name,value)
        dx=self.length_m/(self.nx-1)
        ds=2*math.pi*self.radius_m/self.ns
        wx=np.full(self.nx,dx)
        wx[[0,-1]]*=.5
        wx.flags.writeable=False
        object.__setattr__(self,"dx",dx)
        object.__setattr__(self,"ds",ds)
        object.__setattr__(self,"wx",wx)

    def gradients(self,nodes):
        """Return physical x/s face gradients for a 2D or stacked 3D field."""
        u=np.asarray(nodes,float)
        if (u.ndim not in (2,3) or u.shape[-2:]!=(self.nx,self.ns)
                or not np.isfinite(u).all()):
            raise ValueError("finite nodal field (nx,ns) or (nfield,nx,ns) required")
        return ((u[...,1:,:]-u[...,:-1,:])/self.dx,
                (np.roll(u,-1,axis=-1)-u)/self.ds)

    def weak_face_action(self,axial_flux,circumferential_flux):
        """Return Gx.T Wx_face Fx + Gs.T Ws_face Fs in nodal work units.

        Fx/Fs are thickness-integrated physical coefficients on their own
        faces. Face quadrature is dx*ds axially and wx[i]*ds periodically.
        No edge ghost nodes or artificial periodic duplicate are introduced.
        """
        fx=np.asarray(axial_flux,float)
        fs=np.asarray(circumferential_flux,float)
        if (fx.ndim not in (2,3) or fs.ndim!=fx.ndim or
                fx.shape[-2:]!=(self.nx-1,self.ns) or
                fs.shape[:-2]!=fx.shape[:-2] or
                fs.shape[-2:]!=(self.nx,self.ns) or
                not np.isfinite(fx).all() or not np.isfinite(fs).all()):
            raise ValueError("finite matched axial/circumferential face fluxes required")
        action=np.zeros_like(fs)
        action[...,:-1,:]-=self.ds*fx
        action[...,1:,:]+=self.ds*fx
        weights=self.wx.reshape((1,)*(fs.ndim-2)+(self.nx,1))
        action+=weights*(np.roll(fs,1,axis=-1)-fs)
        return action


__all__=("StaggeredCylinderSurface",)
