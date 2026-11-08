"""MKR-owned field preparation for the existing experimental strong route.

Preserves the former uniform-trapezoid kinematic stencils: second-order
centered/one-sided x derivative and forward periodic s derivative, composed
for higher/mixed jets. Coefficient divergences are assembled separately by
cylinder_direct_strong_fdm. No CylinderFunctional, MVI, SBP or shared surface
matrix operations are called. This extraction is not a new stencil scheme
or certification of the full compatible nonlinear face-flux discretization.
"""
from dataclasses import dataclass
import numpy as np
from cylinder_direct_strong_fdm import DirectStrongGrid


MECHANICAL_JETS = (
    ("q",0,0), ("qx",1,0), ("qs",0,1),
    ("qxx",2,0), ("qxs",1,1), ("qss",0,2),
    ("qxxx",3,0), ("qxxs",2,1), ("qxss",1,2), ("qsss",0,3),
)
ELECTRIC_JETS = (("psi",0,0), ("psix",1,0), ("psis",0,1))


@dataclass(frozen=True)
class MKRFieldJets:
    grid: DirectStrongGrid

    def __post_init__(self):
        if not isinstance(self.grid, DirectStrongGrid):
            raise TypeError("DirectStrongGrid required")

    def _field(self, value):
        a=np.asarray(value,float)
        if (a.ndim!=3 or a.shape[0]<1 or
                a.shape[1:]!=(self.grid.nx,self.grid.ns) or
                not np.isfinite(a).all()):
            raise ValueError("MKR field requires finite (component,nx,ns) values")
        return a

    def _x(self, value):
        out=np.empty_like(value)
        out[:,1:-1,:]=(value[:,2:,:]-value[:,:-2,:])/(2*self.grid.dx)
        out[:,0,:]=(-1.5*value[:,0,:]+2*value[:,1,:]-.5*value[:,2,:])/self.grid.dx
        out[:,-1,:]=(.5*value[:,-3,:]-2*value[:,-2,:]+1.5*value[:,-1,:])/self.grid.dx
        return out

    def _s(self, value):
        return (np.roll(value,-1,axis=-1)-value)/self.grid.ds

    def derivative(self, value, axial_order, circum_order):
        if (type(axial_order) is not int or type(circum_order) is not int or
                min(axial_order,circum_order)<0 or axial_order+circum_order>3):
            raise ValueError("MKR field derivative orders require integers with total <=3")
        out=self._field(value).copy()
        for _ in range(circum_order):
            out=self._s(out)
        for _ in range(axial_order):
            out=self._x(out)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite MKR field derivative")
        return out

    def prepare(self, value, orders):
        """Compute unique jets once, preserving the supplied output order."""
        cache={(0,0):self._field(value).copy()}
        def get(ax,ss):
            key=(ax,ss)
            if key not in cache:
                cache[key]=(self._x(get(ax-1,ss)) if ax else self._s(get(0,ss-1)))
            return cache[key]
        result=[]
        for entry in orders:
            if (not isinstance(entry,(tuple,list)) or len(entry)!=3 or
                    type(entry[1]) is not int or type(entry[2]) is not int or
                    min(entry[1:])<0 or entry[1]+entry[2]>3):
                raise ValueError("MKR jet requires (name,axial,circum), total <=3")
            out=get(entry[1],entry[2])
            if not np.isfinite(out).all():
                raise ValueError("nonfinite MKR field jet")
            result.append(out)
        return tuple(result)


__all__=("MKRFieldJets", "MECHANICAL_JETS", "ELECTRIC_JETS")
