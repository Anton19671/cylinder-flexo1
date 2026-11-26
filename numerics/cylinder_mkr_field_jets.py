"""MKR-owned field preparation for the existing experimental strong route.

Uses second-order centered/one-sided x first derivatives, an independent
three-point x Laplacian with second-order one-sided endpoint closures,
and forward periodic first s derivatives. Second s derivatives use the
compatible periodic Laplacian. Pure third-order axial jets use an explicit
seven-point polynomial derivative, including physical end stencils: taking
D_x(D_xx q) otherwise loses boundary accuracy even for a quartic polynomial.
In particular D_x^2 is NOT implemented as two centered D_x
applications, which would lose the axial checkerboard mode. Coefficient
divergences are assembled separately by cylinder_direct_strong_fdm. No CylinderFunctional, MVI, SBP or shared surface
matrix operations are called. This extraction is not a new stencil scheme
or certification of the full compatible nonlinear face-flux discretization.
"""
from dataclasses import dataclass
import numpy as np
from cylinder_direct_strong_fdm import (
    DirectStrongGrid, direct_coefficient_derivative,
    matched_kinematic_axial_derivative,
)


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
        if self.grid.matched_axial is not None:
            return matched_kinematic_axial_derivative(value,1,self.grid)
        out=np.empty_like(value)
        out[:,1:-1,:]=(value[:,2:,:]-value[:,:-2,:])/(2*self.grid.dx)
        out[:,0,:]=(-1.5*value[:,0,:]+2*value[:,1,:]-.5*value[:,2,:])/self.grid.dx
        out[:,-1,:]=(.5*value[:,-3,:]-2*value[:,-2,:]+1.5*value[:,-1,:])/self.grid.dx
        return out

    def _xx(self, value):
        """Direct axial curvature with nonzero interior Nyquist response.

        Repeated centered first differences would use a two-cell stride,
        decouple alternating axial nodes and annihilate the checkerboard.
        """
        if self.grid.matched_axial is not None:
            return matched_kinematic_axial_derivative(value,2,self.grid)
        h2=self.grid.dx**2
        out=np.empty_like(value)
        out[:,1:-1,:]=(value[:,2:,:]-2.*value[:,1:-1,:]+
                       value[:,:-2,:])/h2
        out[:,0,:]=(2.*value[:,0,:]-5.*value[:,1,:]+
                     4.*value[:,2,:]-value[:,3,:])/h2
        out[:,-1,:]=(2.*value[:,-1,:]-5.*value[:,-2,:]+
                      4.*value[:,-3,:]-value[:,-4,:])/h2
        return out

    def _xxx(self, value):
        """Seven-point axial third derivative, accurate at physical ends.

        A composition of the interior three-point curvature stencil and
        the one-sided first derivative has a first-order endpoint defect:
        for q=x**4 at x=0, D_x(D_xx q)=36*dx instead of zero.
        Use the native direct-strong Fornberg derivative for q_xxx;
        this does not certify its adjoint or sixth-order end closure.
        """
        if self.grid.matched_axial is not None:
            return matched_kinematic_axial_derivative(value,3,self.grid)
        return direct_coefficient_derivative(value,3,0,self.grid)

    def _s(self, value):
        return (np.roll(value,-1,axis=-1)-value)/self.grid.ds

    def _ss(self, value):
        return (np.roll(value,-1,axis=-1)-2*value+
                np.roll(value,1,axis=-1))/self.grid.ds**2

    def derivative(self, value, axial_order, circum_order):
        if (type(axial_order) is not int or type(circum_order) is not int or
                min(axial_order,circum_order)<0 or axial_order+circum_order>3):
            raise ValueError("MKR field derivative orders require integers with total <=3")
        out=self._field(value).copy()
        if circum_order>=2:
            out=self._ss(out)
            if circum_order==3:
                out=self._s(out)
        elif circum_order==1:
            out=self._s(out)
        if axial_order==1:
            out=self._x(out)
        elif axial_order==2:
            out=self._xx(out)
        elif axial_order==3:
            out=self._xxx(out)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite MKR field derivative")
        return out

    def prepare(self, value, orders):
        """Compute unique jets once, preserving the supplied output order."""
        cache={(0,0):self._field(value).copy()}
        def get(ax,ss):
            key=(ax,ss)
            if key not in cache:
                if ax==1:
                    cache[key]=self._x(get(0,ss))
                elif ax==2:
                    cache[key]=self._xx(get(0,ss))
                elif ax==3:
                    cache[key]=self._xxx(get(0,ss))
                elif ss==2:
                    cache[key]=self._ss(get(0,0))
                else:
                    cache[key]=self._s(get(0,ss-1))
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
