#!/usr/bin/env python3
"""Calculation-free checks of generated load resultants on all three faces."""
from __future__ import annotations

import math
import sys
import types
import numpy as np


class CylinderGrid:
    def __init__(self,L,R,nx,ns):
        self.length_x=float(L); self.radius=float(R)
        self.nx=int(nx); self.ns=int(ns)
        self.x=np.linspace(0.0,self.length_x,self.nx)
        self.dx=self.length_x/(self.nx-1)
        self.circumference=2.0*math.pi*self.radius
        self.ds=self.circumference/self.ns
        self.s=np.arange(self.ns,dtype=float)*self.ds


mvi=types.ModuleType("cylinder_te1_mvi_core")
mvi.CylinderGrid=CylinderGrid
sys.modules.setdefault("cylinder_te1_mvi_core",mvi)

g7=types.ModuleType("cylinder_cuf_stage7a_general_mechanics")
def radial_nominal_face_load(qr,*,zq,R,basis):
    qr=np.asarray(qr,float); Hq=1.0+float(zq)/float(R)
    values=np.asarray(basis.values(float(zq)),float)
    out=np.zeros((3,values.size)+qr.shape,float)
    for tau,value in enumerate(values):
        out[2,tau]=qr*Hq*value
    return out
g7.radial_nominal_face_load=radial_nominal_face_load
sys.modules.setdefault("cylinder_cuf_stage7a_general_mechanics",g7)

import final_load_builder as lb


class Basis:
    m=3
    @staticmethod
    def values(z):
        return np.array([1.0,z,z*z],float)


L=4.0; R=10.0; h=2.0; nx=17; ns=41


def integral(q):
    grid=CylinderGrid(L,R,nx,ns)
    wx=np.full(nx,grid.dx,float); wx[0]*=0.5; wx[-1]*=0.5
    return float(np.sum(np.asarray(q,float)*grid.ds*wx[None,:]))


def generated(spec,theory,surface):
    case=dict(spec,load_factors=[1.0],surface=surface)
    return lb.build_generated_loads(
        case,theory=theory,L=L,R=R,h=h,nx=nx,ns=ns,
        stencil=7,basis=Basis() if theory=="CUF" else None,
    )


def generalized_radial(loads,theory):
    return loads[0,2,0] if theory=="CUF" else loads[0,2]


def close(a,b,rtol=2.0e-13):
    assert abs(a-b)<=rtol*max(1.0,abs(b)),(a,b)


def main():
    assert lb.FACES==("OUTER","MIDDLE","INNER")
    for theory in ("CUF","KL"):
        for surface in lb.FACES:
            point={"type":"POINT_RADIAL_FORCE","P":123.5,"x0":1.7,"theta0":0.9}
            loads,_,meta,face_field=generated(point,theory,surface)
            close(integral(generalized_radial(loads,theory)),123.5)
            close(meta["physical_face_radial_resultant"],123.5)
            close((1.0+meta["surface_offset_zq"]/R)*integral(face_field),123.5)

            line={
                "type":"LINE_RADIAL_GENERATOR","p_line":7.25,
                "c0":1.0,"band_width":8.0,
            }
            loads,_,meta,_=generated(line,theory,surface)
            close(integral(generalized_radial(loads,theory)),7.25*L)
            close(meta["physical_face_radial_resultant"],7.25*L)

    # A prescribed pressure acts on the actual selected-face area.
    q=5.0
    for theory in ("CUF","KL"):
        for surface,zq in (("OUTER",h/2),("MIDDLE",0.0),("INNER",-h/2)):
            loads,_,meta,_=generated({"type":"UNIFORM_RADIAL_FACE","q":q},theory,surface)
            expected=q*2.0*math.pi*(R+zq)*L
            close(integral(generalized_radial(loads,theory)),expected)
            close(meta["physical_face_radial_resultant"],expected)

    print("LOAD_SURFACE_RESULTANT_SELF_CHECK: PASS")


if __name__=="__main__":
    main()
