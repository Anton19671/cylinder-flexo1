#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VTK legacy export helpers for the final cylinder solution.

The cylinder surface is topologically CLOSED in the circumferential direction:
last theta row is connected back to the first row with quadrilateral cells.
This avoids the artificial longitudinal seam that appears when a point cloud
or an open surface is exported.
"""
from pathlib import Path
import numpy as np


def _safe_scalar_name(name):
    out="".join(ch if (ch.isalnum() or ch=="_") else "_" for ch in str(name))
    return out or "field"


def export_vtk(filename, xyz, displacement=None, scalars=None):
    """Backward-compatible point-cloud exporter."""
    xyz=np.asarray(xyz,float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be Nx3")
    n=len(xyz)
    filename=Path(filename)
    filename.parent.mkdir(parents=True,exist_ok=True)
    with filename.open("w",encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Solver result\nASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for p in xyz:
            f.write(f"{p[0]:.16g} {p[1]:.16g} {p[2]:.16g}\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        if displacement is not None:
            d=np.asarray(displacement,float)
            if d.shape!=(n,3):
                raise ValueError("displacement must be Nx3")
            f.write("VECTORS displacement float\n")
            for v in d:
                f.write(f"{v[0]:.16g} {v[1]:.16g} {v[2]:.16g}\n")
        if scalars:
            for name,val in scalars.items():
                a=np.asarray(val,float).reshape(-1)
                if a.size!=n:
                    raise ValueError(f"scalar {name!r} must contain {n} values")
                f.write(f"SCALARS {_safe_scalar_name(name)} float 1\nLOOKUP_TABLE default\n")
                for x in a:
                    f.write(f"{x:.16g}\n")


def cylinder_surface_arrays(u,v,w,L,R,deformation_scale=1.0,deformed_points=False):
    """Return cylinder midsurface points and Cartesian displacement vectors.

    Solver components are interpreted as:
      u - axial, v - circumferential, w - radial.
    Arrays have shape (ns, nx); theta=0 and theta=2*pi are NOT duplicated.
    Closure is represented by VTK connectivity, not duplicate points.
    """
    u=np.asarray(u,float);v=np.asarray(v,float);w=np.asarray(w,float)
    if u.shape!=v.shape or u.shape!=w.shape or w.ndim!=2:
        raise ValueError("u, v, w must have the same 2D shape (ns, nx)")
    ns,nx=w.shape
    if ns<3 or nx<2:
        raise ValueError("closed cylinder surface requires ns>=3 and nx>=2")
    xx=np.linspace(0.0,float(L),nx)
    th=np.linspace(0.0,2.0*np.pi,ns,endpoint=False)
    X,T=np.meshgrid(xx,th)
    ct=np.cos(T); st=np.sin(T)
    base=np.stack((X,float(R)*ct,float(R)*st),axis=-1)
    disp=np.stack((u,w*ct-v*st,w*st+v*ct),axis=-1)
    pts=base + float(deformation_scale)*disp if deformed_points else base
    return pts.reshape(-1,3),disp.reshape(-1,3)


def export_cylinder_surface_vtk(filename,u,v,w,L,R,scalars=None,deformed_points=False,deformation_scale=1.0,title="Final cylinder shell"):
    """Export a connected, circumferentially closed cylindrical midsurface.

    The mesh uses quads.  For circumferential row j=ns-1, each quad connects
    to row j=0, so there is no longitudinal gap in ParaView.
    """
    u=np.asarray(u,float);v=np.asarray(v,float);w=np.asarray(w,float)
    pts,disp=cylinder_surface_arrays(u,v,w,L,R,deformation_scale,deformed_points)
    ns,nx=w.shape
    filename=Path(filename); filename.parent.mkdir(parents=True,exist_ok=True)

    scalar_fields={"w_radial":w,"u_axial":u,"v_circumferential":v,"displacement_magnitude":np.sqrt(u*u+v*v+w*w)}
    if scalars:
        scalar_fields.update(scalars)

    npts=ns*nx
    ncells=ns*(nx-1)
    with filename.open("w",encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(str(title).replace("\n"," ")+"\nASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {npts} float\n")
        for p in pts:
            f.write(f"{p[0]:.16g} {p[1]:.16g} {p[2]:.16g}\n")

        # Each surface cell has 4 point ids.  The modulo operation closes the
        # circumference: the last theta strip connects directly to theta=0.
        f.write(f"POLYGONS {ncells} {ncells*5}\n")
        for j in range(ns):
            jn=(j+1)%ns
            for i in range(nx-1):
                p00=j*nx+i
                p01=j*nx+i+1
                p11=jn*nx+i+1
                p10=jn*nx+i
                f.write(f"4 {p00} {p01} {p11} {p10}\n")

        f.write(f"POINT_DATA {npts}\n")
        f.write("VECTORS displacement float\n")
        for d in disp:
            f.write(f"{d[0]:.16g} {d[1]:.16g} {d[2]:.16g}\n")
        for name,val in scalar_fields.items():
            a=np.asarray(val,float)
            if a.shape!=(ns,nx):
                raise ValueError(f"scalar {name!r} must have shape {(ns,nx)}, got {a.shape}")
            f.write(f"SCALARS {_safe_scalar_name(name)} float 1\nLOOKUP_TABLE default\n")
            for x in a.reshape(-1):
                f.write(f"{x:.16g}\n")
    return filename


if __name__=="__main__":
    print("Use export_cylinder_surface_vtk(...) for a closed cylinder surface.")
