"""Reconstruct candidate generalized face tractions of a G3D/Gz shell.

All components use the rotating orthonormal reference basis (x,s,z).
This is a strong-form RECOVERY DIAGNOSTIC from a finite thickness grid,
not a traction imposed in the coupled MVI/FDM equations and not an
independently verified strong solver.

For work-conjugate S=dH/dE, M=dH/d(E|k):
    Peff = F (S-Div M)
    Q_j = (F M_z)_j
    t = n_z (Peff)_z - Div_surface(n_z Q_t)
    g = Q_z                [n_z squared on both faces]
    l_end = +/- n_z Q_x    [line action at the axial edges]

Div M uses curvature of the cylindrical orthonormal basis on both
tensor indices and on the third (gradient) index. The surface
divergence includes rotation of the resulting physical vector.
"""
from __future__ import annotations
import numpy as np


def symmetric_tensor(components6):
    """Work-conjugate engineering shear components are NOT halved."""
    a=np.asarray(components6,dtype=float)
    if a.shape[-1]!=6 or not np.isfinite(a).all():
        raise ValueError("components6 must end in six finite values")
    t=np.empty(a.shape[:-1]+(3,3),float)
    t[...,0,0],t[...,1,1],t[...,2,2]=(a[...,k] for k in range(3))
    t[...,0,1]=t[...,1,0]=a[...,3]
    t[...,0,2]=t[...,2,0]=a[...,4]
    t[...,1,2]=t[...,2,1]=a[...,5]
    return t


def _derivative(values,matrix,axis):
    """Apply dense or CSR derivative in exactly one physical coordinate."""
    a=np.asarray(values,dtype=float)
    moved=np.moveaxis(a,axis,0)
    n=moved.shape[0]
    if matrix.shape!=(n,n):
        raise ValueError("surface derivative shape mismatch")
    result=matrix@moved.reshape(n,-1)
    return np.moveaxis(np.asarray(result).reshape(moved.shape),0,axis)


def generalized_face_tractions(recovered,operators,*,radius_m):
    """Return candidate generalized t, normal double g and axial line forces.

    Arrays: z[nz], F[nz,nx,ns,3,3], S[nz,nx,ns,6],
    M[nz,nx,ns,6,3]; nz>=3 for one-sided second-order z derivatives.
    These reconstructed strong residuals NEVER imply exact satisfaction
    of three-dimensional natural face conditions by finite-order CUF.
    """
    try:
        z=np.asarray(recovered["z_m"],dtype=float)
        f=np.asarray(recovered["deformation_gradient3x3"],dtype=float)
        s6=np.asarray(recovered["stress6_pa"],dtype=float)
        m6=np.asarray(recovered["hyperstress6x3_pa_m"],dtype=float)
    except KeyError as exc:
        raise ValueError("missing recovered deformation, stress or hyperstress") from exc
    try:radius=float(radius_m)
    except (TypeError,ValueError) as exc:
        raise ValueError("radius_m must be finite positive") from exc
    if (z.ndim!=1 or len(z)<3 or not np.isfinite(z).all()
            or not np.all(np.diff(z)>0) or not np.isfinite(radius)
            or radius+z[0]<=0):
        raise ValueError("ordered z grid and R+z>0 are required")
    nx,ns=operators.shape
    shape=(len(z),nx,ns)
    if (f.shape!=shape+(3,3) or s6.shape!=shape+(6,)
            or m6.shape!=shape+(6,3)
            or not np.isfinite(f).all() or not np.isfinite(s6).all()
            or not np.isfinite(m6).all()):
        raise ValueError("invalid thickness field shapes or nonfinite data")
    S=symmetric_tensor(s6)
    Mx,Ms,Mz=(symmetric_tensor(m6[...,k]) for k in range(3))
    omega=np.array(((0.,0.,0.),(0.,0.,1.),(0.,-1.,0.)))
    radius_z=(radius+z)[:,None,None,None,None]
    H=radius_z/radius
    # Covariant divergence in the orthonormal cylindrical frame.
    divM=(_derivative(Mx,operators.Dx,1)
          +_derivative(Ms,operators.Ds,2)/H
          +np.gradient(Mz,z,axis=0,edge_order=2)
          +(omega@Ms+Ms@omega.T+Mz)/radius_z)
    Peff=f@(S-divM)
    Q=f@Mz
    output={}
    for side,k,nz in (("inner",0,-1.),("outer",-1,1.)):
        face_radius=radius+z[k]
        Hf=face_radius/radius
        bx=nz*Q[k,...,0]
        bs=nz*Q[k,...,1]
        surface_div=(_derivative(bx,operators.Dx,0)
                     +_derivative(bs,operators.Ds,1)/Hf
                     +np.einsum("ab,...b->...a",omega,bs)/face_radius)
        output[f"face_{side}_generalized_traction3_pa_candidate"]=(
            nz*Peff[k,...,2]-surface_div)
        output[f"face_{side}_normal_double_traction3_n_per_m_candidate"]=(
            Q[k,...,2].copy())
        output[f"face_{side}_surface_hyper_divergence3_pa"]=surface_div
        output[f"face_{side}_effective_piola_normal3_pa"]=nz*Peff[k,...,2]
        output[f"face_{side}_left_end_line_force3_n_per_m"]=-bx[0].copy()
        output[f"face_{side}_right_end_line_force3_n_per_m"]=bx[-1].copy()
    if any(not np.isfinite(a).all() for a in output.values()):
        raise ValueError("nonfinite reconstructed generalized face traction")
    return output


__all__=("symmetric_tensor","generalized_face_tractions")
