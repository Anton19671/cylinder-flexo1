"""Partial physical mechanical face actions from a complete G3D/Gz state.

These are work-conjugate nominal stress and the *normal-normal* double
traction of the second-gradient bulk enthalpy. They do NOT constitute the
full natural mechanical face traction, which also needs tangential surface
divergence, curvature and the edge-line terms.

Local orthonormal reference frame is (x,s,z), engineering strains have
component ordering (xx,ss,zz,xs,xz,sz). The normal gradient index is z.
"""
from __future__ import annotations
import numpy as np


def symmetric_tensor_from_engineering_conjugate(stress6):
    """Convert dH/d(Exx,Ess,Ezz,2Exs,2Exz,2Esz) to symmetric tensor.

    The work-conjugate tensor off-diagonals equal the engineering
    derivatives, not one-half of them: S:delta E = sum sigma6*delta e6.
    """
    a=np.asarray(stress6,float)
    if a.shape!=(6,) or not np.isfinite(a).all():
        raise ValueError("engineering stress must have finite shape (6,)")
    return np.array(((a[0],a[3],a[4]),
                     (a[3],a[1],a[5]),
                     (a[4],a[5],a[2])),float)


def recover_normal_double_traction(deformation_gradient,stress6,hyper6x3):
    """Compute P_classical=F*S and g_a=(F_aj M_ij,z)_(i=z).

    By the definition of hyperstress M_ij,k=dH/d(E_ij|k), the double
    traction does work against the *outward-normal derivative* of the
    virtual displacement. n_z**2=1 for both shell thickness faces.
    Units: classical P in Pa, double traction in Pa*m = N/m.
    """
    F=np.asarray(deformation_gradient,float)
    h=np.asarray(hyper6x3,float)
    if (F.shape!=(3,3) or h.shape!=(6,3)
            or not np.isfinite(F).all() or not np.isfinite(h).all()):
        raise ValueError("F and hyperstress must have finite shapes (3,3),(6,3)")
    s=symmetric_tensor_from_engineering_conjugate(stress6)
    mz=symmetric_tensor_from_engineering_conjugate(h[:,2])
    nominal=F@s
    double=(F@mz)[:,2]
    return nominal,double


__all__=("symmetric_tensor_from_engineering_conjugate",
         "recover_normal_double_traction")
