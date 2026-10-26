"""Physical 3D recovery from the SAME coupled shell state used by MVI/FDM.

Returns reference strain, covariant gradients, work-conjugate stress,
hyperstress, electric field and displacement across actual thickness points.
This is output only; it never alters equilibrium or boundary conditions.
"""
from __future__ import annotations
import numpy as np
from cylinder_functional_assembly import MECHANICAL_DERIVATIVES,ELECTRICAL_DERIVATIVES
from flexo_gradient_g3d import enthalpy_derivatives,enthalpy_parts


def recover_thickness_fields(functional,mechanical,electrical=None,*,n_thickness=11):
    if type(n_thickness) is not int or not 3<=n_thickness<=201:
        raise ValueError("n_thickness must be an integer in [3,201]")
    nx,ns=functional.ops.shape
    q=np.asarray(mechanical,float)
    if q.shape!=(functional.n_mechanical,nx,ns) or not np.isfinite(q).all():
        raise ValueError("invalid mechanical coefficient fields")
    if functional.electrical_on:
        if electrical is None:raise ValueError("electric coefficients required")
        p=np.asarray(electrical,float)
        if p.ndim!=3 or p.shape[1:]!=(nx,ns) or not 1<=p.shape[0]<=5 or not np.isfinite(p).all():
            raise ValueError("invalid electric coefficients")
    else:
        if electrical is not None:raise ValueError("electric coefficients must be disabled")
        p=np.zeros((1,nx,ns),float)
    z=np.linspace(-functional.h/2,functional.h/2,n_thickness)
    n=n_thickness
    strain=np.empty((n,nx,ns,6))
    gradient=np.empty((n,nx,ns,6,3))
    stress=np.empty_like(strain)
    hyper=np.empty_like(gradient)
    electric=np.empty((n,nx,ns,3))
    induction=np.empty_like(electric)
    deformation_determinant=np.empty((n,nx,ns))
    energy=np.empty((n,nx,ns))
    qjets=functional._field_derivatives(q,MECHANICAL_DERIVATIVES[:functional.n_derivatives])
    pjets=functional._field_derivatives(p,ELECTRICAL_DERIVATIVES)
    for i in range(nx):
        for j in range(ns):
            jets=functional._make_jets(qjets,pjets,i,j)
            for k,zk in enumerate(z):
                point=functional._point(jets,float(zk))
                s,m,d=enthalpy_derivatives(point.strain6,point.gradient6x3,
                                            point.electric3,functional.material)
                strain[k,i,j]=point.strain6
                gradient[k,i,j]=point.gradient6x3
                stress[k,i,j]=s
                hyper[k,i,j]=m
                electric[k,i,j]=point.electric3
                induction[k,i,j]=d
                deformation_determinant[k,i,j]=point.determinant
                energy[k,i,j]=enthalpy_parts(point.strain6,point.gradient6x3,
                                     point.electric3,functional.material)["total"]
    arrays=dict(z_m=z,strain6=strain,gradient6x3=gradient,stress6_pa=stress,
        hyperstress6x3_pa_m=hyper,electric3_v_per_m=electric,
        induction3_c_per_m2=induction,det_F=deformation_determinant,
        enthalpy_j_per_m3=energy)
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite physical thickness fields")
    arrays["face_inner_stress6_pa"]=stress[0].copy()
    arrays["face_outer_stress6_pa"]=stress[-1].copy()
    arrays["face_inner_induction3_c_per_m2"]=induction[0].copy()
    arrays["face_outer_induction3_c_per_m2"]=induction[-1].copy()
    arrays["face_inner_hyperstress6x3_pa_m"]=hyper[0].copy()
    arrays["face_outer_hyperstress6x3_pa_m"]=hyper[-1].copy()
    return arrays


__all__=("recover_thickness_fields",)
