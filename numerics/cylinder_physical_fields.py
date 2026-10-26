"""Physical 3D recovery from the SAME coupled shell state used by MVI/FDM.

Returns reference strain, covariant gradients, work-conjugate stress,
hyperstress, electric field and displacement across actual thickness points.
This is output only; it never alters equilibrium or boundary conditions.
"""
from __future__ import annotations
import numpy as np
from cylinder_functional_assembly import MECHANICAL_DERIVATIVES,ELECTRICAL_DERIVATIVES
from flexo_gradient_g3d import enthalpy_derivatives,enthalpy_parts
from cylinder_director_g3d import _displacement as _director_displacement
from cylinder_kl_g3d_weak import _data as _kl_data, _normal_jets as _kl_normal_jets
from cylinder_normal_double_traction import recover_normal_double_traction


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
    displacement=np.empty((n,nx,ns,3))
    strain=np.empty((n,nx,ns,6))
    gradient=np.empty((n,nx,ns,6,3))
    stress=np.empty_like(strain)
    hyper=np.empty_like(gradient)
    deformation_gradient=np.empty((n,nx,ns,3,3))
    classical_nominal=np.empty((n,nx,ns,3,3))
    double_traction=np.empty((n,nx,ns,3))
    electric=np.empty((n,nx,ns,3))
    induction=np.empty_like(electric)
    deformation_determinant=np.empty((n,nx,ns))
    energy=np.empty((n,nx,ns))
    qjets=functional._field_derivatives(q,MECHANICAL_DERIVATIVES[:functional.n_derivatives])
    pjets=functional._field_derivatives(p,ELECTRICAL_DERIVATIVES)
    for i in range(nx):
        for j in range(ns):
            jets=functional._make_jets(qjets,pjets,i,j)
            normal=(_kl_normal_jets(_kl_data(jets),functional.R)[0]
                    if functional.theory=="KL" else None)
            for k,zk in enumerate(z):
                point=functional._point(jets,float(zk))
                if functional.theory.startswith("CUF_TE"):
                    order=int(functional.theory[-1])
                    basis=np.array([(2.*zk/functional.h)**m for m in range(order+1)])
                    displacement[k,i,j]=jets.q@basis
                elif functional.theory in ("FSDT","REDDY"):
                    displacement[k,i,j]=_director_displacement(
                        jets,functional.theory,float(zk),functional.R,functional.h)
                else:
                    displacement[k,i,j]=jets.q+float(zk)*(
                        normal-np.array((0.,0.,1.)))
                s,m,d=enthalpy_derivatives(point.strain6,point.gradient6x3,
                                            point.electric3,functional.material)
                strain[k,i,j]=point.strain6
                gradient[k,i,j]=point.gradient6x3
                stress[k,i,j]=s
                hyper[k,i,j]=m
                deformation_gradient[k,i,j]=point.deformation_gradient
                nominal,double=recover_normal_double_traction(
                    point.deformation_gradient,s,m)
                classical_nominal[k,i,j]=nominal
                double_traction[k,i,j]=double
                electric[k,i,j]=point.electric3
                induction[k,i,j]=d
                deformation_determinant[k,i,j]=(
                    point.det_f if hasattr(point,"det_f") else point.determinant)
                energy[k,i,j]=enthalpy_parts(point.strain6,point.gradient6x3,
                                     point.electric3,functional.material)["total"]
    arrays=dict(z_m=z,displacement3_m=displacement,strain6=strain,gradient6x3=gradient,stress6_pa=stress,
        hyperstress6x3_pa_m=hyper,
        deformation_gradient3x3=deformation_gradient,
        classical_first_piola3x3_pa=classical_nominal,
        normal_double_traction3_n_per_m=double_traction,
        electric3_v_per_m=electric,
        induction3_c_per_m2=induction,det_F=deformation_determinant,
        enthalpy_j_per_m3=energy)
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite physical thickness fields")
    arrays["face_inner_displacement3_m"]=displacement[0].copy()
    arrays["face_outer_displacement3_m"]=displacement[-1].copy()
    arrays["face_inner_radial_displacement_m"]=displacement[0,:,:,2].copy()
    arrays["face_outer_radial_displacement_m"]=displacement[-1,:,:,2].copy()
    for side,k,normal in (("inner",0,-1.),("outer",-1,1.)):
        arrays[f"face_{side}_normal_double_traction3_n_per_m"]=double_traction[k].copy()
        arrays[f"face_{side}_classical_piola_outward_normal3_pa"]=(
            normal*classical_nominal[k,:,:, :,2]).copy()
    arrays["face_inner_stress6_pa"]=stress[0].copy()
    arrays["face_outer_stress6_pa"]=stress[-1].copy()
    arrays["face_inner_induction3_c_per_m2"]=induction[0].copy()
    arrays["face_outer_induction3_c_per_m2"]=induction[-1].copy()
    arrays["face_inner_hyperstress6x3_pa_m"]=hyper[0].copy()
    arrays["face_outer_hyperstress6x3_pa_m"]=hyper[-1].copy()
    return arrays


__all__=("recover_thickness_fields",)
