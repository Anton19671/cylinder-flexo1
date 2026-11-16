"""Compatible cylindrical face-flux electrical subsystem for potential orders 0..4.

This is an INDEPENDENT electrical discretization of the same local dielectric
and flexoelectric potential enthalpy. It is not the full independent strong
mechanical/electric FDM: polarization, when supplied, is frozen external data.
The missing mechanical derivative of that polarization must be assembled
from the same discrete enthalpy before claiming a two-way coupled tangent.

An axial gradient lives at x-faces, a circumferential gradient at periodic
s-faces, and the through-thickness derivative at surface nodes. Potential
thickness coefficients use psi(x,s,z)=sum_m p_m(x,s)*(2*z/h)**m.
All returned weak actions are in C, not charge densities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from cylinder_staggered_surface import StaggeredCylinderSurface


@dataclass(frozen=True)
class FacePolarizationSamples:
    """P_x,P_s,P_z in C/m2 at Gauss z and compatible physical locations.

    The samples must come from the same constitutive mechanical state when
    used for coupled flexoelectric work. Face P_z is explicitly supplied:
    interior Gauss samples cannot be treated as exact face values.
    """
    axial_c_per_m2: np.ndarray       # (n_gauss,nx-1,ns)
    circumferential_c_per_m2: np.ndarray  # (n_gauss,nx,ns)
    transverse_c_per_m2: np.ndarray  # (n_gauss,nx,ns)
    inner_transverse_c_per_m2: np.ndarray | None = None  # (nx,ns)
    outer_transverse_c_per_m2: np.ndarray | None = None  # (nx,ns)


@dataclass(frozen=True)
class ModalElectricFaceResult:
    electrical_residual_c: np.ndarray                   # (nm,nx,ns)
    axial_mode_flux_c_per_m: np.ndarray                 # (nm,nx-1,ns)
    circumferential_mode_flux_c_per_m: np.ndarray       # (nm,nx,ns)
    local_thickness_mode_action_c: np.ndarray           # (nm,nx,ns)
    electric_potential_enthalpy_j: float
    gauge_mode_net_c: float
    z_gauss_m: np.ndarray                               # (n_gauss,)
    inner_outward_induction_c_per_m2: np.ndarray | None  # (nx,ns)
    outer_outward_induction_c_per_m2: np.ndarray | None  # (nx,ns)


def _scalar(value, name, *, positive=True):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name + " must be finite SI scalar")
    try: number=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError(name + " must be finite SI scalar") from exc
    if not math.isfinite(number) or (positive and number<=0):
        raise ValueError(name + " must be finite positive SI scalar")
    return number


def _polarization_arrays(value, *, n_gauss, nx, ns):
    if value is None:
        return None, None
    if not isinstance(value, FacePolarizationSamples):
        raise TypeError("polarization must be FacePolarizationSamples")
    raw=(value.axial_c_per_m2,value.circumferential_c_per_m2,
         value.transverse_c_per_m2)
    expected=((n_gauss,nx-1,ns),(n_gauss,nx,ns),(n_gauss,nx,ns))
    arrays=[]
    for a,shape in zip(raw,expected):
        a=np.asarray(a,float)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError("polarization Gauss samples have incorrect shape or nonfinite entries")
        arrays.append(a)
    inn=value.inner_transverse_c_per_m2
    out=value.outer_transverse_c_per_m2
    if (inn is None)!=(out is None):
        raise ValueError("both inner and outer polarization face values must be supplied together")
    ends=None
    if inn is not None:
        ends=[]
        for a in (inn,out):
            a=np.asarray(a,float)
            if a.shape!=(nx,ns) or not np.isfinite(a).all():
                raise ValueError("polarization face values must be finite (nx,ns)")
            ends.append(a)
    return tuple(arrays), ends


def modal_electric_face_weak(potential_coefficients_v, *, length_m, radius_m,
                             thickness_m, kappa_f_per_m, n_gauss=12,
                             polarization=None):
    """Independent electric potential weak action and compatible face flux.

    The scalar electric sector is H_e=-kappa*E^2/2-P.E, where
    E=(-psi_x,-psi_s/H,-psi_z) and H=1+z/R. P may be sampled externally,
    but its mechanical derivative is NOT computed here. Axial and periodic
    face gradients are paired with the exact negative discrete divergence,
    so even periodic grids have no spurious checkerboard electric null.

    The transverse derivative acts locally on thickness coefficients, not
    through a fictitious transverse nodal dimension. End x-faces are natural
    (no prescribed normal electric flux); electrodes/gauge constraints and
    free charges belong to the surrounding boundary assembler.
    """
    p=np.asarray(potential_coefficients_v,float)
    if (p.ndim!=3 or not 1<=p.shape[0]<=5 or min(p.shape[1:])<4
            or not np.isfinite(p).all()):
        raise ValueError("potential coefficients must be finite (1..5,nx>=4,ns>=4)")
    nm,nx,ns=p.shape
    L=_scalar(length_m,"length_m")
    R=_scalar(radius_m,"radius_m")
    h=_scalar(thickness_m,"thickness_m")
    kappa=_scalar(kappa_f_per_m,"kappa_f_per_m")
    if R<=h/2.:
        raise ValueError("require radius_m > thickness_m/2")
    if (isinstance(n_gauss,(bool,np.bool_)) or not isinstance(n_gauss,(int,np.integer))
            or n_gauss<max(2,nm)):
        raise ValueError("n_gauss must be integer >= max(2, number of potential modes)")
    if n_gauss>64:
        raise ValueError("n_gauss > 64 is not supported")
    surface=StaggeredCylinderSurface(nx,ns,L,R)
    p_x,p_s=surface.gradients(p)
    roots,weights=leggauss(int(n_gauss))
    z=roots*(h/2.)
    w=weights*(h/2.)
    pol,pol_ends=_polarization_arrays(
        polarization,n_gauss=int(n_gauss),nx=nx,ns=ns)
    fx=np.zeros((nm,nx-1,ns),float)
    fs=np.zeros((nm,nx,ns),float)
    local=np.zeros((nm,nx,ns),float)
    electric_enthalpy=0.
    nodal_area=surface.wx[:,None]*surface.ds
    axial_area=surface.dx*surface.ds
    powers_order=np.arange(nm,dtype=int)
    for k,(zk,wk) in enumerate(zip(z,w)):
        t=2.*zk/h
        b=t**powers_order
        db=np.zeros(nm)
        if nm>1:
            db[1:]=(2./h)*powers_order[1:]*t**(powers_order[1:]-1)
        H=1.+zk/R
        ex=-np.einsum("m,mij->ij",b,p_x)
        es=-np.einsum("m,mij->ij",b,p_s)/H
        ez=-np.einsum("m,mij->ij",db,p)
        if pol is None:
            px=ps=pz=0.
        else:
            px,ps,pz=(component[k] for component in pol)
        Dx=kappa*ex+px
        Ds=kappa*es+ps
        Dz=kappa*ez+pz
        fx+=wk*H*b[:,None,None]*Dx
        fs+=wk*b[:,None,None]*Ds
        local+=wk*H*db[:,None,None]*Dz
        electric_enthalpy-=wk*H*(
            axial_area*float(np.sum(.5*kappa*ex*ex+px*ex))+
            float(np.sum(nodal_area*(.5*kappa*es*es+ps*es)))+
            float(np.sum(nodal_area*(.5*kappa*ez*ez+pz*ez)))
        )
    local*=nodal_area[None,...]
    action=surface.weak_face_action(fx,fs)+local
    inner_b=np.array([(-1.)**m for m in range(nm)])
    outer_b=np.ones(nm)
    inner_db=np.zeros(nm);outer_db=np.zeros(nm)
    if nm>1:
        inner_db[1:]=(2./h)*powers_order[1:]*inner_b[:-1]
        outer_db[1:]=(2./h)*powers_order[1:]*outer_b[:-1]
    inner_dz=-kappa*np.einsum("m,mij->ij",inner_db,p)
    outer_dz=-kappa*np.einsum("m,mij->ij",outer_db,p)
    if pol is not None:
        if pol_ends is None:
            inner_outward=outer_outward=None
        else:
            inner_outward=-(inner_dz+pol_ends[0])
            outer_outward=outer_dz+pol_ends[1]
    else:
        inner_outward=-inner_dz
        outer_outward=outer_dz
    if not np.isfinite(electric_enthalpy) or not np.isfinite(action).all():
        raise ValueError("nonfinite electric modal flux/enthalpy")
    return ModalElectricFaceResult(
        electrical_residual_c=action,
        axial_mode_flux_c_per_m=fx,
        circumferential_mode_flux_c_per_m=fs,
        local_thickness_mode_action_c=local,
        electric_potential_enthalpy_j=float(electric_enthalpy),
        gauge_mode_net_c=float(np.sum(action[0])),
        z_gauss_m=z.copy(),
        inner_outward_induction_c_per_m2=inner_outward,
        outer_outward_induction_c_per_m2=outer_outward)




def collocate_cylindrical_polarization(nodal_polarization_c_per_m2):
    """Second-order vector-aware interpolation to physical x/s faces.

    Input is (ngauss,nx,ns,3) in local orthonormal (x,s,z) components.
    The axial frame is constant. At circumferential faces, adjacent local
    (s,z) vectors are first rotated into the common half-step face frame.
    This exactly preserves an arbitrary spatially constant Cartesian vector
    sampled on a circular cylinder, including an even periodic grid.
    """
    v=np.asarray(nodal_polarization_c_per_m2,float)
    if (v.ndim!=4 or v.shape[0]<1 or v.shape[1]<4 or v.shape[2]<4
            or v.shape[3]!=3 or not np.isfinite(v).all()):
        raise ValueError("nodal polarization must be finite (ngauss,nx>=4,ns>=4,3)")
    alpha=math.pi/v.shape[2]
    axial=.5*(v[:,1:,:,0]+v[:,:-1,:,0])
    circumferential=(.5*math.cos(alpha)*(v[...,1]+np.roll(v[...,1],-1,axis=2))
                     +.5*math.sin(alpha)*(np.roll(v[...,2],-1,axis=2)-v[...,2]))
    return axial,circumferential,v[...,2].copy()


def shell_flexopolarization_face_samples(functional, mechanical, *, n_gauss):
    """Sample the SAME G3D/Gz flexopolarization at electric face locations.

    Uses the active full shell's physical Green--Lagrange strain gradients,
    including the covariant cylindrical basis, to obtain P=mu1*t+mu2*b.
    The scalar mode-flux subsystem then freezes these samples: it returns
    the electric derivative of a well-defined face enthalpy, but not the
    mechanical derivative of that enthalpy. It cannot serve as a complete
    independent two-way flexoelectric strong FDM solver by itself.

    The face values of Pz are sampled at the ACTUAL z=+-h/2, not inferred
    by extrapolating from thickness Gauss points. In-plane P is sampled
    at each thickness Gauss point, then interpolated to corresponding x/s
    faces, with rotation into the physical circumferential face frame.
    """
    if (isinstance(n_gauss,(bool,np.bool_)) or
            not isinstance(n_gauss,(int,np.integer)) or
            not 2<=n_gauss<=64):
        raise ValueError("n_gauss must be an integer in [2,64]")
    nx,ns=functional.ops.shape
    q=np.asarray(mechanical,float)
    if (q.shape!=(functional.n_mechanical,nx,ns) or
            not np.isfinite(q).all()):
        raise ValueError("mechanical field has invalid shape or nonfinite entries")
    R=_scalar(functional.R,"functional.R")
    h=_scalar(functional.h,"functional.h")
    if R<=h/2:
        raise ValueError("require functional.R > functional.h/2")
    material=functional.material
    m1=float(material.mu1_c_per_m)
    m2=float(material.mu2_c_per_m)
    if not (math.isfinite(m1) and math.isfinite(m2)):
        raise ValueError("nonfinite material flexoelectric coefficient")
    zeros_axial=np.zeros((n_gauss,nx-1,ns))
    zeros_circ=np.zeros((n_gauss,nx,ns))
    zeros_node=np.zeros((n_gauss,nx,ns))
    zeros_face=np.zeros((nx,ns))
    if m1==0. and m2==0.:
        return FacePolarizationSamples(zeros_axial,zeros_circ,zeros_node,
                                       zeros_face.copy(),zeros_face.copy())
    from cylinder_functional_assembly import (MECHANICAL_DERIVATIVES,
                                                ELECTRICAL_DERIVATIVES)
    from flexo_gradient_g3d import (active_strain_gradients,
                                     gradient_invariants)
    mechanical_jets=functional._field_derivatives(
        q,MECHANICAL_DERIVATIVES[:functional.n_derivatives])
    electrical_jets=functional._field_derivatives(
        np.zeros((1,nx,ns),float),ELECTRICAL_DERIVATIVES)
    roots,_=leggauss(int(n_gauss))
    z=roots*(h/2.)
    sampled=np.empty((n_gauss,nx,ns,3),float)
    inn=np.empty((nx,ns),float)
    out=np.empty((nx,ns),float)
    for i in range(nx):
        for j in range(ns):
            jets=functional._make_jets(mechanical_jets,electrical_jets,i,j)
            for k,zk in enumerate(z):
                point=functional._point(jets,float(zk))
                active=active_strain_gradients(
                    point.gradient6x3,material.gradient_model)
                t,b=gradient_invariants(active)
                sampled[k,i,j]=m1*t+m2*b
            for face_z,face_array in ((-h/2.,inn),(h/2.,out)):
                point=functional._point(jets,float(face_z))
                active=active_strain_gradients(
                    point.gradient6x3,material.gradient_model)
                t,b=gradient_invariants(active)
                face_array[i,j]=(m1*t+m2*b)[2]
    if (not np.isfinite(sampled).all() or not np.isfinite(inn).all()
            or not np.isfinite(out).all()):
        raise ValueError("nonfinite constitutive shell polarization")
    px,ps,pz=collocate_cylindrical_polarization(sampled)
    return FacePolarizationSamples(px,ps,pz,inn,out)


__all__=("FacePolarizationSamples","ModalElectricFaceResult",
         "modal_electric_face_weak", "collocate_cylindrical_polarization",
         "shell_flexopolarization_face_samples")
