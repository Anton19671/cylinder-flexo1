"""Manufactured local checks of the candidate covariant face operator."""
import numpy as np
from scipy.sparse import csr_matrix
from cylinder_generalized_face_traction import (
    symmetric_tensor,generalized_face_tractions)


class Operators:
    def __init__(self,nx=4,ns=6,dx=.2,ds=.4):
        self.shape=(nx,ns)
        self.Dx=np.zeros((nx,nx))
        self.Ds=np.zeros((ns,ns))
        for i in range(nx):
            if i==0:self.Dx[i,0]=-1/dx;self.Dx[i,1]=1/dx
            elif i==nx-1:self.Dx[i,i-1]=-1/dx;self.Dx[i,i]=1/dx
            else:self.Dx[i,i-1]=-1/(2*dx);self.Dx[i,i+1]=1/(2*dx)
        for j in range(ns):
            self.Ds[j,j]=-1/ds
            self.Ds[j,(j+1)%ns]=1/ds


def fields():
    z=np.linspace(-.1,.1,7)
    F=np.broadcast_to(np.eye(3),(7,4,6,3,3)).copy()
    S=np.zeros((7,4,6,6))
    M=np.zeros((7,4,6,6,3))
    return dict(z_m=z,deformation_gradient3x3=F,
                stress6_pa=S,hyperstress6x3_pa_m=M)


def test_constant_second_piola_zz_produces_opposite_faces():
    data=fields()
    data["stress6_pa"][...,2]=11.
    out=generalized_face_tractions(data,Operators(),radius_m=2.)
    for side,sign in (("inner",-1.),("outer",1.)):
        traction=out[f"face_{side}_generalized_traction3_pa_candidate"]
        np.testing.assert_allclose(traction,
            np.broadcast_to([0.,0.,sign*11.],traction.shape))


def test_cylindrical_curvature_cancels_constant_isotropic_Mz():
    data=fields()
    data["hyperstress6x3_pa_m"][...,[0,1,2],2]=4.
    out=generalized_face_tractions(data,Operators(),radius_m=2.)
    for side in ("inner","outer"):
        np.testing.assert_allclose(
            out[f"face_{side}_generalized_traction3_pa_candidate"],
            0.,atol=1e-13)
        g=out[f"face_{side}_normal_double_traction3_n_per_m_candidate"]
        np.testing.assert_allclose(g,np.broadcast_to([0,0,4.],g.shape))


def test_radial_linear_Mz_includes_covariant_radius_term():
    data=fields()
    data["hyperstress6x3_pa_m"][...,2,2]=data["z_m"][:,None,None]*8.
    out=generalized_face_tractions(data,Operators(),radius_m=2.)
    for side,sign in (("inner",-1.),("outer",1.)):
        expected=-sign*(8.+8.*(sign*.1)/(2.+sign*.1))
        traction=out[f"face_{side}_generalized_traction3_pa_candidate"]
        np.testing.assert_allclose(traction,
            np.broadcast_to([0,0,expected],traction.shape),atol=2e-12)


def test_engineering_shear_not_halved_twice():
    s=symmetric_tensor(np.array([1.,2.,3.,4.,5.,6.]))
    assert s[1,0]==4. and s[2,0]==5. and s[2,1]==6.


def test_dense_and_sparse_reconstruction_identical():
    op=Operators()
    class SparseOperators:
        shape=op.shape
        Dx=csr_matrix(op.Dx)
        Ds=csr_matrix(op.Ds)
    data=fields()
    data["stress6_pa"][...,2]=3.
    data["hyperstress6x3_pa_m"][...,3,1]=.01
    a=generalized_face_tractions(data,op,radius_m=2.)
    b=generalized_face_tractions(data,SparseOperators(),radius_m=2.)
    for key in a:
        np.testing.assert_allclose(a[key],b[key],rtol=0.,atol=1e-13)
