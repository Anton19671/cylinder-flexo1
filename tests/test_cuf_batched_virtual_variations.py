"""Regression of vectorized CUF exact virtual-jet variations (TE1--TE4)."""
import unittest
import numpy as np
from cuf_g3d_weak import (
    CUFSurfaceJets,_cuf_virtual_kinematic_library,
    _batched_virtual_strain_gradients,_strain_variation_from_deformation_jets,
    _kinematic_matrix_jets,thickness_weak_coefficients,gauss_thickness,
    point_state)
from flexo_gradient_g3d import (
    G3DMaterial,covariant_strain_gradients,enthalpy_derivatives)


class BatchedCUFVirtualWorkTests(unittest.TestCase):
    def test_all_te_orders_and_gradient_models(self):
        rng=np.random.default_rng(12034)
        R,h=2.,.2
        for order in (1,2,3,4):
            for model in ("G3D","Gz"):
                with self.subTest(order=order,model=model):
                    jets=CUFSurfaceJets(*(
                        .001*rng.normal(size=(3,order+1)) for _ in range(6)),
                        *(0.0001*rng.normal(size=(2,)) for _ in range(3)))
                    z=.031
                    F,Fx,Fs,Fz=_kinematic_matrix_jets(jets,z,R,h)
                    virtual=_cuf_virtual_kinematic_library(order,z,R,h)
                    e,g=_batched_virtual_strain_gradients(
                        virtual,F,Fx,Fs,Fz,radius_m=R,z_m=z)
                    for index,(dF,dFx,dFs,dFz) in enumerate(virtual.reshape(-1,4,3,3)):
                        de,(dex,des,dez)=_strain_variation_from_deformation_jets(
                            F,dF,Fx,Fs,Fz,dFx,dFs,dFz)
                        dg=covariant_strain_gradients(de,dex,des,dez,radius_m=R,z_m=z)
                        np.testing.assert_allclose(e[index],de,atol=3e-14,rtol=2e-13)
                        np.testing.assert_allclose(g[index],dg,atol=5e-13,rtol=2e-13)

    def test_thickness_integral_matches_original_scalar_work(self):
        rng=np.random.default_rng(76)
        R,h=2.,.2
        for order in (1,4):
            for model in ("G3D","Gz"):
                with self.subTest(order=order,model=model):
                    jets=CUFSurfaceJets(*(
                        1e-4*rng.normal(size=(3,order+1)) for _ in range(6)),
                        *(1e-5*rng.normal(size=(2,)) for _ in range(3)))
                    mat=G3DMaterial(50.,.25,.08,.001,-.002,1.5,gradient_model=model)
                    actual=thickness_weak_coefficients(
                        jets,mat,radius_m=R,thickness_m=h,n_gauss=3)
                    direct=np.zeros_like(actual.mechanical)
                    for z,w in zip(*gauss_thickness(h,3)):
                        state=point_state(jets,z_m=z,radius_m=R,thickness_m=h)
                        F,Fx,Fs,Fz=_kinematic_matrix_jets(jets,z,R,h)
                        S,M,_=enthalpy_derivatives(
                            state.strain6,state.gradient6x3,state.electric3,mat)
                        virtual=_cuf_virtual_kinematic_library(order,float(z),R,h)
                        for a in range(3):
                            for i in range(order+1):
                                for k in range(6):
                                    dF,dFx,dFs,dFz=virtual[a,i,k]
                                    de,(dx,ds,dz)=_strain_variation_from_deformation_jets(
                                        F,dF,Fx,Fs,Fz,dFx,dFs,dFz)
                                    dg=covariant_strain_gradients(
                                        de,dx,ds,dz,radius_m=R,z_m=z)
                                    direct[a,i,k]+=w*state.H*(np.dot(S,de)+np.sum(M*dg))
                    np.testing.assert_allclose(actual.mechanical,direct,atol=1e-11,rtol=2e-12)

if __name__=="__main__":unittest.main()
