#!/usr/bin/env python3
"""Small deterministic tests of KL/G3D and temporal MVI contracts (not FEM)."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from cylinder_paths import activate
activate()
from cylinder_kl_g3d_weak import (KLSurfaceJets,kl_point,
     thickness_enthalpy,thickness_weak_coefficients,FIELDS,ELEC_FIELDS)
from flexo_gradient_g3d import G3DMaterial
from cylinder_newmark_mvi import (TransientState,NewmarkSettings,SpatialStep,
     integrate_newmark,DynamicInputError,DynamicStepError,validate_dynamic_case)

def zero_kl():
    return KLSurfaceJets(*[np.zeros(3) for _ in FIELDS],
                         *[np.zeros(2) for _ in ELEC_FIELDS])

class TestKL(unittest.TestCase):
    def test_undeformed_cylinder_and_translation(self):
        j=zero_kl()
        for z in (-.18,0.,.18):
            p=kl_point(j,z_m=z,radius_m=1.5,thickness_m=.4)
            np.testing.assert_allclose(p.strain6,0,atol=3e-14)
            np.testing.assert_allclose(p.gradient6x3,0,atol=3e-14)
            self.assertAlmostEqual(p.determinant,1.,places=12)
        q=j.q.copy();q[0]=.04
        p=kl_point(j.replaced(q=q),z_m=.07,radius_m=1.5,thickness_m=.4)
        np.testing.assert_allclose(p.strain6,0,atol=3e-14)

    def test_g3d_flexo_switch_and_gauge(self):
        j=zero_kl()
        qx=j.qx.copy();qx[0]=.02
        qxx=j.qxx.copy();qxx[0]=.03
        psix=j.psix.copy();psix[0]=.2
        j=j.replaced(qx=qx,qxx=qxx,psix=psix)
        both=G3DMaterial(110.,.24,.2,.3,-.1,2.)
        off=G3DMaterial(110.,.24,.2,0.,0.,0.)
        e_on=thickness_enthalpy(j,both,radius_m=1.5,thickness_m=.4,n_gauss=5)
        e_off=thickness_enthalpy(j,off,radius_m=1.5,thickness_m=.4,n_gauss=5)
        self.assertNotAlmostEqual(e_on,e_off,places=8)
        self.assertGreater(e_off,0.)
        p=j.psi.copy();p[0]+=5
        self.assertAlmostEqual(thickness_enthalpy(j.replaced(psi=p),both,
            radius_m=1.5,thickness_m=.4,n_gauss=5),e_on,places=12)

    def test_directional_gateaux_coefficients(self):
        rng=np.random.default_rng(12)
        j=zero_kl()
        j=j.replaced(**{k:rng.normal(0,.006,3) for k in FIELDS},
                     **{k:rng.normal(0,.006,2) for k in ELEC_FIELDS})
        d=zero_kl()
        d=d.replaced(**{k:rng.normal(0,.003,3) for k in FIELDS},
                     **{k:rng.normal(0,.003,2) for k in ELEC_FIELDS})
        mat=G3DMaterial(120.,.23,.11,.01,-.02,.6)
        params=dict(radius_m=2.,thickness_m=.25,n_gauss=6)
        weak=thickness_weak_coefficients(j,mat,**params)
        directional=sum(float(weak.mechanical[:,k]@getattr(d,key))
            for k,key in enumerate(FIELDS))
        directional+=sum(float(weak.electrical[:,k]@getattr(d,key))
            for k,key in enumerate(ELEC_FIELDS))
        eps=1e-5
        def energy(sign):
            return thickness_enthalpy(j.replaced(**{
                k:getattr(j,k)+sign*eps*getattr(d,k)
                for k in (*FIELDS,*ELEC_FIELDS)
            }),mat,**params)
        np.testing.assert_allclose(directional,(energy(1)-energy(-1))/(2*eps),
                                   rtol=1e-5,atol=1e-7)

class TestNewmark(unittest.TestCase):
    def test_dynamic_config_mvi_selected(self):
        c=dict(calculation_kind="DYNAMIC",method="MVI",
               dynamics=dict(time_step_s=".01",n_steps="10",density_kg_m3=2500.))
        x=validate_dynamic_case(c)
        self.assertEqual(x["method"],"MVI")
        self.assertEqual(x["spatial_solver_status"],"NOT_CONNECTED_TO_CYLINDER_PDE")

    def test_requires_actual_mvi_callback(self):
        state=TransientState(0.,np.zeros(1),np.zeros(1),np.zeros(1))
        with self.assertRaises(DynamicInputError):
            list(integrate_newmark(state,NewmarkSettings(.1,2),method="MVI",
                spatial_solvers={},mass_action=lambda a:a,
                internal_force=lambda u,t:u,external_force=lambda t:np.zeros(1)))

    def test_full_residual_and_stepwise_mvi_callbacks(self):
        state=TransientState(0.,np.zeros(1),np.zeros(1),np.zeros(1))
        calls=[]
        # Exact solution of 1-DOF oscillator with Newmark effective residual.
        def mvi_solver(problem,predictor,index):
            calls.append(index)
            rhs=problem.external_force+problem.predicted_displacement/(problem.beta*problem.dt_s**2)
            u=rhs/(2.+1./(problem.beta*problem.dt_s**2))
            return SpatialStep(np.asarray(u),True,xy_cycles=3)
        data=list(integrate_newmark(state,NewmarkSettings(.1,5),method="MVI",
            spatial_solvers={"MVI":mvi_solver},
            mass_action=lambda a:a,internal_force=lambda u,t:2*u,
            external_force=lambda t:np.array([np.sin(t)])))
        self.assertEqual(calls,[1,2,3,4,5])
        self.assertTrue(all(item.xy_cycles==3 for item in data))
        self.assertTrue(all(item.full_residual_relative<1e-10 for item in data))

if __name__=="__main__":unittest.main()
