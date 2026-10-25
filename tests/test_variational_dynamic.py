#!/usr/bin/env python3
"""Effective Newmark residual, MVI/FDM parity and dynamic CASE persistence."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_functional_assembly import surface_trapezoid
from cylinder_variational_dynamic import solve_dynamic_fdm,solve_dynamic_mvi
from cylinder_variational_fdm import StaticFDMSettings
from cylinder_variational_mvi import StaticMVISettings,reconstruct_profiles
from cylinder_reference_case import prepare_reference_case
from cylinder_final_stage11a_dispatcher import validate_final_case,run_final_case
from flexo_gradient_g3d import G3DMaterial

ROOT=Path(__file__).resolve().parent.parent


class DynamicFunctional:
    theory="CUF_TE1";n_mechanical=3;n_derivatives=10;electrical_on=True
    R=2.;h=.2
    def __init__(self,nx=4,ns=4,nonlinear=False):
        self.ops=surface_trapezoid(nx,ns,1.2,2.);self.W=self.ops.area_weights
        self.material=G3DMaterial(100.,.25,.08,.01,-.02,1.5)
        self.nonlinear=nonlinear
    def evaluate(self,q,p,*,dead_load=None,free_charge=None,**kwargs):
        load=np.zeros_like(q) if dead_load is None else dead_load
        charge=np.zeros_like(p) if free_charge is None else free_charge
        return SimpleNamespace(total_enthalpy_j=float(np.sum(self.W*(q*q+q**4/4-load*q))),
            mechanical_residual=self.W*(2*q+(q**3 if self.nonlinear else 0)-load),
            electrical_residual=self.W*(-p-charge))
    def mass_action(self,acceleration,*,density_kg_m3):
        return self.W*float(density_kg_m3)*acceleration


class DynamicSpatialTest(unittest.TestCase):
    def problem(self):
        f=DynamicFunctional();spec={"mechanical":{"left":{"support":"CLAMPED"}},
            "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
        return BoundarySpatialProblem(f,bc_spec=spec,calculation_kind="DYNAMIC"),np.zeros((3,4,4)),np.zeros((2,4,4))

    def kwargs(self,q):
        load=np.zeros_like(q);load[0]=1.
        charge=np.ones((2,*q.shape[1:]))
        return dict(q_predictor=q,velocity_predictor=np.zeros_like(q),time_s=.1,
            dt_s=.1,beta=.25,gamma=.5,density_kg_m3=1.,dead_load=load,
            free_charge=charge,settings=StaticFDMSettings())

    def test_mvi_and_fdm_solve_same_effective_coupled_equations(self):
        problem,q,p=self.problem();kwargs=self.kwargs(q)
        fdm=solve_dynamic_fdm(problem,q,p,**kwargs)
        mvi=solve_dynamic_mvi(problem,q,p,**kwargs,mvi_settings=StaticMVISettings(rank=1))
        self.assertTrue(fdm.converged,fdm.status);self.assertTrue(mvi.converged,mvi.status)
        np.testing.assert_allclose(mvi.mechanical,fdm.mechanical,atol=2e-12)
        np.testing.assert_allclose(mvi.electrical,fdm.electrical,atol=2e-10)
        np.testing.assert_allclose(mvi.velocity,2/(.1)*mvi.mechanical,atol=2e-12)
        np.testing.assert_allclose(mvi.acceleration,4/(.1**2)*mvi.mechanical,atol=2e-12)
        self.assertEqual(mvi.sweeps,1)
        self.assertLess(mvi.admissible_residual_scaled,1e-10)
        represented=mvi.scales*(mvi.lift_scaled+reconstruct_profiles(mvi.x_profiles,mvi.y_profiles))
        np.testing.assert_allclose(represented,np.concatenate((mvi.mechanical,mvi.electrical)),atol=1e-13)

    def test_electric_unknown_has_no_velocity_or_inertia(self):
        problem,q,p=self.problem();r=solve_dynamic_fdm(problem,q,p,**self.kwargs(q))
        self.assertEqual(r.velocity.shape,q.shape)
        self.assertEqual(r.acceleration.shape,q.shape)
        self.assertNotEqual(r.electrical.shape,q.shape)
        # Electrical solution is the quasi-static saddle value at this time.
        np.testing.assert_allclose(r.electrical,-1.,atol=2e-10)

    def test_nonconvergence_does_not_become_an_accepted_step(self):
        problem,q,p=self.problem();problem.functional.nonlinear=True
        kwargs=self.kwargs(q);kwargs["settings"]=StaticFDMSettings(max_iterations=1)
        kwargs["dead_load"]*=1000
        result=solve_dynamic_fdm(problem,q,p,**kwargs)
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"MAX_ITERATIONS")


class DynamicCaseTest(unittest.TestCase):
    def sample(self):
        return json.loads((ROOT/'config/CASE_REFERENCE_CUF_G3D_MVI_DYNAMIC.json').read_text())

    def test_contract_rejects_inconsistent_initial_or_unsupported_inertia(self):
        c=self.sample();c['theory']='KL'
        with self.assertRaisesRegex(ValueError,'director inertia'):prepare_reference_case(c)
        c=self.sample();c['dynamics']['time_law']={'kind':'CONSTANT','amplitude':1.}
        with self.assertRaisesRegex(ValueError,'not in equilibrium'):prepare_reference_case(c)
        c=self.sample();c['boundary_conditions']['variational']['mechanical']['left']['time_law']={
            'kind':'RAMP','amplitude':1.,'duration_s':1.}
        with self.assertRaisesRegex(ValueError,'C2 SMOOTHSTEP'):prepare_reference_case(c)
        c=self.sample();c['load_definition']['load_factors']=[1.]
        with self.assertRaisesRegex(ValueError,'time_law'):prepare_reference_case(c)

    def test_smoothstep_prescribed_support_accepts_kinematic_boundary(self):
        c=self.sample()
        c["boundary_conditions"]["variational"]["mechanical"]["left"].update(
            prescribed=[0.,0.,0.,0.,0.,0.],
            time_law={"kind":"SMOOTHSTEP","amplitude":1.,
                      "start_time_s":0.,"duration_s":.24})
        prepared=prepare_reference_case(c)
        self.assertEqual(prepared.calculation_kind,"DYNAMIC")

    def test_moving_support_requires_zero_initial_target(self):
        c=self.sample()
        c["boundary_conditions"]["variational"]["mechanical"]["left"].update(
            prescribed=[.002,0.,0.,0.,0.,0.],
            time_law={"kind":"SMOOTHSTEP","amplitude":1.,
                      "start_time_s":-.24,"duration_s":.24})
        with self.assertRaisesRegex(ValueError,"not in equilibrium|initial"):
            prepare_reference_case(c)

    def test_time_dependent_electrode_is_assembled_by_newmark(self):
        c=self.sample()
        c["boundary_conditions"]["variational"]["electrical"]["inner"].update(
            value=2.,time_law={
                "kind":"RAMP","amplitude":1.,"start_time_s":0.,
                "duration_s":0.02})
        prepared=prepare_reference_case(c)
        self.assertEqual(prepared.calculation_kind,"DYNAMIC")
        at_start=prepared.problem.assemble(
            prepared.mechanical,prepared.electrical,time_s=0.,
            dead_load=np.zeros_like(prepared.mechanical),
            acceleration=np.zeros_like(prepared.mechanical),
            density_kg_m3=prepared.dynamics.density_kg_m3)
        at_end=prepared.problem.assemble(
            prepared.mechanical,prepared.electrical,time_s=.02,
            dead_load=np.zeros_like(prepared.mechanical),
            acceleration=np.zeros_like(prepared.mechanical),
            density_kg_m3=prepared.dynamics.density_kg_m3)
        self.assertEqual(at_start.space.matrix.shape,at_end.space.matrix.shape)
        self.assertAlmostEqual(float(np.max(abs(at_start.space.target))),0.)
        self.assertAlmostEqual(float(np.max(abs(at_end.space.target))),2.)

    def test_boundary_time_law_rejects_undersampled_frequency(self):
        c=self.sample()
        c["boundary_conditions"]["variational"]["electrical"]["inner"].update(
            value=1.,time_law={"kind":"HARMONIC","amplitude":1.,
                "frequency_hz":100.,"start_time_s":0.})
        with self.assertRaisesRegex(ValueError,"undersamples"):
            prepare_reference_case(c)

    def test_actual_coupled_mvi_newmark_steps_save_complete_state(self):
        with TemporaryDirectory() as d:
            root=Path(d);path=root/'CASE.json';c=self.sample()
            c['dynamics']['n_steps']=2;c['output_dir']=str(root/'run')
            path.write_text(json.dumps(c))
            valid=validate_final_case(path)
            self.assertEqual(valid['calculation_kind'],'DYNAMIC')
            with contextlib.redirect_stdout(io.StringIO()):summary=run_final_case(path)
            self.assertTrue(summary['converged'],summary)
            self.assertEqual(summary['accepted_steps'],2)
            self.assertFalse(summary['electric_inertia'])
            self.assertTrue((root/'run'/'REFERENCE_DYNAMIC_INITIAL.npz').exists())
            times=[]
            for step in summary['steps']:
                self.assertTrue(step['full_equilibrium_satisfied'])
                self.assertLess(step['residual_relative_to_initial'],1e-7)
                with np.load(root/'run'/step['state_file']) as state:
                    times.append(float(state['time_s']))
                    self.assertEqual(state['velocity'].shape,state['mechanical'].shape)
                    self.assertEqual(state['acceleration'].shape,state['mechanical'].shape)
                    self.assertGreater(np.max(abs(state['electrical'])),0.)
                    represented=state['state_scales']*(state['lift_scaled']+
                        reconstruct_profiles(state['x_profiles'],state['y_profiles']))
                    np.testing.assert_allclose(represented,
                        np.concatenate((state['mechanical'],state['electrical'])),atol=1e-15)
            np.testing.assert_allclose(times,[.01,.02])
            with self.assertRaisesRegex(ValueError,'OUTPUT_DIR_NOT_EMPTY'):run_final_case(path)


if __name__=='__main__':unittest.main()
