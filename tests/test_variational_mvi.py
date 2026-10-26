#!/usr/bin/env python3
"""MVI activation, rank adequacy, saddle equations, constraints and CASE I/O."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from test_variational_fdm import AlgebraicFunctional
from cylinder_functional_assembly import surface_trapezoid
from cylinder_boundary_spatial import BoundarySpatialProblem, affine_trace_space
from cylinder_variational_fdm import StaticFDMSettings, solve_static_fdm
from cylinder_variational_mvi import (
    StaticMVISettings, solve_static_mvi, reconstruct_profiles, _normalize_partner,
)
from cylinder_reference_case import prepare_reference_case, run_reference_mvi_series, run_reference_case
from cylinder_final_stage11a_dispatcher import validate_final_case, run_final_case
from final_capability_registry import case_capability, LIMITED

ROOT=Path(__file__).resolve().parent.parent


class VariationalMVITest(unittest.TestCase):
    def problem(self,nx=4,ns=4,**kwargs):
        f=AlgebraicFunctional(**kwargs)
        f.ops=surface_trapezoid(nx,ns,1.2,2.);f.W=f.ops.area_weights
        spec={"mechanical":{"left":{"support":"CLAMPED"}},
              "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
        return BoundarySpatialProblem(f,bc_spec=spec),np.zeros((3,nx,ns)),np.zeros((2,nx,ns))

    def assert_representation(self,r):
        u=r.scales*(reconstruct_profiles(r.x_profiles,r.y_profiles))
        actual=r.mechanical if r.electrical is None else np.concatenate((r.mechanical,r.electrical))
        np.testing.assert_allclose(u,actual,atol=1e-13)

    def test_all_eight_ranks_activate_from_zero_and_solve_electrical_saddle(self):
        for rank in range(1,9):
            with self.subTest(rank=rank):
                problem,q,p=self.problem(8,8)
                force=q.copy();force[0]=1.
                charge=p.copy();charge[0]=1.
                r=solve_static_mvi(problem,q,p,dead_load=force,free_charge=charge,
                    mvi_settings=StaticMVISettings(rank=rank))
                self.assertTrue(r.converged,r.status)
                np.testing.assert_allclose(r.mechanical[0,1:],1.,atol=1e-8)
                np.testing.assert_allclose(r.electrical,-.5,atol=1e-8)
                self.assertLess(r.constraint_defect_scaled,1e-12)
                self.assert_representation(r)
                self.assertEqual({a['direction'] for a in r.history if a['event']=='halfstep'},{'X','Y'})
                self.assertTrue(any('tangent_condition' in a and 'step_factor' in a for a in r.history))

    def test_rank_one_projected_solution_does_not_claim_full_equilibrium(self):
        problem,q,p=self.problem()
        force=q.copy()
        force[0]=np.outer([0,1,1,0],[2,2,2,2])+np.outer([0,1,-1,0],[1,-1,1,-1])
        r=solve_static_mvi(problem,q,p,dead_load=force)
        self.assertFalse(r.converged)
        self.assertTrue(r.projected_equilibrium_satisfied)
        self.assertIn('PROJECTED_ONLY',r.status)
        self.assertGreater(r.admissible_residual_scaled,1e-5)
        r2=solve_static_mvi(problem,q,p,dead_load=force,mvi_settings=StaticMVISettings(rank=2))
        self.assertTrue(r2.converged,r2.status)
        np.testing.assert_allclose(r2.mechanical,force,atol=1e-8)
        self.assert_representation(r2)

    def test_nonzero_boundary_lift_and_load_continuation(self):
        problem,q,p=self.problem()
        problem.spec['mechanical']['left']['prescribed']=[.2,0,0]
        problem.spec['electrical']['inner']['value']=.3
        force=q.copy();force[1]=.4
        first=solve_static_mvi(problem,q,p,dead_load=force)
        self.assertTrue(first.converged,first.status)
        second=solve_static_mvi(problem,first.mechanical,first.electrical,dead_load=2*force)
        fdm=solve_static_fdm(problem,q,p,dead_load=2*force)
        self.assertTrue(second.converged,second.status)
        np.testing.assert_allclose(second.mechanical,fdm.mechanical,atol=1e-8)
        np.testing.assert_allclose(second.electrical,fdm.electrical,atol=1e-8)
        self.assert_representation(second)

    def test_exact_qr_changes_only_coordinates_even_for_inactive_terms(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(3,4,8));y=rng.normal(size=(3,4,9))
        x[0]=0;x[1,2:]=0;y[2,1]=y[2,0]
        original=reconstruct_profiles(x,y)
        for axis in ('X','Y'):
            xx,yy=_normalize_partner(x,y,axis)
            np.testing.assert_allclose(reconstruct_profiles(xx,yy),original,atol=1e-14)
            frozen=yy if axis=='X' else xx
            for a in frozen:np.testing.assert_allclose(a@a.T,np.eye(4),atol=1e-14)

    def test_roundoff_zero_trace_is_tolerated_but_unreachable_data_rejected(self):
        c=np.array([[1.,0.,0.]])
        directions=np.array([[0.],[1.],[0.]])
        a=affine_trace_space(c,np.zeros(1),np.array([1e-16,1.,0.]),directions=directions)
        self.assertLess(abs(a.defect(a.origin)[0]),1e-14)
        with self.assertRaisesRegex(ValueError,'unreachable'):
            affine_trace_space(c,np.zeros(1),np.array([1e-7,1.,0.]),directions=directions)

    def test_singular_halfstep_fails_without_full_grid_fallback(self):
        problem,q,p=self.problem(singular=True)
        force=q.copy();force[0]=1
        with patch('cylinder_variational_fdm.solve_static_fdm',side_effect=AssertionError('FDM fallback')):
            r=solve_static_mvi(problem,q,p,dead_load=force)
        self.assertFalse(r.converged)
        self.assertIn('SINGULAR_TANGENT',r.status)

    def test_mechanics_only_does_not_create_electric_profiles(self):
        problem,q,_=self.problem()
        problem.functional.electrical_on=False
        problem.spec['electrical']={}
        def evaluate(q,p,*,dead_load=None,**kwargs):
            self.assertIsNone(p)
            w=problem.functional.W
            return SimpleNamespace(total_enthalpy_j=float(np.sum(w*(q*q/2-dead_load*q))),
                                   mechanical_residual=w*(q-dead_load),electrical_residual=None)
        problem.functional.evaluate=evaluate
        load=q.copy();load[0]=1.
        r=solve_static_mvi(problem,q,dead_load=load)
        self.assertTrue(r.converged,r.status)
        self.assertIsNone(r.electrical)
        self.assertEqual(r.x_profiles.shape[0],3)
        self.assert_representation(r)

    def test_iteration_limit_and_dynamic_request_are_not_success(self):
        problem,q,p=self.problem(nonlinear=True)
        force=q.copy();force[0]=10
        r=solve_static_mvi(problem,q,p,dead_load=force,settings=StaticFDMSettings(max_iterations=1))
        self.assertFalse(r.converged)
        self.assertIn('MAX_ITERATIONS',r.status)
        problem.kind='DYNAMIC'
        with self.assertRaisesRegex(ValueError,'STATIC'):solve_static_mvi(problem,q,p)
        for rank in (0,9,True,1.5):
            with self.assertRaises(ValueError):StaticMVISettings(rank=rank)


class VariationalMVICaseTest(unittest.TestCase):
    def sample(self):
        return json.loads((ROOT/'config/CASE_REFERENCE_CUF_G3D_MVI_STATIC.json').read_text())

    def test_all_theories_accept_static_mvi_with_flexo_on_and_off(self):
        for theory in ('KL','FSDT','REDDY','CUF_TE1','CUF_TE2','CUF_TE3','CUF_TE4'):
            for on in (True,False):
                with self.subTest(theory=theory,on=on):
                    c=self.sample();c['micro']['enable_flexoelectric']=on
                    if not on:c['boundary_conditions']['variational']['electrical']={}
                    if theory.startswith('CUF'):
                        c['cuf_model']=theory[-3:];c['quadrature']['n_gauss']=2*int(theory[-1])+1
                    else:c['theory']=theory
                    if theory=='REDDY':c['quadrature']['n_gauss']=7
                    prepared=prepare_reference_case(c)
                    self.assertEqual(prepared.method,'MVI')
                    self.assertEqual(prepared.problem.functional.theory,theory)
                    self.assertEqual(prepared.electrical is not None,on)
        self.assertEqual(case_capability(self.sample()).status,LIMITED)
        self.assertEqual(case_capability(self.sample()).code,'REFERENCE_STATIC_MVI')

    def test_case_rejects_incompatible_rank_and_ambiguous_settings(self):
        for change in ({'rank':5},{'rank':True},{'rank':1.2},
                       {'reference_mvi':{'rank':2}}, {'reference_mvi':{'max_sweeps':0}},
                       {'fdm_grid':{'nx':4,'ns':4}},{'calculation_kind':'DYNAMIC'}):
            c=self.sample();c.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):prepare_reference_case(c)

    def test_rank_series_preflight_and_continuation_after_failure(self):
        with TemporaryDirectory() as d:
            c=self.sample();c['output_dir']=str(Path(d)/'results')
            with self.assertRaisesRegex(ValueError,'grid dimensions'):
                run_reference_mvi_series(c,Path(d)/'case.json',range(1,9))
            self.assertFalse(Path(c['output_dir']).exists())
            outcomes=[{'converged':False,'status':'PROJECTED_ONLY'},{'converged':True,'status':'CONVERGED'}]
            with patch('cylinder_reference_case.run_reference_case',side_effect=outcomes) as run:
                summary=run_reference_mvi_series(c,Path(d)/'case.json',[1,2])
            self.assertFalse(summary['all_converged'])
            self.assertEqual(run.call_count,2)
            self.assertTrue((Path(summary['output_dir'])/'MVI_1_8_SUMMARY.json').is_file())

    def test_projected_only_case_saves_failure_and_stops_load_series(self):
        with TemporaryDirectory() as d:
            c=self.sample();c['theory']='KL';c['output_dir']=str(Path(d)/'results')
            prepared=prepare_reference_case(c)
            problem,q,p=VariationalMVITest().problem()
            load=q.copy()
            load[0]=np.outer([0,1,1,0],[2,2,2,2])+np.outer([0,1,-1,0],[1,-1,1,-1])
            prepared=replace(prepared,problem=problem,mechanical=q,electrical=p,
                             dead_load=load,factors=(1.,2.))
            with patch('cylinder_reference_case.prepare_reference_case',return_value=prepared), contextlib.redirect_stdout(io.StringIO()):
                result=run_reference_case(c,Path(d)/'CASE.json')
            self.assertFalse(result['converged'])
            self.assertEqual(result['accepted_steps'],0)
            self.assertEqual(len(result['steps']),1)
            self.assertTrue(result['steps'][0]['projected_equilibrium_satisfied'])
            self.assertFalse(result['steps'][0]['full_equilibrium_satisfied'])
            self.assertTrue((Path(c['output_dir'])/'REFERENCE_FAILED_ITERATE.npz').exists())
            self.assertFalse((Path(c['output_dir'])/'REFERENCE_STEP_0001.npz').exists())

    def test_actual_coupled_shell_mvi_matches_fdm_and_saves_profiles(self):
        with TemporaryDirectory() as d:
            root=Path(d);c=self.sample();c['output_dir']=str(root/'mvi')
            path=root/'CASE.json';path.write_text(json.dumps(c))
            self.assertEqual(validate_final_case(path)['method'],'MVI')
            self.assertFalse((root/'mvi').exists())
            with contextlib.redirect_stdout(io.StringIO()):summary=run_final_case(path)
            self.assertTrue(summary['converged'],summary)
            self.assertTrue(summary['steps'][0]['full_equilibrium_satisfied'])
            fdm=copy.deepcopy(c);fdm['method']='FDM';fdm['fdm_grid']=fdm.pop('mvi_grid')
            del fdm['rank'];del fdm['reference_mvi'];fdm['output_dir']=str(root/'fdm')
            path.write_text(json.dumps(fdm))
            with contextlib.redirect_stdout(io.StringIO()):baseline=run_final_case(path)
            self.assertTrue(baseline['converged'])
            with np.load(root/'mvi'/summary['steps'][0]['state_file']) as a, np.load(root/'fdm'/baseline['steps'][0]['state_file']) as b:
                np.testing.assert_allclose(a['mechanical'],b['mechanical'],rtol=1e-6,atol=1e-11)
                np.testing.assert_allclose(a['electrical'],b['electrical'],rtol=1e-5,atol=1e-13)
                fields=a['state_scales']*(reconstruct_profiles(a['x_profiles'],a['y_profiles']))
                np.testing.assert_allclose(fields,np.concatenate((a['mechanical'],a['electrical'])),atol=1e-15)
                np.testing.assert_allclose(a['full_residual'],a['constraint_reaction'],atol=1e-8)


if __name__=='__main__':unittest.main()
