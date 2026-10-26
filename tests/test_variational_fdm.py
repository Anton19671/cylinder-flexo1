#!/usr/bin/env python3
"""Newton saddle contract and an actual coupled shell CASE solve."""
import copy
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

from cylinder_functional_assembly import surface_trapezoid
from cylinder_boundary_spatial import BoundarySpatialProblem
from cylinder_variational_fdm import StaticFDMSettings, solve_static_fdm
from cylinder_reference_case import prepare_reference_case, selected_backend
from cylinder_final_stage11a_dispatcher import (
    validate_final_case, run_final_case, run_mvi_rank_series, FinalCaseError,
)
from flexo_gradient_g3d import G3DMaterial
from final_capability_registry import case_capability, LIMITED, UNAVAILABLE

ROOT=Path(__file__).resolve().parent.parent


class AlgebraicFunctional:
    """Synthetic saddle for testing root finding independently of shell algebra."""
    theory="KL";n_mechanical=3;n_derivatives=10;electrical_on=True
    R=2.;h=.2
    def __init__(self,nonlinear=False,singular=False):
        self.ops=surface_trapezoid(4,4,1.2,2.)
        self.W=self.ops.area_weights
        self.material=G3DMaterial(100.,.25,.08,0.,0.,1.)
        self.nonlinear=nonlinear;self.singular=singular

    def evaluate(self,q,p,*,dead_load=None,free_charge=None,**kwargs):
        load=np.zeros_like(q) if dead_load is None else dead_load
        charge=np.zeros_like(p) if free_charge is None else free_charge
        stiffness=np.ones_like(q)
        if self.singular:stiffness[0]=0.
        value=.5*stiffness*q*q-load*q
        rm=stiffness*q-load
        if self.nonlinear:
            value+=q**4/4;rm+=q**3
        energy=float(np.sum(self.W*value)-np.sum(self.W*(.5*p*p+charge*p)))
        return SimpleNamespace(total_enthalpy_j=energy,
            mechanical_residual=self.W*rm,electrical_residual=self.W*(-p-charge))


class VariationalFDMSolverTest(unittest.TestCase):
    def problem(self,**kwargs):
        f=AlgebraicFunctional(**kwargs)
        spec={"mechanical":{"left":{"support":"CLAMPED"}},
              "electrical":{"inner":{"mode":"POTENTIAL","value":0.}}}
        return BoundarySpatialProblem(f,bc_spec=spec),np.zeros((3,4,4)),np.zeros((2,4,4))

    def test_indefinite_electric_equations_allow_increasing_enthalpy(self):
        problem,q,p=self.problem()
        charge=p.copy();charge[0]=1.
        result=solve_static_fdm(problem,q,p,free_charge=charge)
        self.assertTrue(result.converged,result.status)
        self.assertGreater(result.total_potential_j,0.)
        # Stationarity of -(.5 phi0^2+.5 phi1^2)-phi0 subject to
        # inner electrode phi0-phi1=0 gives phi0=phi1=-1/2.
        np.testing.assert_allclose(result.electrical,-.5,atol=1e-9)
        self.assertLess(result.constraint_defect_scaled,1e-12)
        self.assertLess(np.max(abs(result.full_residual-result.constraint_reaction)),1e-9)

    def test_singular_tangent_is_reported_without_regularization(self):
        problem,q,p=self.problem(singular=True)
        force=q.copy();force[0]=1.
        result=solve_static_fdm(problem,q,p,dead_load=force)
        self.assertFalse(result.converged)
        self.assertIn("SINGULAR_TANGENT",result.status)
        self.assertEqual(result.iterations,0)

    def test_iteration_exhaustion_is_not_success(self):
        problem,q,p=self.problem(nonlinear=True)
        force=q.copy();force[0]=1.
        result=solve_static_fdm(problem,q,p,dead_load=force,
                                settings=StaticFDMSettings(max_iterations=1))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"MAX_ITERATIONS")
        self.assertLess(result.constraint_defect_scaled,1e-12)

    def test_no_static_substitution_for_dynamics(self):
        problem,q,p=self.problem();problem.kind="DYNAMIC"
        with self.assertRaisesRegex(ValueError,"STATIC"):
            solve_static_fdm(problem,q,p)
        with self.assertRaises(ValueError):StaticFDMSettings(tangent_step_scaled=True)
        with self.assertRaises(ValueError):StaticFDMSettings(max_free_dofs=513)


class VariationalFDMCaseTest(unittest.TestCase):
    def sample(self):
        return json.loads((ROOT/"config/CASE_REFERENCE_CUF_G3D_STATIC.json").read_text())

    def test_all_theories_validate_explicit_reference_route(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
            with self.subTest(theory=theory):
                c=self.sample()
                if theory.startswith("CUF_"):
                    c["theory"]="CUF";c["cuf_model"]=theory[-3:]
                    c["quadrature"]["n_gauss"]=2*int(theory[-1])+1
                else:c["theory"]=theory
                if theory=="REDDY":c["quadrature"]["n_gauss"]=7
                p=prepare_reference_case(c)
                self.assertEqual(p.problem.functional.theory,theory)
        c=self.sample();c["micro"]["enable_flexoelectric"]=False
        c["boundary_conditions"]["variational"]["electrical"]={}
        self.assertIsNone(prepare_reference_case(c).electrical)
        c=self.sample();c["scale_model"]="MACRO"
        c["boundary_conditions"]["variational"]["electrical"]={}
        self.assertIsNone(prepare_reference_case(c).electrical)

    def test_backend_limits_and_invalid_settings_fail_closed(self):
        for change in ({"method":"MVI"},{"calculation_kind":"DYNAMIC"},
                       {"geometry_model":"LINEAR"},{"solver_backend":"TYPO"},
                       {"units":{"system":"N_MM_MPA"}},
                       {"fdm_grid":{"nx":4,"ns":4,"stencil":3}},
                       {"quadrature":{"type":"GAUSS","n_gauss":2}},
                       {"reference_newton":{"unknown":1}}):
            c=self.sample();c.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                prepare_reference_case(c)
        c=self.sample();c["load_definition"]["face"]="OUTER"
        with self.assertRaisesRegex(ValueError,"MIDDLE"):prepare_reference_case(c)
        self.assertEqual(selected_backend({}),"CLASSICAL")

    def test_capability_distinguishes_reference_and_default_micro(self):
        c=self.sample()
        capability=case_capability(c)
        self.assertEqual(capability.status,LIMITED)
        self.assertTrue(capability.can_run)
        del c["solver_backend"]
        self.assertEqual(case_capability(c).status,UNAVAILABLE)

    def test_validation_and_rank_series_do_not_create_results(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);path=root/"CASE.json"
            c=self.sample();c["output_dir"]="run"
            path.write_text(json.dumps(c))
            self.assertTrue(validate_final_case(path)["valid"])
            self.assertFalse((root/"run").exists())
            with self.assertRaisesRegex(FinalCaseError,"MVI rank"):
                run_mvi_rank_series(path)
            self.assertFalse((root/"run").exists())
            c["solver_backend"]="CLASSICAL";path.write_text(json.dumps(c))
            with self.assertRaisesRegex(FinalCaseError,"VARIATIONAL_REFERENCE"):
                run_final_case(path)

    def test_actual_coupled_shell_case_solves_and_saves_fields(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);path=root/"CASE.json"
            c=self.sample();c["output_dir"]="run"
            path.write_text(json.dumps(c))
            with contextlib.redirect_stdout(io.StringIO()):summary=run_final_case(path)
            self.assertTrue(summary["converged"],summary)
            self.assertEqual(summary["accepted_steps"],1)
            out=root/"run"
            saved=json.loads((out/"FINAL_RUN_SUMMARY.json").read_text())
            self.assertEqual(saved["steps"],summary["steps"])
            self.assertLess(saved["steps"][0]["residual_relative_to_initial"],1e-7)
            with np.load(out/summary["steps"][0]["state_file"]) as state:
                q,p=state["mechanical"],state["electrical"]
                self.assertGreater(np.max(abs(q)),1e-6)
                self.assertGreater(np.max(abs(p)),1e-10)
                np.testing.assert_allclose(q[:,[0,-1],:],0.,atol=1e-14)
                np.testing.assert_allclose(p[0]-p[1],0.,atol=1e-14)
                np.testing.assert_allclose(state["full_residual"],state["constraint_reaction"],atol=1e-8)
                # The residual includes nonzero support reactions.
                self.assertGreater(np.linalg.norm(state["constraint_reaction"]),1e-4)
            with self.assertRaisesRegex(ValueError,"OUTPUT_DIR_NOT_EMPTY"):
                run_final_case(path)


if __name__=="__main__":unittest.main()
