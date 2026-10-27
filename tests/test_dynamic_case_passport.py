"""Canonical physical CASE identity: no cross-physics temporal comparisons."""
import copy
import unittest
from cylinder_dynamic_case_passport import canonical_dynamic_physics, dynamic_physics_sha256

def case():
    return dict(calculation_kind="DYNAMIC",method="MVI",rank=2,
        mvi_grid=dict(nx=9,ns=9,operator_scheme="SBP21_PERIODIC_ODD"),
        material=dict(E0=120.,nu0=.25),theory="CUF_TE1",
        geometry=dict(L=2.,R=1.,h=.08),
        boundary_conditions=dict(variational=dict(mechanical=dict(left=dict(support="CLAMPED")))),
        load_definition=dict(q=5.),
        dynamics=dict(time_step_s=.02,n_steps=8,start_time_s=0.,
                      density_kg_m3=1300.,newmark_beta=.25,
                      newmark_gamma=.5,residual_tolerance_relative=1e-7,
                      time_law=dict(kind="HARMONIC",frequency_hz=1.)))

class PassportTests(unittest.TestCase):
    def test_integrator_and_solver_controls_not_physics(self):
        left=case();right=copy.deepcopy(left)
        right["dynamics"]["newmark_beta"]=.3025
        right["dynamics"]["newmark_gamma"]=.6
        right["dynamics"]["residual_tolerance_relative"]=1e-9
        right["rank"]=6
        right["reference_mvi"]={"max_sweeps":15}
        right["output_dir"]="/different"
        self.assertEqual(dynamic_physics_sha256(left),dynamic_physics_sha256(right))

    def test_physical_changes_are_always_detected(self):
        original=case()
        for change in (
            lambda c:c["material"].__setitem__("E0",140.),
            lambda c:c["geometry"].__setitem__("h",.06),
            lambda c:c["load_definition"].__setitem__("q",6.),
            lambda c:c["dynamics"].__setitem__("density_kg_m3",1100.),
            lambda c:c["dynamics"].__setitem__("time_step_s",.01),
            lambda c:c["dynamics"]["time_law"].__setitem__("frequency_hz",3.),
            lambda c:c["mvi_grid"].__setitem__("ns",11),
            lambda c:c["mvi_grid"].__setitem__("operator_scheme","LEGACY_REFERENCE"),
            lambda c:c["boundary_conditions"]["variational"]["mechanical"]["left"].__setitem__("support","FREE"),
        ):
            modified=copy.deepcopy(original);change(modified)
            self.assertNotEqual(dynamic_physics_sha256(original),dynamic_physics_sha256(modified))

    def test_invalid_input_fails_closed(self):
        for value in (float("nan"),float("inf"),0.,-1.,True):
            modified=case();modified["dynamics"]["density_kg_m3"]=value
            with self.assertRaisesRegex(ValueError,"DYNAMIC_PASSPORT"):
                dynamic_physics_sha256(modified)
        modified=case();modified["method"]="FDM"
        with self.assertRaisesRegex(ValueError,"REQUIRES_MVI"):
            canonical_dynamic_physics(modified)

    def test_non_mutating(self):
        source=case();snapshot=copy.deepcopy(source)
        self.assertEqual(len(dynamic_physics_sha256(source)),64)
        self.assertEqual(source,snapshot)

if __name__=="__main__":
    unittest.main()
