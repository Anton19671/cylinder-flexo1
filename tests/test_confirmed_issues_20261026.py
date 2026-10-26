"""Regressions for confirmed isolated G3D and legacy-interface issues."""
import unittest
import numpy as np
from unittest.mock import patch
from cylinder_functional_assembly import surface_trapezoid
from cylinder_variational_loads import project_radial_load
from final_units import load_axis
from cylinder_rank_study import assess_rank_series
from final_step_predictor import predict_state
from cylinder_kl_stage8b2_profiles import ProfileStateKL


class MockFunctional:
    R=2.
    h=.2
    theory="CUF_TE2"
    n_mechanical=9

    def __init__(self):
        self.ops=surface_trapezoid(8,13,1.2,self.R)


class ConfirmedIssuesTests(unittest.TestCase):
    def test_line_does_not_require_patch_bounds_on_rectangular_grid(self):
        f=MockFunctional()
        d=dict(type="LINE_RADIAL_GENERATOR",face="OUTER",
               kinematics="DEAD_REFERENCE",p_line=9.,
               c0=.3,band_width=2.*np.pi*f.R/3.,
               x1=0.,x2=1.2)
        result=project_radial_load(f,d)
        self.assertEqual(result.face_pressure_pa.shape,(8,13))
        self.assertEqual(result.generalized.shape,(9,8,13))
        self.assertAlmostEqual(result.metadata["face_force_radial_n"],10.8,places=9)
        with self.assertRaisesRegex(ValueError,"patch-window"):
            project_radial_load(f,{**d,"c1":0.})

    def test_point_force_axis_alias(self):
        for key in ("P","force"):
            a,label,kind=load_axis([0.,.5,1.],{"type":"POINT_RADIAL_FORCE",key:8.},"SI_N_M_PA")
            self.assertEqual(a,[0.,4.,8.])
            self.assertEqual(kind,"P")

    def test_predictor_never_mutates_cuf_or_kl_history(self):
        from cylinder_cuf_stage7b_general_mvi_profiles import ProfileStateCUF
        for cls,shape in ((ProfileStateKL,(3,2,5)),
                          (ProfileStateCUF,(3,2,2,5))):
            last=cls(np.ones(shape),np.ones((*shape[:-1],7)))
            prev=cls(np.zeros(shape),np.zeros((*shape[:-1],7)))
            before_x=last.X.copy();before_y=last.Y.copy()
            predicted=predict_state(prev,last,2.,1.,0.,mode="LINEAR")
            np.testing.assert_array_equal(last.X,before_x)
            np.testing.assert_array_equal(last.Y,before_y)
            self.assertFalse(np.shares_memory(last.X,predicted.X))
            self.assertFalse(np.shares_memory(last.Y,predicted.Y))

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_rank_eight_has_real_rank_seven_neighbor(self,compare):
        fields=("radial_displacement","mechanical","velocity","acceleration")
        def fake(folder,fdm):
            rank=int(folder[4:])
            return dict(mvi_rank=rank,steps=[dict(
                field_errors={key:dict(max_abs_difference=0.,fdm_peak_abs=1.)
                              for key in fields},
                mvi_admissible_residual_scaled=1.e-10,
                fdm_admissible_residual_scaled=1.e-11,
                mvi_constraint_defect_scaled=1.e-12,
                fdm_constraint_defect_scaled=1.e-12)])
        compare.side_effect=fake
        output=assess_rank_series("FDM",{r:f"rank{r}" for r in range(1,9)},
            field_tolerances={k:{"abs":1e-5,"rel":1e-3} for k in fields},
            max_full_residual_scaled=1e-7,max_boundary_defect_scaled=1e-9)
        self.assertTrue(output["ranks"][7]["qualified"])
        self.assertEqual(output["candidate_rank"],1)


if __name__=="__main__":
    unittest.main()
