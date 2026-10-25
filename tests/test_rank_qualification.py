"""Ranks qualify only through full equilibrium and physical-field comparison."""
import unittest
from unittest.mock import patch
from cylinder_rank_study import assess_rank_series


TOL={name:{"abs":1.e-5,"rel":1.e-3}
     for name in ("radial_displacement","mechanical","velocity","acceleration","electrical")}


def fake(rank, *, residual=1.e-10, error=0.):
    errors={name:{"max_abs_difference":error,"fdm_peak_abs":1.}
            for name in TOL}
    return {"mvi_rank":rank,"steps":[{
        "field_errors":errors,
        "mvi_admissible_residual_scaled":residual,
        "fdm_admissible_residual_scaled":1.e-11,
        "mvi_constraint_defect_scaled":1.e-12,
        "fdm_constraint_defect_scaled":1.e-12}]}


class TestRankQualification(unittest.TestCase):
    def run_study(self,ranks,**kwargs):
        return assess_rank_series("FDM",{r:f"rank{r}" for r in ranks},
            field_tolerances=TOL,max_full_residual_scaled=1.e-7,
            max_boundary_defect_scaled=1.e-9,**kwargs)

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_minimum_requires_adjacent_rank(self,compare):
        compare.side_effect=lambda f,_:fake(int(f[4:]))
        out=self.run_study((1,2,3))
        self.assertEqual(out["candidate_rank"],1)
        self.assertFalse(out["certified_reference"])

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_full_residual_rejects_rank(self,compare):
        compare.side_effect=lambda f,_:fake(int(f[4:]),residual=1.e-3)
        self.assertIsNone(self.run_study((1,2))["candidate_rank"])

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_field_mismatch_rejects_rank(self,compare):
        compare.side_effect=lambda f,_:fake(int(f[4:]),error=.02)
        self.assertIsNone(self.run_study((1,2))["candidate_rank"])

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_no_skipping_lower_rank(self,compare):
        compare.side_effect=lambda f,_:fake(int(f[4:]))
        self.assertIsNone(self.run_study((3,4))["candidate_rank"])

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_missing_electrical_rejects_rank(self,compare):
        def wrong(f,_):
            x=fake(int(f[4:]))
            x["steps"][0]["field_errors"].pop("electrical")
            return x
        compare.side_effect=wrong
        self.assertIsNone(self.run_study((1,2))["candidate_rank"])

    @patch("cylinder_rank_study.compare_dynamic_runs")
    def test_incomplete_case_is_rejected(self,compare):
        compare.side_effect=ValueError("missing accepted step")
        data=self.run_study((1,2))
        self.assertEqual(data["ranks"][0]["status"],"RUN_INCOMPLETE_OR_INVALID")
        self.assertIsNone(data["candidate_rank"])

    def test_invalid_tolerances(self):
        with self.assertRaises(ValueError):
            self.run_study((1,),field_tolerances={})
        with self.assertRaises(ValueError):
            assess_rank_series("FDM",{},field_tolerances=TOL,
                max_full_residual_scaled=float("nan"),
                max_boundary_defect_scaled=1.e-9)

if __name__=="__main__":unittest.main()
