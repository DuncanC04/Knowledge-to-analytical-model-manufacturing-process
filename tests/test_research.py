import numpy as np
import pandas as pd
import unittest

from src.research import (
    committee_disagreement,
    development_threshold_grid,
    interval_metrics,
    pareto_front,
    single_axis_split,
    validate_expression,
    variance_decomposition,
)
from src.modeling import adjusted_r2, information_criteria, select_committee


class ResearchTests(unittest.TestCase):
    def test_single_axis_split_keeps_boundary_levels_and_is_reproducible(self):
        frame = pd.DataFrame({"axis": np.repeat([1, 2, 3, 4], 3), "group": np.arange(12) // 3, "y": np.arange(12)})
        left = single_axis_split(frame, "axis", "group", random_state=7)
        right = single_axis_split(frame, "axis", "group", random_state=7)
        self.assertEqual(left, right)
        self.assertEqual(set(frame.loc[list(left.test_indices), "axis"]), {4})
        self.assertFalse(set(left.train_indices) & set(left.validation_indices))
        self.assertFalse(set(left.train_indices) & set(left.test_indices))


    def test_disagreement_flags_invalid_positive_scale_metrics(self):
        metrics = committee_disagreement(np.array([[1.0, -1.0], [3.0, -2.0]]), tau=1.0)
        self.assertTrue(np.isfinite(metrics.loc[0, "paper_cv"]))
        self.assertTrue(np.isnan(metrics.loc[1, "paper_cv"]))
        self.assertTrue(np.isnan(metrics.loc[1, "paper_ibmd"]))


    def test_variance_decomposition_identity(self):
        result = variance_decomposition(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[.25, .5], [.25, .5]]))
        np.testing.assert_allclose(result["total"], result["within"] + result["between"])

    def test_interval_metrics_penalizes_missed_outcomes(self):
        result = interval_metrics([0.0, 2.0], [-1.0, -1.0], [1.0, 1.0], nominal=0.9)
        self.assertEqual(result["coverage"], 0.5)
        self.assertGreater(result["interval_score"], result["mean_width"])

    def test_safe_expression_rejects_attribute_execution(self):
        validate_expression("a + 2 * x", {"a", "x"})
        with self.assertRaises(ValueError):
            validate_expression("__import__('os').system('whoami')", {"x"})

    def test_pareto_front_keeps_non_dominated_candidates(self):
        np.testing.assert_array_equal(pareto_front([1.0, 2.0, 1.5], [3.0, 1.0, 2.0]), [True, True, True])

    def test_threshold_grid_is_labeled_statistical(self):
        result = development_threshold_grid([0.0, 1.0, 2.0], [0.5])
        self.assertEqual(result.loc[0, "interpretation"], "statistical_threshold_not_engineering_tolerance")

    def test_bic_penalizes_complexity_more_than_aic_for_large_n(self):
        criteria = information_criteria(np.ones(100), n_parameters=4)
        self.assertGreater(criteria["bic"], criteria["aic"])
        self.assertGreater(criteria["aicc"], criteria["aic"])

    def test_cv_and_ibmd_match_positive_prediction_definitions(self):
        metrics = committee_disagreement(np.array([[1.0, 2.0], [3.0, 4.0]]), tau=1.0)
        self.assertAlmostEqual(metrics.loc[0, "paper_cv"], 0.5)
        self.assertAlmostEqual(metrics.loc[1, "paper_cv"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics.loc[0, "paper_ibmd"], np.log1p(2.0 / 3.0))
        self.assertTrue(np.isfinite(metrics.loc[1, "paper_ibmd"]))

    def test_committee_selection_removes_duplicate_validation_predictions(self):
        base = {"validation_mse": 1.0, "expression_complexity": 2}
        candidates = [
            {**base, "candidate_id": "a", "bic": 1.0, "aic": 1.0, "aicc": 1.0, "validation_prediction_signature": "same"},
            {**base, "candidate_id": "b", "bic": 2.0, "aic": 2.0, "aicc": 2.0, "validation_prediction_signature": "same"},
            {**base, "candidate_id": "c", "bic": 3.0, "aic": 3.0, "aicc": 3.0, "validation_prediction_signature": "different"},
        ]
        selected = select_committee(candidates, committee_size=2)
        self.assertEqual([item["candidate_id"] for item in selected], ["a", "c"])

    def test_adjusted_r2_selection_prefers_distinct_high_adjusted_r2_models(self):
        candidates = [
            {"candidate_id": "simple", "validation_adjusted_r2": 0.90, "validation_mse": 1.0, "expression_complexity": 2, "validation_prediction_signature": "a"},
            {"candidate_id": "complex", "validation_adjusted_r2": 0.95, "validation_mse": 0.5, "expression_complexity": 8, "validation_prediction_signature": "b"},
            {"candidate_id": "duplicate", "validation_adjusted_r2": 0.94, "validation_mse": 0.1, "expression_complexity": 9, "validation_prediction_signature": "b"},
        ]
        selected = select_committee(candidates, committee_size=2, criterion="adjusted_r2")
        self.assertEqual([item["candidate_id"] for item in selected], ["complex", "simple"])
        self.assertAlmostEqual(adjusted_r2(0.8, 20, 2), 0.7764705882352941)
