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
