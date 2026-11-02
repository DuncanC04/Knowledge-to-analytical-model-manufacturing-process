import numpy as np
import pandas as pd

from src.active_learning.experiment import _task_data, pca_target_diagnostic
from src.active_learning.metrics import rank_by_disagreement


def test_rank_by_disagreement_is_label_blind_and_descending():
    predictions = np.array([[1.0, 2.0, 10.0], [1.1, 2.0, 5.0], [0.9, 2.0, 1.0]])
    ranked = rank_by_disagreement(predictions, metric="paper_ibmd")
    assert ranked.iloc[0]["pool_index"] == 2
    assert ranked.iloc[-1]["pool_index"] == 1
    assert "y" not in ranked.columns


def test_task_data_uses_axis_holdout_and_pca_is_x_only():
    frame = pd.DataFrame({"x": np.repeat([1, 2, 3, 4], 3), "z": np.tile([1.0, 2.0, 3.0], 4), "y": np.arange(12, dtype=float)})
    data, split = _task_data(frame, ["x", "z"], "y", "x", holdout_levels=1, validation_fraction=0.25, seed=4)
    assert set(frame.loc[list(split.test_indices), "x"]) == {4}
    assert len(data["train"][0]) == 2
    diagnostic = pca_target_diagnostic(data)
    assert np.isclose(sum(diagnostic["explained_variance_ratio"]), 1.0)
    assert len(diagnostic["target_correlation_by_component"]) == 2
