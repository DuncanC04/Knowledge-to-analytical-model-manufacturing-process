"""Label-blind disagreement metrics for equation committees.

The committee predictions have shape ``(n_equations, n_points)``.  CV and
IBMD are only meaningful for positive predictions, as in the cited method;
undefined values are retained as NaN instead of silently becoming acquisition
scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..research import committee_disagreement


def disagreement_table(predictions: np.ndarray, tau: float = 1.0) -> pd.DataFrame:
    """Return CV, IBMD, and supporting statistics for every query point."""
    return committee_disagreement(np.asarray(predictions, dtype=float), tau=tau)


def rank_by_disagreement(
    predictions: np.ndarray,
    *,
    metric: str = "paper_ibmd",
    tau: float = 1.0,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank points without reading their labels.

    NaN scores are placed last.  The caller should use the returned row index
    to identify the point in its pool; no target values are required here.
    """
    table = disagreement_table(predictions, tau=tau).copy()
    if metric not in table.columns:
        raise ValueError(f"Unknown disagreement metric: {metric}")
    table.insert(0, "pool_index", np.arange(len(table)))
    table["acquisition_score"] = table[metric]
    table = table.sort_values("acquisition_score", ascending=False, na_position="last", ignore_index=True)
    return table if top_n is None else table.head(top_n).copy()
