# Axis-wise equation active learning

This subpackage turns the existing equation generator into a query-by-
committee experiment.

For each configured output (`W`, `D`, `MRR`, `HAZ`) and each input axis, it:

1. Holds out the highest axis level as an extrapolative, initially unlabeled pool.
2. Generates one candidate archive and fits each candidate on the train split.
3. Reuses that archive to form committees of several sizes, such as 3, 5, and 9.
4. Scores pool points using the paper's coefficient of variation (`paper_cv`) or
   information-based measure (`paper_ibmd`). Labels are only used afterward for
   retrospective evaluation.
5. Saves an optional X-only PCA diagnostic. PCA is not used as a replacement for
   the physical inputs and is never fit on `[X, y]`.

The main entry point is `run_active_learning_grid` in `experiment.py`. It takes
the same `cfg`, OpenAI client, and retrieval callback used by the existing
pipeline, plus a dataframe containing all configured outputs. The returned
dictionary can be saved with `save_active_learning_result`.

Saving a result writes both the complete `grid.json` and an `equations/`
subfolder. Each task has a `*_candidates.json` file containing the fitted
parameters and metrics, plus a readable `*_equations.py` file containing the
generated equations as uniquely named importable functions. These archives can
be inspected or reused without regenerating candidates. For example, pass one
of the generated Python files to `exploration/active_learning_uncertainty.py`
with `--equation-file`.

Recommended first experiment:

```python
result = run_active_learning_grid(
    client, cfg, retrieve, frame,
    committee_sizes=(3, 5, 9), generation_count=50,
    holdout_levels=1, seed=50,
)
save_active_learning_result(result, "output/active_learning/grid.json")
```

Use the same generation budget and data split for all committee sizes. Treat
falling disagreement as a diagnostic only: success is lower error on the
locked evaluation set after the queried labels are revealed.
