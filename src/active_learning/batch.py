"""Batch active-learning experiment for ten-point source-to-target adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..config_loader import load_config, resolve_path
from .adaptation import (
    METHODS,
    _choose,
    _fit_committee,
    _matrix,
    _predict,
    load_equation_committee,
)


def run_batch_adaptation(
    frame: pd.DataFrame,
    records: list[dict],
    input_columns: list[str],
    output_column: str,
    *,
    source_axis: str,
    source_max: float,
    target_value: float,
    source_sizes: tuple[int, ...] = (10,),
    acquisition_size: int = 10,
    target_test_size: int = 10,
    trials: int = 100,
    seed: int = 50,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    source = frame[frame[source_axis] <= source_max].copy()
    target = frame[np.isclose(frame[source_axis], target_value)].copy()
    if target_test_size + acquisition_size > len(target):
        raise ValueError("Target domain is too small for test and acquisition partitions")
    results: list[dict] = []
    split_rows: list[dict] = []
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        target_indices = target.index.to_numpy(copy=True)
        rng.shuffle(target_indices)
        test_indices = target_indices[:target_test_size]
        candidate_indices = target_indices[target_test_size:]
        shuffled_source = source.index.to_numpy(copy=True)
        rng.shuffle(shuffled_source)
        test = target.loc[test_indices]
        pool = target.loc[candidate_indices]
        scaler = StandardScaler().fit(pd.concat([source[input_columns], pool[input_columns]])[input_columns].to_numpy(float))
        for source_size in source_sizes:
            source_indices = shuffled_source[:source_size]
            initial = source.loc[source_indices]
            x_initial, y_initial = _matrix(initial, input_columns), initial[output_column].to_numpy(float)
            fitted_initial, initial_parameters, initial_failures = _fit_committee(records, x_initial, y_initial)
            initial_prediction = _predict(records, fitted_initial, _matrix(test, input_columns)).mean(axis=0)
            initial_metrics = {
                "rmse": float(mean_squared_error(test[output_column], initial_prediction) ** 0.5),
                "mae": float(mean_absolute_error(test[output_column], initial_prediction)),
                "r2": float(r2_score(test[output_column], initial_prediction)),
            }
            for method in methods:
                observed_x = x_initial
                candidate_x = _matrix(pool, input_columns)
                candidate_prediction = _predict(records, fitted_initial, candidate_x)
                choice, score = _choose(method, candidate_x, observed_x, candidate_prediction, y_initial, scaler, rng)
            # Random, distance, GP, and disagreement each return one best point.
            # Batch selection is greedy: after every choice, the model is not
            # refit and labels remain hidden; only geometric/prediction scores
            # are recomputed from the fixed source-fitted committee.
                selected_positions = [choice]
                remaining = np.ones(len(pool), dtype=bool)
                remaining[choice] = False
                while len(selected_positions) < acquisition_size:
                    remaining_positions = np.flatnonzero(remaining)
                    rem_x = candidate_x[remaining_positions]
                # Reuse the same label-blind score with the selected batch added
                # to the coverage set. GP/disagreement remain source-model scores.
                    selected_x = candidate_x[selected_positions]
                    if method in {"maximin", "hybrid"}:
                        coverage_x = np.vstack([observed_x, selected_x])
                    else:
                        coverage_x = observed_x
                    if method == "random":
                        next_local = int(rng.integers(len(remaining_positions)))
                        next_position = int(remaining_positions[next_local])
                    else:
                        rem_predictions = candidate_prediction[:, remaining_positions]
                        next_local, _ = _choose(method, rem_x, coverage_x, rem_predictions, y_initial, scaler, rng)
                        next_position = int(remaining_positions[next_local])
                    selected_positions.append(next_position)
                    remaining[next_position] = False
                selected_indices = pool.index.to_numpy()[selected_positions]
                acquired = pool.loc[selected_indices]
                adapted = pd.concat([initial, acquired])
                x_adapted, y_adapted = _matrix(adapted, input_columns), adapted[output_column].to_numpy(float)
                fitted, parameters, failures = _fit_committee(records, x_adapted, y_adapted)
                prediction = _predict(records, fitted, _matrix(test, input_columns)).mean(axis=0)
                metrics = {
                    "rmse": float(mean_squared_error(test[output_column], prediction) ** 0.5),
                    "mae": float(mean_absolute_error(test[output_column], prediction)),
                    "r2": float(r2_score(test[output_column], prediction)),
                }
                results.append({"trial": trial, "method": method, "source_size": source_size, "acquisition_size": acquisition_size, "initial_rmse": initial_metrics["rmse"], "initial_mae": initial_metrics["mae"], "initial_r2": initial_metrics["r2"], "rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"], "rmse_reduction_pct": 100 * (initial_metrics["rmse"] - metrics["rmse"]) / initial_metrics["rmse"], "parameter_failures": sum(bool(x) for x in failures), "initial_parameter_failures": sum(bool(x) for x in initial_failures), "selected_indices": [int(i) for i in selected_indices], "selected_inputs": acquired[input_columns].to_dict("records"), "selection_score_first": score, "parameters": parameters, "initial_parameters": initial_parameters})
            for source_size in source_sizes:
                split_rows.append({"trial": trial, "source_size": source_size, "source_indices": [int(i) for i in shuffled_source[:source_size]], "target_test_indices": [int(i) for i in test_indices], "target_candidate_indices": [int(i) for i in candidate_indices]})
    return {"settings": {"source_axis": source_axis, "source_max": source_max, "target_value": target_value, "source_sizes": list(source_sizes), "acquisition_size": acquisition_size, "target_test_size": target_test_size, "trials": trials, "seed": seed, "methods": list(methods), "equation_ids": [r.get("candidate_id") for r in records], "input_columns": input_columns, "output_column": output_column, "protocol": "fit source-only; batch-select target points label-blind; refit source plus acquired target points; evaluate locked target test"}, "splits": split_rows, "results": results}


def save_batch_result(result: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(result["results"]).drop(columns=["selected_inputs", "parameters", "initial_parameters"], errors="ignore").to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(result["splits"]).to_json(output / "splits.json", orient="records", indent=2)
    summary = pd.DataFrame(result["results"]).groupby("method").agg({"initial_rmse": "mean", "rmse": ["mean", "std"], "mae": "mean", "r2": "mean", "rmse_reduction_pct": "mean"})
    summary.to_csv(output / "summary.csv")
    return output / "experiment.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated batch active-learning adaptation.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--equations", default="output/active_learning/equations/MRR__WaterSpeed__m_s__candidates.json")
    parser.add_argument("--output-dir", default="output/batch_adaptation")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--source-sizes", nargs="+", type=int, default=[10, 15, 20, 30])
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    records = load_equation_committee(resolve_path(cfg, args.equations), count=5)
    result = run_batch_adaptation(frame, records, cfg["input_columns"], cfg["output_column"], source_axis=cfg["input_columns"][0], source_max=7.333333333333333, target_value=8.0, source_sizes=tuple(args.source_sizes), trials=args.trials, seed=args.seed)
    print(f"Saved {save_batch_result(result, resolve_path(cfg, args.output_dir))}")


if __name__ == "__main__":
    main()
