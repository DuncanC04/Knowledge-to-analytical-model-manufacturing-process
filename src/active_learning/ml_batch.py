"""Batch adaptation using a five-model machine-learning committee."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from ..config_loader import load_config, resolve_path
from ..modeling import _extract_function
from .adaptation import _choose, _fit_committee, _matrix, _predict, load_equation_committee


MODEL_NAMES = ("random_forest", "gradient_boosting", "svr_rbf", "polynomial_ridge", "best_literature_equation")
METHODS = ("random", "maximin", "ml_disagreement", "hybrid")


def _models(seed: int) -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=seed, min_samples_leaf=1, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, random_state=seed, loss="squared_error"),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.05)),
        "polynomial_ridge": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), KernelRidge(alpha=1.0, kernel="linear")),
    }


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int, literature_records: list[dict]) -> np.ndarray:
    predictions = []
    for model in _models(seed).values():
        model.fit(x_train, y_train)
        predictions.append(np.asarray(model.predict(x_eval), dtype=float))
    fitted, _parameters, failures = _fit_committee(literature_records, x_train, y_train)
    if any(failures):
        raise ValueError(failures)
    predictions.append(_predict(literature_records, fitted, x_eval)[0])
    result = np.asarray(predictions)
    if not np.all(np.isfinite(result)):
        raise ValueError("ML committee produced non-finite predictions")
    return result


def _score_batch(method: str, candidate_x: np.ndarray, observed_x: np.ndarray, predictions: np.ndarray, y_observed: np.ndarray, scaler: StandardScaler, rng: np.random.Generator) -> tuple[int, dict]:
    chooser = "disagreement" if method == "ml_disagreement" else method
    return _choose(chooser, candidate_x, observed_x, predictions, y_observed, scaler, rng)


def run_ml_batch(
    frame: pd.DataFrame,
    input_columns: list[str],
    output_column: str,
    *,
    source_axis: str,
    source_max: float,
    target_value: float,
    source_sizes: tuple[int, ...] = (10, 15, 20, 30),
    acquisition_size: int = 10,
    target_test_size: int = 10,
    trials: int = 100,
    seed: int = 50,
    literature_records: list[dict] | None = None,
) -> dict:
    source = frame[frame[source_axis] <= source_max].copy()
    target = frame[np.isclose(frame[source_axis], target_value)].copy()
    results, splits = [], []
    literature_records = literature_records or []
    if len(literature_records) != 1:
        raise ValueError("Provide exactly one best literature equation for the fifth committee member")
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        target_indices = target.index.to_numpy(copy=True); rng.shuffle(target_indices)
        test_indices, candidate_indices = target_indices[:target_test_size], target_indices[target_test_size:]
        shuffled_source = source.index.to_numpy(copy=True); rng.shuffle(shuffled_source)
        test, pool = target.loc[test_indices], target.loc[candidate_indices]
        scaler = StandardScaler().fit(pd.concat([source[input_columns], pool[input_columns]])[input_columns].to_numpy(float))
        for source_size in source_sizes:
            source_indices = shuffled_source[:source_size]
            initial = source.loc[source_indices]
            x_initial, y_initial = _matrix(initial, input_columns), initial[output_column].to_numpy(float)
            initial_predictions = _fit_predict(x_initial, y_initial, _matrix(test, input_columns), seed + trial, literature_records)
            initial_mean = initial_predictions.mean(axis=0)
            initial_rmse = float(mean_squared_error(test[output_column], initial_mean) ** 0.5)
            candidate_x = _matrix(pool, input_columns)
            candidate_predictions = _fit_predict(x_initial, y_initial, candidate_x, seed + trial, literature_records)
            for method in METHODS:
                selected_positions = []
                remaining = np.ones(len(pool), dtype=bool)
                observed_x = x_initial
                for _ in range(acquisition_size):
                    remaining_positions = np.flatnonzero(remaining)
                    available_x = candidate_x[remaining_positions]
                    available_predictions = candidate_predictions[:, remaining_positions]
                    coverage_x = np.vstack([observed_x, candidate_x[selected_positions]]) if method == "hybrid" else observed_x
                    if method == "random":
                        local_choice = int(rng.integers(len(remaining_positions)))
                        score = {"score": 0.0}
                    else:
                        local_choice, score = _score_batch(method, available_x, coverage_x, available_predictions, y_initial, scaler, rng)
                    selected_position = int(remaining_positions[local_choice])
                    selected_positions.append(selected_position)
                    remaining[selected_position] = False
                selected_indices = pool.index.to_numpy()[selected_positions]
                acquired = pool.loc[selected_indices]
                adapted = pd.concat([initial, acquired])
                adapted_predictions = _fit_predict(_matrix(adapted, input_columns), adapted[output_column].to_numpy(float), _matrix(test, input_columns), seed + trial, literature_records)
                adapted_mean = adapted_predictions.mean(axis=0)
                rmse = float(mean_squared_error(test[output_column], adapted_mean) ** 0.5)
                results.append({"trial": trial, "source_size": source_size, "method": method, "source_only_rmse": initial_rmse, "rmse": rmse, "mae": float(mean_absolute_error(test[output_column], adapted_mean)), "r2": float(r2_score(test[output_column], adapted_mean)), "rmse_reduction_pct": 100 * (initial_rmse - rmse) / initial_rmse, "selected_indices": [int(i) for i in selected_indices], "selected_inputs": acquired[input_columns].to_dict("records"), "committee_disagreement_mean": float(candidate_predictions.std(axis=0).mean()), "committee_disagreement_selected_mean": float(candidate_predictions[:, selected_positions].std(axis=0).mean()), "first_selection_score": score})
        for source_size in source_sizes:
            splits.append({"trial": trial, "source_size": source_size, "source_indices": [int(i) for i in shuffled_source[:source_size]], "target_test_indices": [int(i) for i in test_indices], "target_candidate_indices": [int(i) for i in candidate_indices]})
    return {"settings": {"model_names": list(MODEL_NAMES), "literature_equation": literature_records[0].get("candidate_id"), "methods": list(METHODS), "source_sizes": list(source_sizes), "acquisition_size": acquisition_size, "target_test_size": target_test_size, "trials": trials, "seed": seed, "source_max": source_max, "target_value": target_value, "input_columns": input_columns, "output_column": output_column, "protocol": "fit four ML models plus best literature equation on source; batch-select target points without target labels; refit on source plus acquired target points; evaluate locked target test"}, "splits": splits, "results": results}


def save_ml_batch(result: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(result["results"]).drop(columns=["selected_inputs"], errors="ignore").to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(result["splits"]).to_json(output / "splits.json", orient="records", indent=2)
    pd.DataFrame(result["results"]).groupby(["source_size", "method"])[["source_only_rmse", "rmse", "mae", "r2", "rmse_reduction_pct"]].mean().to_csv(output / "summary.csv")
    return output / "experiment.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch adaptation with a five-model ML committee.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default="output/ml_batch_adaptation")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--source-sizes", nargs="+", type=int, default=[10, 15, 20, 30])
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    equations = load_equation_committee(resolve_path(cfg, "output/active_learning/equations/MRR__WaterSpeed__m_s__candidates.json"), count=1)
    result = run_ml_batch(frame, cfg["input_columns"], cfg["output_column"], source_axis=cfg["input_columns"][0], source_max=7.333333333333333, target_value=8.0, source_sizes=tuple(args.source_sizes), trials=args.trials, seed=args.seed, literature_records=equations)
    print(f"Saved {save_ml_batch(result, resolve_path(cfg, args.output_dir))}")


if __name__ == "__main__":
    main()
