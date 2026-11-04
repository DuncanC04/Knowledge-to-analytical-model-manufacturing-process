"""Fixed-equation active learning for source-to-target FLIPMM adaptation.

This module deliberately does not generate equations.  It loads a saved
committee, refits its coefficients on each trial's observed labels, and keeps
target labels out of every acquisition score.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ..config_loader import load_config, resolve_path
from ..modeling import _extract_function


METHODS = ("random", "maximin", "gp_uncertainty", "disagreement", "hybrid")


def load_equation_committee(path: str | Path, count: int = 5) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < count:
        raise ValueError(f"Equation archive must contain at least {count} candidates")
    selected = records[:count]
    for record in selected:
        if not record.get("model_code"):
            raise ValueError(f"Candidate {record.get('candidate_id')} has no model_code")
        _extract_function(record["model_code"])
    return selected


def build_ensembles(records: list[dict]) -> dict[str, list[dict]]:
    """Build top-k and source-validation-diverse committees from the archive."""
    if len(records) < 5:
        raise ValueError("Need at least five saved candidates")
    ensembles = {"top3": records[:3], "top5": records[:5], "top9": records[:9], "all20": records}
    eligible = [r for r in records if r.get("validation_predictions")]
    if not eligible:
        raise ValueError("Saved archive lacks validation_predictions")
    selected = [max(eligible, key=lambda r: float(r.get("validation_adjusted_r2", -np.inf)))]
    selected_predictions = [np.asarray(selected[0]["validation_predictions"], dtype=float)]
    while len(selected) < min(9, len(eligible)):
        def score(record: dict) -> tuple[float, float]:
            prediction = np.asarray(record["validation_predictions"], dtype=float)
            scale = max(float(np.std(np.concatenate([prediction, *selected_predictions]))), 1e-12)
            distance = min(float(np.sqrt(np.mean(((prediction - previous) / scale) ** 2))) for previous in selected_predictions)
            return distance, float(record.get("validation_adjusted_r2", -np.inf))
        choice = max((r for r in eligible if r not in selected), key=score)
        selected.append(choice)
        selected_predictions.append(np.asarray(choice["validation_predictions"], dtype=float))
    ensembles["diverse5"] = selected[:5]
    ensembles["diverse9"] = selected[:9]
    return ensembles


def attach_source_validation_predictions(records: list[dict], frame: pd.DataFrame, input_columns: list[str], validation_indices: list[int]) -> list[dict]:
    """Reconstruct archive validation predictions without using target labels."""
    x = _matrix(frame.loc[validation_indices], input_columns)
    inputs = tuple(x[:, i] for i in range(x.shape[1]))
    enriched = []
    for record in records:
        copy = dict(record)
        value = np.asarray(_extract_function(record["model_code"])(inputs, *record["parameters"]), dtype=float)
        if np.all(np.isfinite(value)):
            copy["validation_predictions"] = value.tolist()
        enriched.append(copy)
    return enriched


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=float)


def _fit_committee(records: list[dict], x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[list[float]], list[str]]:
    values, parameters, failures = [], [], []
    inputs = tuple(x[:, i] for i in range(x.shape[1]))
    for record in records:
        function = _extract_function(record["model_code"])
        n_parameters = len(record.get("parameters", []))
        if not n_parameters:
            n_parameters = len(function.__code__.co_varnames) - 1
        try:
            fitted, _ = curve_fit(function, inputs, y, p0=np.ones(n_parameters), maxfev=100_000)
            prediction = np.asarray(function(inputs, *fitted), dtype=float)
            if not np.all(np.isfinite(prediction)):
                raise ValueError("non-finite fitted prediction")
            values.append(fitted)
            parameters.append([float(v) for v in fitted])
            failures.append("")
        except Exception as exc:  # fitting failures are retained for auditability
            values.append(None)
            parameters.append([])
            failures.append(f"{type(exc).__name__}: {exc}")
    return values, parameters, failures


def _predict(records: list[dict], fitted: list[np.ndarray | None], x: np.ndarray) -> np.ndarray:
    inputs = tuple(x[:, i] for i in range(x.shape[1]))
    predictions = []
    for record, parameters in zip(records, fitted):
        if parameters is None:
            continue
        try:
            value = np.asarray(_extract_function(record["model_code"])(inputs, *parameters), dtype=float)
            if np.all(np.isfinite(value)):
                predictions.append(value)
        except Exception:
            continue
    if not predictions:
        raise ValueError("No equation in the committee produced finite predictions")
    return np.asarray(predictions)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = np.nanmax(values) - np.nanmin(values)
    return (values - np.nanmin(values)) / spread if spread > 0 else np.zeros_like(values)


def _choose(method: str, candidates: np.ndarray, observed: np.ndarray, predictions: np.ndarray,
            y_observed: np.ndarray, scaler: StandardScaler, rng: np.random.Generator) -> tuple[int, dict]:
    if method == "random":
        index = int(rng.integers(len(candidates)))
        return index, {"random_score": 1.0}
    candidate_scaled = scaler.transform(candidates)
    observed_scaled = scaler.transform(observed)
    distances = np.sqrt(((candidate_scaled[:, None, :] - observed_scaled[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    disagreement = np.std(predictions, axis=0, ddof=0)
    if method == "maximin":
        score = distances
    elif method == "disagreement":
        score = disagreement
    elif method == "hybrid":
        score = 0.5 * _normalise(distances) + 0.5 * _normalise(disagreement)
    elif method == "gp_uncertainty":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel(1e-5, (1e-8, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0, n_restarts_optimizer=0)
        gp.fit(observed_scaled, y_observed)
        _, std = gp.predict(candidate_scaled, return_std=True)
        score = std
    else:
        raise ValueError(f"Unknown acquisition method: {method}")
    index = int(np.argmax(score))
    return index, {"score": float(score[index]), "distance": float(distances[index]), "disagreement": float(disagreement[index])}


def run_adaptation(
    frame: pd.DataFrame,
    records: list[dict],
    input_columns: list[str],
    output_column: str,
    *,
    source_axis: str,
    source_max: float,
    target_axis: str,
    target_value: float,
    source_sizes: tuple[int, ...] = (10, 15, 20, 30),
    trials: int = 30,
    acquisition_budget: int = 10,
    target_test_size: int = 10,
    seed: int = 50,
    methods: tuple[str, ...] = METHODS,
    ensembles: dict[str, list[dict]] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    source = frame[frame[source_axis] <= source_max].copy()
    target = frame[np.isclose(frame[target_axis], target_value)].copy()
    if source.empty or target.empty:
        raise ValueError("Source or target domain is empty")
    if target_test_size + acquisition_budget > len(target):
        raise ValueError("Target domain is too small for test set plus acquisition budget")
    all_rows = []
    split_rows = []
    ensembles = ensembles or {"top5": records}
    ensemble_metadata = {name: [r.get("candidate_id", f"equation_{i}") for i, r in enumerate(items)] for name, items in ensembles.items()}
    for trial in range(trials):
        trial_rng = np.random.default_rng(seed + trial)
        target_indices = target.index.to_numpy(copy=True)
        trial_rng.shuffle(target_indices)
        test_indices = target_indices[:target_test_size]
        candidate_indices = target_indices[target_test_size:]
        test = target.loc[test_indices]
        pool = target.loc[candidate_indices]
        scaler = StandardScaler().fit(pd.concat([source[input_columns], pool[input_columns]])[input_columns].to_numpy(float))
        for source_size in source_sizes:
            if source_size > len(source):
                continue
            source_indices = trial_rng.choice(source.index.to_numpy(), size=source_size, replace=False)
            base = source.loc[source_indices]
            split_rows.append({"trial": trial, "source_size": source_size, "source_indices": [int(i) for i in source_indices], "target_test_indices": [int(i) for i in test_indices], "target_pool_indices": [int(i) for i in candidate_indices]})
            for ensemble_name, ensemble_records in ensembles.items():
                for method in methods:
                    observed = base.copy()
                    available = pool.copy()
                    method_key = f"{trial}:{source_size}:{ensemble_name}:{method}"
                    for k in range(acquisition_budget + 1):
                        x_obs, y_obs = _matrix(observed, input_columns), observed[output_column].to_numpy(float)
                        fitted, parameters, failures = _fit_committee(ensemble_records, x_obs, y_obs)
                        test_prediction = _predict(ensemble_records, fitted, _matrix(test, input_columns)).mean(axis=0)
                        rmse = float(mean_squared_error(test[output_column], test_prediction) ** 0.5)
                        all_rows.append({"trial": trial, "source_size": source_size, "ensemble": ensemble_name, "method": method, "acquisitions": k, "rmse": rmse, "parameter_failures": sum(bool(x) for x in failures), "parameters": parameters, "method_key": method_key})
                        if k == acquisition_budget:
                            break
                        candidate_x = _matrix(available, input_columns)
                        candidate_prediction = _predict(ensemble_records, fitted, candidate_x)
                        choice, score, = _choose(method, candidate_x, x_obs, candidate_prediction, y_obs, scaler, trial_rng)
                        chosen_index = available.index.to_numpy()[choice]
                        chosen = available.loc[chosen_index]
                        all_rows[-1].update({"selected_index": int(chosen_index), "selected_inputs": chosen[input_columns].to_dict(), **score})
                        observed = pd.concat([observed, available.loc[[chosen_index]]])
                        available = available.drop(index=chosen_index)
                    log(f"trial={trial + 1}/{trials} source={source_size} ensemble={ensemble_name} method={method}")
    return {"settings": {"source_axis": source_axis, "source_max": source_max, "target_axis": target_axis, "target_value": target_value, "source_sizes": list(source_sizes), "trials": trials, "acquisition_budget": acquisition_budget, "target_test_size": target_test_size, "seed": seed, "methods": list(methods), "ensembles": ensemble_metadata, "input_columns": input_columns, "output_column": output_column}, "splits": split_rows, "results": all_rows}


def save_adaptation_result(result: dict, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "experiment.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(result["results"]).to_csv(destination / "learning_curves.csv", index=False)
    pd.DataFrame(result["splits"]).to_json(destination / "splits.json", orient="records", indent=2)
    try:
        import matplotlib.pyplot as plt
        curves = pd.DataFrame(result["results"])
        for ensemble, ensemble_curves in curves.groupby("ensemble"):
            summary = ensemble_curves.groupby(["source_size", "method", "acquisitions"], as_index=False)["rmse"].agg(["mean", "std"]).reset_index()
            fig, axes = plt.subplots(1, len(result["settings"]["source_sizes"]), figsize=(16, 4), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, size in zip(axes, result["settings"]["source_sizes"]):
                part = summary[summary.source_size == size]
                for method, group in part.groupby("method"):
                    ax.plot(group.acquisitions, group["mean"], marker="o", label=method)
                    ax.fill_between(group.acquisitions, group["mean"] - group["std"].fillna(0), group["mean"] + group["std"].fillna(0), alpha=.15)
                ax.set_title(f"{ensemble}: source n={size}"); ax.set_xlabel("Acquired target observations"); ax.grid(alpha=.25)
            axes[0].set_ylabel("Target RMSE"); axes[-1].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(destination / f"rmse_learning_curves_{ensemble}.png", dpi=180); plt.close(fig)
    except ImportError:
        pass
    return destination / "experiment.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-equation source-to-target active learning.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--equations", default="output/active_learning/equations/MRR__WaterSpeed__m_s__candidates.json")
    parser.add_argument("--output-dir", default="output/adaptation")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--source-sizes", nargs="+", type=int, default=[10, 15, 20, 30])
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    equation_path = resolve_path(cfg, args.equations)
    records = load_equation_committee(equation_path, count=20)
    checkpoint_path = equation_path.parent.parent / "grid_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        task = next(iter(checkpoint["tasks"].values()))
        records = attach_source_validation_predictions(records, frame, cfg["input_columns"], task["split"]["validation"])
    result = run_adaptation(frame, records, cfg["input_columns"], cfg["output_column"], source_axis=cfg["input_columns"][0], source_max=7.333333333333333, target_axis=cfg["input_columns"][0], target_value=8.0, source_sizes=tuple(args.source_sizes), trials=2 if args.smoke else args.trials, seed=args.seed, ensembles=build_ensembles(records))
    print(f"Saved {save_adaptation_result(result, resolve_path(cfg, args.output_dir))}")


if __name__ == "__main__":
    main()
