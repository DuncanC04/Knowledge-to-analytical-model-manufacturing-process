"""Equation-pool active learning experiment.

This script tests a more explicit version of the idea:

1. Start with a one-shot batch of literature-derived equations.
2. Query a small seed set from the RBF-simulated FLIPMM experiment.
3. Fit every equation locally with Ridge regularization.
4. Re-rank equations by observed-data error.
5. Use only the surviving top equations to guide the next sample.
6. Repeat without calling the LLM again.
7. Flag when an LLM refresh would be reasonable, but do not call it here.

This is separate from active_learning_uncertainty.py so we can compare the
method against the acquisition-strategy benchmark outputs without mixing code.

Run:
    uv run python equation_pool_active_learning.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from active_learning_uncertainty import (
    OUT_DIR,
    RBFTruth,
    acquisition_scores,
    fit_rbf_truth,
    load_equations_from_file,
    load_equations_from_summary,
    predict_equation_ensemble,
    random_candidates,
    zscore,
)
from src.config_loader import load_config, resolve_path


DEFAULT_EQUATION_FILE = "output/one_shot_literature_equations.py"
DEFAULT_BENCHMARK = "output/active_learning_uncertainty/benchmark_summary.csv"


def equation_features(func, n_params: int, X_raw: np.ndarray) -> np.ndarray:
    """Build the same linear-in-parameters design matrix used elsewhere."""
    from active_learning_uncertainty import equation_design_matrix

    return equation_design_matrix(func, n_params, X_raw)


def fit_all_equations(equations: list[tuple], X_obs: np.ndarray, y_obs: np.ndarray):
    """Fit every equation and return metadata sorted by observed RMSE."""
    fitted = []
    for func, n_params, name in equations:
        features = equation_features(func, n_params, X_obs)
        model = Ridge(alpha=1.0, fit_intercept=False)
        model.fit(features, y_obs)
        pred = model.predict(features)
        rmse = mean_squared_error(y_obs, pred) ** 0.5
        fitted.append(
            {
                "name": name,
                "func": func,
                "n_params": n_params,
                "model": model,
                "observed_rmse": rmse,
            }
        )
    fitted.sort(key=lambda item: item["observed_rmse"])
    return fitted


def fitted_to_tuple_list(fitted: list[dict]) -> list[tuple]:
    """Adapt fitted dictionaries to active_learning_uncertainty prediction format."""
    return [
        (item["func"], item["n_params"], item["model"], item["name"])
        for item in fitted
    ]


def fit_residual_gp(X_obs_scaled: np.ndarray, residuals: np.ndarray) -> GaussianProcessRegressor:
    """Fit a small GP to residuals left by the current surviving equation pool."""
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.ones(X_obs_scaled.shape[1]), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_obs_scaled, residuals)
    return gp


def should_refresh_llm(
    recent_r2: list[float],
    step: int,
    min_step: int,
    patience: int,
    threshold: float,
) -> bool:
    """Return True when local equation-pool learning appears to be failing."""
    if step < min_step or len(recent_r2) < patience + 1:
        return False
    window = recent_r2[-patience:]
    previous = recent_r2[-patience - 1]
    improvement = max(window) - previous
    return max(window) < threshold and improvement < 0.02


def run_one_seed(
    cfg: dict,
    truth: RBFTruth,
    equations: list[tuple],
    seed: int,
    seed_size: int,
    add_steps: int,
    n_candidates: int,
    n_validation: int,
    survivors: int,
    refresh_threshold: float,
    refresh_patience: int,
    refresh_min_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one equation-pool active-learning trace."""
    rng = np.random.default_rng(seed)
    input_cols = cfg["input_columns"]

    X_obs = random_candidates(truth.input_bounds, seed_size, rng)
    y_obs = truth(X_obs)
    X_validation = random_candidates(truth.input_bounds, n_validation, rng)
    y_validation = truth(X_validation)

    rows = []
    rankings = []
    r2_history = []

    for step in range(add_steps + 1):
        all_fitted = fit_all_equations(equations, X_obs, y_obs)
        top_fitted = all_fitted[:survivors]
        top_tuple = fitted_to_tuple_list(top_fitted)

        eq_obs = predict_equation_ensemble(top_tuple, X_obs)
        residuals = y_obs - eq_obs.mean(axis=1)

        scaler = StandardScaler()
        X_obs_scaled = scaler.fit_transform(X_obs)
        residual_gp = fit_residual_gp(X_obs_scaled, residuals)

        X_val_scaled = scaler.transform(X_validation)
        eq_val = predict_equation_ensemble(top_tuple, X_validation)
        residual_mean, residual_std = residual_gp.predict(X_val_scaled, return_std=True)
        combined_val = eq_val.mean(axis=1) + residual_mean

        rmse = mean_squared_error(y_validation, combined_val) ** 0.5
        r2 = r2_score(y_validation, combined_val)
        r2_history.append(r2)

        refresh_flag = should_refresh_llm(
            r2_history,
            step=step,
            min_step=refresh_min_step,
            patience=refresh_patience,
            threshold=refresh_threshold,
        )

        rows.append(
            {
                "seed": seed,
                "step": step,
                "n_observed": len(X_obs),
                "rmse_on_rbf_truth": rmse,
                "r2_on_rbf_truth": r2,
                "top_equation": top_fitted[0]["name"],
                "top_observed_rmse": top_fitted[0]["observed_rmse"],
                "survivor_count": len(top_fitted),
                "llm_refresh_recommended": refresh_flag,
            }
        )

        for rank, item in enumerate(all_fitted, start=1):
            rankings.append(
                {
                    "seed": seed,
                    "step": step,
                    "rank": rank,
                    "equation": item["name"],
                    "observed_rmse": item["observed_rmse"],
                    "survived": rank <= survivors,
                }
            )

        if step == add_steps:
            break

        X_candidates = random_candidates(truth.input_bounds, n_candidates, rng)
        X_candidates_scaled = scaler.transform(X_candidates)
        eq_candidates = predict_equation_ensemble(top_tuple, X_candidates)
        eq_std = eq_candidates.std(axis=1)
        _, candidate_residual_std = residual_gp.predict(X_candidates_scaled, return_std=True)
        nearest_distance = cdist(X_candidates_scaled, X_obs_scaled).min(axis=1)

        # The pool-specific acquisition emphasizes the active equation set:
        # disagreement between surviving equations plus residual uncertainty.
        scores = (
            zscore(eq_std)
            + zscore(candidate_residual_std)
            + 0.25 * zscore(nearest_distance)
        )
        next_x = X_candidates[int(np.argmax(scores))]
        next_y = truth(next_x.reshape(1, -1))[0]
        X_obs = np.vstack([X_obs, next_x])
        y_obs = np.append(y_obs, next_y)

    return pd.DataFrame(rows), pd.DataFrame(rankings)


def maybe_plot_pool_results(summary: pd.DataFrame, benchmark_summary: pd.DataFrame | None) -> None:
    """Write a comparison plot against prior benchmark strategies when available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        summary["n_observed"],
        summary["r2_mean"],
        marker="o",
        linewidth=2.4,
        label="equation_pool_rerank",
    )
    ax.fill_between(
        summary["n_observed"],
        summary["r2_mean"] - summary["r2_std"],
        summary["r2_mean"] + summary["r2_std"],
        alpha=0.16,
    )

    if benchmark_summary is not None:
        for strategy, group in benchmark_summary.groupby("strategy"):
            ax.plot(
                group["n_observed"],
                group["r2_mean"],
                linewidth=1.3,
                alpha=0.75,
                label=strategy,
            )

    ax.set_xlabel("Synthetic observations queried from RBF truth")
    ax.set_ylabel("Mean R2 on validation grid")
    ax.set_title("Equation-pool reranking vs acquisition benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "equation_pool_vs_benchmark.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")


def run_experiment(args: argparse.Namespace) -> None:
    cfg = load_config()
    input_cols = cfg["input_columns"]
    output_col = cfg["output_column"]
    df = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    truth = fit_rbf_truth(df, input_cols, output_col)

    if args.equation_file:
        equations = load_equations_from_file(resolve_path(cfg, args.equation_file), args.top_models)
        equation_source = resolve_path(cfg, args.equation_file)
    else:
        equations = load_equations_from_summary(Path(args.summary_file), args.top_models)
        equation_source = Path(args.summary_file)

    traces = []
    rankings = []
    for seed_offset in range(args.seeds):
        seed = args.random_state + seed_offset
        trace, ranking = run_one_seed(
            cfg=cfg,
            truth=truth,
            equations=equations,
            seed=seed,
            seed_size=args.seed_size,
            add_steps=args.add_steps,
            n_candidates=args.n_candidates,
            n_validation=args.n_validation,
            survivors=args.survivors,
            refresh_threshold=args.refresh_threshold,
            refresh_patience=args.refresh_patience,
            refresh_min_step=args.refresh_min_step,
        )
        traces.append(trace)
        rankings.append(ranking)

    all_traces = pd.concat(traces, ignore_index=True)
    all_rankings = pd.concat(rankings, ignore_index=True)
    summary = (
        all_traces.groupby(["step", "n_observed"], as_index=False)
        .agg(
            r2_mean=("r2_on_rbf_truth", "mean"),
            r2_std=("r2_on_rbf_truth", "std"),
            rmse_mean=("rmse_on_rbf_truth", "mean"),
            rmse_std=("rmse_on_rbf_truth", "std"),
            refresh_rate=("llm_refresh_recommended", "mean"),
        )
        .fillna(0.0)
    )
    final = all_traces[all_traces["step"] == args.add_steps]
    final_row = {
        "method": "equation_pool_rerank",
        "final_r2_mean": final["r2_on_rbf_truth"].mean(),
        "final_r2_std": final["r2_on_rbf_truth"].std(),
        "final_rmse_mean": final["rmse_on_rbf_truth"].mean(),
        "final_rmse_std": final["rmse_on_rbf_truth"].std(),
        "refresh_rate_final_step": final["llm_refresh_recommended"].mean(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = OUT_DIR / "equation_pool_all_traces.csv"
    summary_path = OUT_DIR / "equation_pool_summary.csv"
    rankings_path = OUT_DIR / "equation_pool_rankings.csv"
    final_path = OUT_DIR / "equation_pool_final_summary.csv"
    all_traces.to_csv(trace_path, index=False)
    summary.to_csv(summary_path, index=False)
    all_rankings.to_csv(rankings_path, index=False)
    pd.DataFrame([final_row]).to_csv(final_path, index=False)

    benchmark_summary = None
    benchmark_path = Path(args.benchmark_summary)
    if benchmark_path.exists():
        benchmark_summary = pd.read_csv(benchmark_path)

    print(f"Equation source: {equation_source}")
    print(f"Seeds: {args.seeds}")
    print(pd.DataFrame([final_row]))
    print(f"\nWrote: {trace_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {rankings_path}")
    print(f"Wrote: {final_path}")
    maybe_plot_pool_results(summary, benchmark_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equation-file", default=DEFAULT_EQUATION_FILE)
    parser.add_argument("--summary-file", default="output/run_summary.txt")
    parser.add_argument("--benchmark-summary", default=DEFAULT_BENCHMARK)
    parser.add_argument("--top-models", type=int, default=12)
    parser.add_argument("--survivors", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-size", type=int, default=4)
    parser.add_argument("--add-steps", type=int, default=12)
    parser.add_argument("--n-candidates", type=int, default=1200)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=50)
    parser.add_argument("--refresh-threshold", type=float, default=0.65)
    parser.add_argument("--refresh-patience", type=int, default=3)
    parser.add_argument("--refresh-min-step", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
