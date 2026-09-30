"""RBF-simulated active learning for knowledge-guided equations.

This is a standalone exploration script. It does not modify the main pipeline.

Idea being tested
-----------------
The real FLIPMM spreadsheet is used once to build an RBF surrogate. After that,
the active learner pretends the RBF is the experimental system:

    choose (WS, P, F, SS) -> query RBF truth -> observe synthetic MRR

This lets us test a Bayesian-optimization-like loop without being limited to
existing spreadsheet rows. The learner combines:

1. equation models from output/run_summary.txt,
2. a GP residual model with an RBF kernel,
3. equation disagreement,
4. distance from already sampled points.

The result is not a finished method. It is a readable sandbox for asking:

    "Where should we sample next if our goal is to learn the response surface
     and understand which equation structures are plausible?"

Run:
    uv run python active_learning_uncertainty.py
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config, resolve_path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "output" / "run_summary.txt"
OUT_DIR = ROOT / "output" / "active_learning_uncertainty"


@dataclass
class RBFTruth:
    """Synthetic experiment fitted to the full FLIPMM dataset."""

    model: RBFInterpolator
    scaler: StandardScaler
    input_bounds: np.ndarray

    def __call__(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(np.asarray(X_raw, dtype=float))
        return np.asarray(self.model(X_scaled), dtype=float).reshape(-1)


def extract_unique_equation_codes(summary_path: Path, max_models: int) -> list[str]:
    """Extract unique generated equation forms from run_summary.txt."""
    text = summary_path.read_text(encoding="utf-8")
    raw_codes = re.findall(
        r"Model Code:\s*\n(?P<code>def\s+\w+\(.*?return\s+\w+)",
        text,
        flags=re.DOTALL,
    )

    unique_codes = []
    seen = set()
    for code in raw_codes:
        normalized = re.sub(r"\s+", " ", code.strip())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_codes.append(code.strip())
        if len(unique_codes) == max_models:
            break

    if not unique_codes:
        raise RuntimeError(f"No equation functions found in {summary_path}.")
    return unique_codes


def load_equations_from_file(equation_file: Path, max_models: int) -> list[tuple]:
    """Load generated equation functions from an importable Python file.

    The one-shot generator writes GENERATED_FUNCTIONS, but this also falls back
    to any function whose name starts with literature_model_.
    """
    spec = importlib.util.spec_from_file_location("one_shot_equations", equation_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import equation file: {equation_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = list(getattr(module, "GENERATED_FUNCTIONS", []))
    if not names:
        names = [
            name
            for name, value in vars(module).items()
            if name.startswith("literature_model_") and callable(value)
        ]

    equations = []
    for name in names[:max_models]:
        func = getattr(module, name)
        n_params = len(inspect.signature(func).parameters) - 1
        equations.append((func, n_params, name))

    if not equations:
        raise RuntimeError(f"No equation functions found in {equation_file}")
    return equations


def load_equations_from_summary(summary_path: Path, max_models: int) -> list[tuple]:
    """Load generated equation functions from output/run_summary.txt."""
    equations = []
    for i, code in enumerate(extract_unique_equation_codes(summary_path, max_models), start=1):
        func, n_params = compile_equation(code)
        equations.append((func, n_params, f"summary_model_{i}"))
    return equations


def compile_equation(code: str):
    """Compile one generated equation function from this project's run summary."""
    namespace = {"np": np, "numpy": np, "math": math}
    exec(code, namespace)  # noqa: S102 - local generated code from this project.
    function_name = re.search(r"def\s+(\w+)\s*\(", code).group(1)
    func = namespace[function_name]
    n_params = len(inspect.signature(func).parameters) - 1
    return func, n_params


def as_model_input(X_2d: np.ndarray) -> tuple[np.ndarray, ...]:
    """Generated equation code expects X as a tuple of input arrays."""
    return tuple(X_2d[:, i] for i in range(X_2d.shape[1]))


def equation_design_matrix(func, n_params: int, X_raw: np.ndarray) -> np.ndarray:
    """Turn a generated equation form into a linear feature matrix.

    The generated FLIPMM equations in run_summary.txt are linear in their
    coefficients a0, a1, ... even when the variables appear nonlinearly
    (WS**2, P**2, etc.). To avoid unstable curve_fit with very few observations,
    we evaluate one column per coefficient and fit those columns with Ridge.
    """
    columns = []
    for i in range(n_params):
        params = np.zeros(n_params)
        params[i] = 1.0
        values = np.asarray(func(as_model_input(X_raw), *params), dtype=float)
        columns.append(values)
    return np.column_stack(columns)


def fit_equation_ensemble(equations: list[tuple], X_obs: np.ndarray, y_obs: np.ndarray):
    """Fit equation forms with regularization so tiny datasets do not explode."""
    fitted = []
    for func, n_params, name in equations:
        features = equation_design_matrix(func, n_params, X_obs)
        model = Ridge(alpha=1.0, fit_intercept=False)
        model.fit(features, y_obs)
        fitted.append((func, n_params, model, name))
    return fitted


def predict_equation_ensemble(fitted, X_raw: np.ndarray) -> np.ndarray:
    """Return an (n_points, n_equations) prediction matrix."""
    predictions = []
    for func, n_params, model, _ in fitted:
        features = equation_design_matrix(func, n_params, X_raw)
        predictions.append(model.predict(features))
    return np.column_stack(predictions)


def fit_rbf_truth(df: pd.DataFrame, input_cols: list[str], output_col: str) -> RBFTruth:
    """Fit the RBF surrogate that acts as our synthetic experimental system."""
    X = df[input_cols].to_numpy(dtype=float)
    y = df[output_col].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Smoothing keeps the synthetic truth from chasing every small dataset wiggle.
    # This makes it more like a smooth physical response surface.
    model = RBFInterpolator(
        X_scaled,
        y,
        kernel="thin_plate_spline",
        smoothing=1e-3,
        degree=1,
    )
    bounds = np.column_stack([X.min(axis=0), X.max(axis=0)])
    return RBFTruth(model=model, scaler=scaler, input_bounds=bounds)


def random_candidates(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample continuous candidate experiments inside the FLIPMM input bounds."""
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low=low, high=high, size=(n, len(low)))


def fit_residual_gp(
    X_obs_scaled: np.ndarray,
    residuals: np.ndarray,
) -> GaussianProcessRegressor:
    """Fit a GP to what the equation ensemble misses."""
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
        # These warnings are common with tiny early designs. They are useful
        # during method development, but too noisy for this simple script.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gp.fit(X_obs_scaled, residuals)
    return gp


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardize an acquisition component before combining terms."""
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def acquisition_scores(
    strategy: str,
    residual_std: np.ndarray,
    equation_std: np.ndarray,
    nearest_distance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return candidate scores for one active-learning strategy."""
    if strategy == "random":
        return rng.random(len(residual_std))
    if strategy == "gp_uncertainty":
        return residual_std
    if strategy == "equation_disagreement":
        return equation_std
    if strategy == "distance":
        return nearest_distance
    if strategy == "equation_plus_gp":
        return zscore(equation_std) + zscore(residual_std)
    if strategy == "combined":
        return zscore(residual_std) + zscore(equation_std) + 0.5 * zscore(nearest_distance)
    raise ValueError(f"Unknown acquisition strategy: {strategy}")


def build_slice_grid(
    bounds: np.ndarray,
    observed_X: np.ndarray,
    slice_index: int,
    n_points: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a readable 1D slice through the 4D design space."""
    baseline = np.median(observed_X, axis=0)
    values = np.linspace(bounds[slice_index, 0], bounds[slice_index, 1], n_points)
    grid = np.tile(baseline, (n_points, 1))
    grid[:, slice_index] = values
    return values, grid


def maybe_write_plots(
    trace: pd.DataFrame,
    stack_frames: list[dict],
    validation_frame: pd.DataFrame,
    sampled_points: pd.DataFrame,
    original_points: pd.DataFrame,
    input_cols: list[str],
    slice_label: str,
    output_label: str,
) -> None:
    """Write simple PNG plots if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV outputs were still written.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trace["n_observed"], trace["rmse_on_rbf_truth"], marker="o")
    ax.set_xlabel("Synthetic observations queried from RBF truth")
    ax.set_ylabel("RMSE on RBF truth validation grid")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "rbf_truth_learning_curve.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    axes[0].plot(trace["n_observed"], trace["next_residual_std"], marker="o")
    axes[0].set_ylabel("Residual GP std")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trace["n_observed"], trace["next_equation_std"], marker="o", color="tab:green")
    axes[1].set_ylabel("Equation disagreement")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trace["n_observed"], trace["next_nearest_distance"], marker="o", color="tab:purple")
    axes[2].set_xlabel("Synthetic observations queried from RBF truth")
    axes[2].set_ylabel("Distance from sampled data")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Why the active learner chose each next point")
    path = OUT_DIR / "rbf_truth_uncertainty_components.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(
        validation_frame["rbf_truth"],
        validation_frame["combined_mean"],
        c=validation_frame["residual_std"],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    limits = [
        min(validation_frame["rbf_truth"].min(), validation_frame["combined_mean"].min()),
        max(validation_frame["rbf_truth"].max(), validation_frame["combined_mean"].max()),
    ]
    ax.plot(limits, limits, color="black", linewidth=1.2)
    ax.set_xlabel("RBF truth")
    ax.set_ylabel("Equation mean + GP residual")
    ax.set_title("Final validation parity; color = residual uncertainty")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / "rbf_truth_final_parity.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")

    # PCA gives a readable 2D map of the 4D design space. It is only a view,
    # not the true geometry, but it helps show whether the acquisition strategy
    # is spreading samples through the RBF experiment domain.
    pca = PCA(n_components=2, random_state=0)
    pca.fit(original_points[input_cols].to_numpy(dtype=float))
    original_2d = pca.transform(original_points[input_cols].to_numpy(dtype=float))
    sampled_2d = pca.transform(sampled_points[input_cols].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(original_2d[:, 0], original_2d[:, 1], s=12, color="lightgray", label="Original FLIPMM rows")
    path_colors = np.arange(len(sampled_points))
    points = ax.scatter(
        sampled_2d[:, 0],
        sampled_2d[:, 1],
        c=path_colors,
        cmap="plasma",
        s=42,
        edgecolor="black",
        linewidth=0.4,
        label="RBF queries",
    )
    ax.plot(sampled_2d[:, 0], sampled_2d[:, 1], color="tab:red", alpha=0.55, linewidth=1.2)
    ax.set_xlabel("PCA 1 of input space")
    ax.set_ylabel("PCA 2 of input space")
    ax.set_title("Continuous experiment points selected by acquisition")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Query order")
    fig.tight_layout()
    path = OUT_DIR / "rbf_truth_sample_path_pca.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")

    rows = len(stack_frames)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(9, max(2.2 * rows, 5)),
        sharex=True,
        constrained_layout=True,
    )
    if rows == 1:
        axes = [axes]

    n_models = stack_frames[0]["equation_predictions"].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 1)))
    for ax, frame in zip(axes, stack_frames):
        x = frame["slice_values"]
        y_true = frame["truth"]
        eq = frame["equation_predictions"]
        residual_mean = frame["residual_mean"]
        residual_std = frame["residual_std"]
        combined_mean = eq.mean(axis=1) + residual_mean

        ax.plot(x, y_true, color="black", linewidth=2.0, label="RBF truth")
        ax.plot(x, combined_mean, color="tab:blue", linewidth=2.0, label="Eq mean + GP residual")
        ax.fill_between(
            x,
            combined_mean - 2 * residual_std,
            combined_mean + 2 * residual_std,
            color="tab:blue",
            alpha=0.14,
            label="Residual GP +/- 2 std",
        )
        for i in range(n_models):
            ax.plot(
                x,
                eq[:, i],
                color=colors[i],
                alpha=0.75,
                linewidth=1.2,
                linestyle="--",
                label=f"Equation {i + 1}",
            )
        ax.scatter(
            frame["observed_slice_x"],
            frame["observed_y"],
            color="black",
            s=16,
            alpha=0.6,
        )
        ax.set_ylabel(f"n={frame['n_observed']}")
        ax.grid(True, alpha=0.25)

    axes[0].set_title("Equation disagreement and residual uncertainty as RBF samples are added")
    axes[-1].set_xlabel(slice_label)
    fig.supylabel(output_label)

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate legend labels.
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper right", bbox_to_anchor=(0.98, 0.995))

    path = OUT_DIR / "rbf_truth_equation_stack.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")


def run_experiment(
    seed_size: int,
    add_steps: int,
    top_models: int,
    equation_file: str | None,
    n_candidates: int,
    n_validation: int,
    random_state: int,
    slice_column: str | None,
    strategy: str = "combined",
) -> None:
    """Run the RBF-truth active-learning experiment."""
    cfg = load_config()
    input_cols = cfg["input_columns"]
    output_col = cfg["output_column"]

    if slice_column is None:
        slice_column = input_cols[min(1, len(input_cols) - 1)]
    if slice_column not in input_cols:
        raise ValueError(f"{slice_column!r} is not one of the input columns: {input_cols}")
    slice_index = input_cols.index(slice_column)

    df = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    truth = fit_rbf_truth(df, input_cols, output_col)
    if equation_file:
        equation_path = resolve_path(cfg, equation_file)
        equations = load_equations_from_file(equation_path, top_models)
        equation_source = equation_path
    else:
        equations = load_equations_from_summary(SUMMARY_PATH, top_models)
        equation_source = SUMMARY_PATH

    rng = np.random.default_rng(random_state)
    X_obs = random_candidates(truth.input_bounds, seed_size, rng)
    y_obs = truth(X_obs)
    X_validation = random_candidates(truth.input_bounds, n_validation, rng)
    y_validation = truth(X_validation)

    trace_rows = []
    stack_frames = []
    final_candidates = None
    final_validation = None

    for step in range(add_steps + 1):
        equation_models = fit_equation_ensemble(equations, X_obs, y_obs)
        eq_obs = predict_equation_ensemble(equation_models, X_obs)
        eq_obs_mean = eq_obs.mean(axis=1)
        residuals = y_obs - eq_obs_mean

        obs_scaler = StandardScaler()
        X_obs_scaled = obs_scaler.fit_transform(X_obs)
        residual_gp = fit_residual_gp(X_obs_scaled, residuals)

        X_val_scaled = obs_scaler.transform(X_validation)
        eq_val = predict_equation_ensemble(equation_models, X_validation)
        val_residual_mean, val_residual_std = residual_gp.predict(
            X_val_scaled,
            return_std=True,
        )
        val_mean = eq_val.mean(axis=1) + val_residual_mean
        val_eq_std = eq_val.std(axis=1)

        rmse = mean_squared_error(y_validation, val_mean) ** 0.5
        r2 = r2_score(y_validation, val_mean)

        X_candidates = random_candidates(truth.input_bounds, n_candidates, rng)
        X_candidates_scaled = obs_scaler.transform(X_candidates)
        eq_candidates = predict_equation_ensemble(equation_models, X_candidates)
        eq_mean = eq_candidates.mean(axis=1)
        eq_std = eq_candidates.std(axis=1)
        residual_mean, residual_std = residual_gp.predict(
            X_candidates_scaled,
            return_std=True,
        )
        nearest_distance = cdist(X_candidates_scaled, X_obs_scaled).min(axis=1)

        # This acquisition is not "the" correct one. It is a transparent first
        # version: sample where the selected strategy says we learn most.
        acquisition = acquisition_scores(
            strategy,
            residual_std=residual_std,
            equation_std=eq_std,
            nearest_distance=nearest_distance,
            rng=rng,
        )
        next_pos = int(np.argmax(acquisition))
        next_x = X_candidates[next_pos]
        next_y = truth(next_x.reshape(1, -1))[0]

        trace_rows.append(
            {
                "step": step,
                "n_observed": len(X_obs),
                "n_equation_forms": len(equations),
                "rmse_on_rbf_truth": rmse,
                "r2_on_rbf_truth": r2,
                "next_true_y": next_y,
                "next_equation_mean": eq_mean[next_pos],
                "next_equation_std": eq_std[next_pos],
                "next_residual_mean": residual_mean[next_pos],
                "next_residual_std": residual_std[next_pos],
                "next_nearest_distance": nearest_distance[next_pos],
                "next_acquisition": acquisition[next_pos],
                "strategy": strategy,
                **{f"next_{col}": next_x[i] for i, col in enumerate(input_cols)},
            }
        )

        slice_values, X_slice = build_slice_grid(
            truth.input_bounds,
            X_obs,
            slice_index=slice_index,
        )
        slice_scaled = obs_scaler.transform(X_slice)
        slice_eq = predict_equation_ensemble(equation_models, X_slice)
        slice_residual_mean, slice_residual_std = residual_gp.predict(
            slice_scaled,
            return_std=True,
        )
        stack_frames.append(
            {
                "n_observed": len(X_obs),
                "slice_values": slice_values,
                "truth": truth(X_slice),
                "equation_predictions": slice_eq,
                "residual_mean": slice_residual_mean,
                "residual_std": slice_residual_std,
                "observed_slice_x": X_obs[:, slice_index],
                "observed_y": y_obs,
            }
        )

        final_candidates = pd.DataFrame(X_candidates, columns=input_cols)
        final_candidates["rbf_truth"] = truth(X_candidates)
        final_candidates["equation_mean"] = eq_mean
        final_candidates["equation_std"] = eq_std
        final_candidates["residual_mean"] = residual_mean
        final_candidates["residual_std"] = residual_std
        final_candidates["combined_mean"] = eq_mean + residual_mean
        final_candidates["nearest_distance"] = nearest_distance
        final_candidates["acquisition"] = acquisition
        final_validation = pd.DataFrame(
            {
                "rbf_truth": y_validation,
                "equation_mean": eq_val.mean(axis=1),
                "equation_std": val_eq_std,
                "residual_mean": val_residual_mean,
                "residual_std": val_residual_std,
                "combined_mean": val_mean,
            }
        )

        if step == add_steps:
            break

        X_obs = np.vstack([X_obs, next_x])
        y_obs = np.append(y_obs, next_y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(trace_rows)
    trace_path = OUT_DIR / "rbf_truth_learning_trace.csv"
    candidates_path = OUT_DIR / "rbf_truth_final_candidates.csv"
    validation_path = OUT_DIR / "rbf_truth_final_validation.csv"
    sampled_path = OUT_DIR / "rbf_truth_sampled_points.csv"
    trace.to_csv(trace_path, index=False)
    final_candidates.to_csv(candidates_path, index=False)
    final_validation.to_csv(validation_path, index=False)
    sampled_points = pd.DataFrame(X_obs, columns=input_cols)
    sampled_points["rbf_truth"] = y_obs
    sampled_points["query_order"] = np.arange(len(sampled_points))
    sampled_points.to_csv(sampled_path, index=False)

    print(f"RBF truth fitted from {len(df)} FLIPMM rows.")
    print(f"Used {len(equations)} equation form(s) from {equation_source}.")
    print(f"Acquisition strategy: {strategy}")
    print(trace[["step", "n_observed", "rmse_on_rbf_truth", "r2_on_rbf_truth"]])
    print(f"\nWrote: {trace_path}")
    print(f"Wrote: {candidates_path}")
    print(f"Wrote: {validation_path}")
    print(f"Wrote: {sampled_path}")
    maybe_write_plots(
        trace,
        stack_frames,
        final_validation,
        sampled_points,
        df,
        input_cols,
        slice_column,
        output_col,
    )


def simulate_trace(
    cfg: dict,
    df: pd.DataFrame,
    truth: RBFTruth,
    equations: list[tuple],
    strategy: str,
    seed_size: int,
    add_steps: int,
    n_candidates: int,
    n_validation: int,
    random_state: int,
) -> pd.DataFrame:
    """Run one benchmark trace without writing per-step plots/files."""
    input_cols = cfg["input_columns"]
    output_col = cfg["output_column"]
    rng = np.random.default_rng(random_state)

    X_obs = random_candidates(truth.input_bounds, seed_size, rng)
    y_obs = truth(X_obs)
    X_validation = random_candidates(truth.input_bounds, n_validation, rng)
    y_validation = truth(X_validation)

    rows = []
    for step in range(add_steps + 1):
        equation_models = fit_equation_ensemble(equations, X_obs, y_obs)
        eq_obs = predict_equation_ensemble(equation_models, X_obs)
        residuals = y_obs - eq_obs.mean(axis=1)

        obs_scaler = StandardScaler()
        X_obs_scaled = obs_scaler.fit_transform(X_obs)
        residual_gp = fit_residual_gp(X_obs_scaled, residuals)

        X_val_scaled = obs_scaler.transform(X_validation)
        eq_val = predict_equation_ensemble(equation_models, X_validation)
        val_residual_mean, _ = residual_gp.predict(X_val_scaled, return_std=True)
        val_mean = eq_val.mean(axis=1) + val_residual_mean

        rmse = mean_squared_error(y_validation, val_mean) ** 0.5
        r2 = r2_score(y_validation, val_mean)
        rows.append(
            {
                "strategy": strategy,
                "seed": random_state,
                "step": step,
                "n_observed": len(X_obs),
                "rmse_on_rbf_truth": rmse,
                "r2_on_rbf_truth": r2,
            }
        )

        if step == add_steps:
            break

        X_candidates = random_candidates(truth.input_bounds, n_candidates, rng)
        X_candidates_scaled = obs_scaler.transform(X_candidates)
        eq_candidates = predict_equation_ensemble(equation_models, X_candidates)
        eq_std = eq_candidates.std(axis=1)
        _, residual_std = residual_gp.predict(X_candidates_scaled, return_std=True)
        nearest_distance = cdist(X_candidates_scaled, X_obs_scaled).min(axis=1)
        acquisition = acquisition_scores(
            strategy,
            residual_std=residual_std,
            equation_std=eq_std,
            nearest_distance=nearest_distance,
            rng=rng,
        )
        next_x = X_candidates[int(np.argmax(acquisition))]
        next_y = truth(next_x.reshape(1, -1))[0]
        X_obs = np.vstack([X_obs, next_x])
        y_obs = np.append(y_obs, next_y)

    return pd.DataFrame(rows)


def maybe_write_benchmark_plot(summary: pd.DataFrame) -> None:
    """Plot mean R2 learning curves by acquisition strategy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for strategy, group in summary.groupby("strategy"):
        ax.plot(
            group["n_observed"],
            group["r2_mean"],
            marker="o",
            label=strategy,
        )
        ax.fill_between(
            group["n_observed"],
            group["r2_mean"] - group["r2_std"],
            group["r2_mean"] + group["r2_std"],
            alpha=0.12,
        )
    ax.set_xlabel("Synthetic observations queried from RBF truth")
    ax.set_ylabel("Mean R2 on validation grid")
    ax.set_title("Acquisition strategy benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = OUT_DIR / "benchmark_r2_curves.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Compare acquisition strategies over multiple random seeds."""
    cfg = load_config()
    input_cols = cfg["input_columns"]
    output_col = cfg["output_column"]
    df = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    truth = fit_rbf_truth(df, input_cols, output_col)

    if args.equation_file:
        equations = load_equations_from_file(resolve_path(cfg, args.equation_file), args.top_models)
        equation_source = resolve_path(cfg, args.equation_file)
    else:
        equations = load_equations_from_summary(SUMMARY_PATH, args.top_models)
        equation_source = SUMMARY_PATH

    strategies = [
        "random",
        "gp_uncertainty",
        "equation_disagreement",
        "distance",
        "equation_plus_gp",
        "combined",
    ]
    traces = []
    for seed_offset in range(args.benchmark_seeds):
        seed = args.random_state + seed_offset
        for strategy in strategies:
            traces.append(
                simulate_trace(
                    cfg=cfg,
                    df=df,
                    truth=truth,
                    equations=equations,
                    strategy=strategy,
                    seed_size=args.seed_size,
                    add_steps=args.add_steps,
                    n_candidates=args.n_candidates,
                    n_validation=args.n_validation,
                    random_state=seed,
                )
            )

    all_traces = pd.concat(traces, ignore_index=True)
    summary = (
        all_traces.groupby(["strategy", "step", "n_observed"], as_index=False)
        .agg(
            r2_mean=("r2_on_rbf_truth", "mean"),
            r2_std=("r2_on_rbf_truth", "std"),
            rmse_mean=("rmse_on_rbf_truth", "mean"),
            rmse_std=("rmse_on_rbf_truth", "std"),
        )
        .fillna(0.0)
    )
    final = (
        all_traces[all_traces["step"] == args.add_steps]
        .groupby("strategy", as_index=False)
        .agg(
            final_r2_mean=("r2_on_rbf_truth", "mean"),
            final_r2_std=("r2_on_rbf_truth", "std"),
            final_rmse_mean=("rmse_on_rbf_truth", "mean"),
            final_rmse_std=("rmse_on_rbf_truth", "std"),
        )
        .sort_values("final_r2_mean", ascending=False)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = OUT_DIR / "benchmark_all_traces.csv"
    summary_path = OUT_DIR / "benchmark_summary.csv"
    final_path = OUT_DIR / "benchmark_final_rankings.csv"
    all_traces.to_csv(trace_path, index=False)
    summary.to_csv(summary_path, index=False)
    final.to_csv(final_path, index=False)

    print(f"Benchmark equation source: {equation_source}")
    print(f"Seeds per strategy: {args.benchmark_seeds}")
    print(final)
    print(f"\nWrote: {trace_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {final_path}")
    maybe_write_benchmark_plot(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run multiple acquisition strategies over multiple seeds.",
    )
    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        default=10,
        help="Number of random seeds for --benchmark.",
    )
    parser.add_argument("--seed-size", type=int, default=4)
    parser.add_argument("--add-steps", type=int, default=12)
    parser.add_argument("--top-models", type=int, default=5)
    parser.add_argument(
        "--equation-file",
        default=None,
        help=(
            "Optional Python file of generated equation functions, such as "
            "output/one_shot_literature_equations.py. If omitted, uses run_summary.txt."
        ),
    )
    parser.add_argument("--n-candidates", type=int, default=3000)
    parser.add_argument("--n-validation", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=50)
    parser.add_argument(
        "--strategy",
        default="combined",
        choices=[
            "random",
            "gp_uncertainty",
            "equation_disagreement",
            "distance",
            "equation_plus_gp",
            "combined",
        ],
        help="Acquisition strategy for a single run.",
    )
    parser.add_argument(
        "--slice-column",
        default=None,
        help="Input column to vary in stacked plots. Defaults to Pulse Energy for FLIPMM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.benchmark:
        run_benchmark(args)
    else:
        run_experiment(
            seed_size=args.seed_size,
            add_steps=args.add_steps,
            top_models=args.top_models,
            equation_file=args.equation_file,
            n_candidates=args.n_candidates,
            n_validation=args.n_validation,
            random_state=args.random_state,
            slice_column=args.slice_column,
            strategy=args.strategy,
        )
