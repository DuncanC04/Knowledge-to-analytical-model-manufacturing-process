"""Visualize the RBF test environment used for active-learning experiments.

The RBF is not the method being tested. It is a fast, local stand-in for
expensive FLIPMM experiments. These plots help sanity-check that stand-in and
then compare how different acquisition strategies learn from it.

Run:
    uv run python visualize_rbf_test_environment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from active_learning_uncertainty import fit_rbf_truth
from src.config_loader import load_config, resolve_path


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE = ROOT / "output" / "active_learning_uncertainty" / "benchmark_all_traces.csv"
OUT_DIR = ROOT / "output" / "active_learning_uncertainty" / "rbf_test_environment"


def load_data() -> tuple[pd.DataFrame, list[str], str]:
    """Load the configured FLIPMM dataset and variable names."""
    cfg = load_config()
    df = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    return df, cfg["input_columns"], cfg["output_column"]


def noisy_response(y_clean: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Add reproducible Gaussian measurement noise to a clean RBF response."""
    if noise_std <= 0:
        return y_clean
    return y_clean + rng.normal(loc=0.0, scale=noise_std, size=len(y_clean))


def plot_rbf_parity(
    df: pd.DataFrame,
    input_cols: list[str],
    output_col: str,
    out_dir: Path,
    noise_std: float,
    rng: np.random.Generator,
) -> Path:
    """Check whether the fake lab reproduces the data it was built from."""
    truth = fit_rbf_truth(df, input_cols, output_col)
    X = df[input_cols].to_numpy(dtype=float)
    y = df[output_col].to_numpy(dtype=float)
    y_hat_clean = truth(X)
    y_hat = noisy_response(y_hat_clean, rng, noise_std)

    rmse = mean_squared_error(y, y_hat) ** 0.5
    r2 = r2_score(y, y_hat)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(y, y_hat, s=34, alpha=0.75, edgecolor="white", linewidth=0.4)
    limits = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(f"Measured {output_col}")
    ax.set_ylabel("Noisy RBF fake-lab response")
    ax.set_title("RBF test environment parity")
    ax.text(
        0.04,
        0.96,
        f"Noise std: {noise_std:.3g}\nRMSE: {rmse:.3g}\nR2: {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()

    path = out_dir / "rbf_fake_lab_parity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rbf_slices(
    df: pd.DataFrame,
    input_cols: list[str],
    output_col: str,
    out_dir: Path,
    grid_size: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Path:
    """Show simple one-input-at-a-time views of the fake experiment."""
    truth = fit_rbf_truth(df, input_cols, output_col)
    baseline = df[input_cols].median().to_numpy(dtype=float)
    y = df[output_col].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, col in enumerate(input_cols):
        ax = axes[i]
        values = np.linspace(df[col].min(), df[col].max(), grid_size)
        X_slice = np.tile(baseline, (grid_size, 1))
        X_slice[:, i] = values
        y_slice = truth(X_slice)
        noisy_idx = np.linspace(0, grid_size - 1, min(32, grid_size), dtype=int)
        y_slice_noisy = noisy_response(y_slice[noisy_idx], rng, noise_std)

        ax.scatter(df[col], y, s=18, alpha=0.35, color="#4c78a8", label="FLIPMM rows")
        ax.plot(values, y_slice, color="#d62728", linewidth=2.1, label="Clean RBF fake lab")
        if noise_std > 0:
            ax.scatter(
                values[noisy_idx],
                y_slice_noisy,
                s=18,
                alpha=0.8,
                color="#f28e2b",
                label="Noisy queried response",
            )
        ax.axvline(baseline[i], color="black", linestyle=":", linewidth=1.0)
        ax.set_xlabel(col)
        ax.set_ylabel(output_col)
        ax.set_title(f"Vary {col}; hold others at median")
        ax.grid(alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("One-dimensional views of the RBF test environment", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    path = out_dir / "rbf_fake_lab_1d_slices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_strategy_comparison(trace_path: Path, out_dir: Path) -> Path | None:
    """Plot the current benchmark traces, if they have been generated."""
    if not trace_path.exists():
        return None

    traces = pd.read_csv(trace_path)
    summary = (
        traces.groupby(["strategy", "n_observed"], as_index=False)
        .agg(
            r2_mean=("r2_on_rbf_truth", "mean"),
            r2_std=("r2_on_rbf_truth", "std"),
            rmse_mean=("rmse_on_rbf_truth", "mean"),
        )
        .fillna(0.0)
        .sort_values(["strategy", "n_observed"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for strategy, group in summary.groupby("strategy"):
        ax.plot(group["n_observed"], group["r2_mean"], marker="o", linewidth=1.8, label=strategy)
        ax.fill_between(
            group["n_observed"],
            group["r2_mean"] - group["r2_std"],
            group["r2_mean"] + group["r2_std"],
            alpha=0.12,
        )

    ax.set_xlabel("RBF fake-lab samples observed")
    ax.set_ylabel("R2 against held-out RBF samples")
    ax.set_title("How each next-point strategy learns the fake lab")
    ax.set_ylim(-1.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    path = out_dir / "strategy_comparison_on_rbf_fake_lab.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE), help="Benchmark trace CSV to summarize.")
    parser.add_argument("--grid-size", type=int, default=120, help="Resolution for 1D RBF slices.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation to visualize around RBF fake-lab observations.",
    )
    parser.add_argument("--noise-seed", type=int, default=50, help="Random seed for visualized noise.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, input_cols, output_col = load_data()
    rng = np.random.default_rng(args.noise_seed)

    written = [
        plot_rbf_parity(df, input_cols, output_col, OUT_DIR, args.noise_std, rng),
        plot_rbf_slices(df, input_cols, output_col, OUT_DIR, args.grid_size, args.noise_std, rng),
    ]

    comparison = plot_strategy_comparison(Path(args.trace), OUT_DIR)
    if comparison is not None:
        written.append(comparison)

    print("Wrote RBF test-environment visualization files:")
    for path in written:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
