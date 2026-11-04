"""Research-group visual and tabular report for adaptation experiments."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASELINES = ("random", "maximin", "gp_uncertainty", "hybrid")


def _holm(p_values: pd.Series) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (m - rank)))
        adjusted[index] = running
    return adjusted


def paired_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ensemble in sorted(curves["ensemble"].unique()):
        for source_size in sorted(curves["source_size"].unique()):
            for k in (1, 2, 3, 10):
                disagreement = curves.query("ensemble == @ensemble and source_size == @source_size and method == 'disagreement' and acquisitions == @k")
                for baseline in BASELINES:
                    base = curves.query("ensemble == @ensemble and source_size == @source_size and method == @baseline and acquisitions == @k")
                    joined = disagreement.merge(base, on=["trial", "source_size", "acquisitions", "ensemble"], suffixes=("_d", "_b"))
                    difference = joined["rmse_d"] - joined["rmse_b"]
                    mean = float(difference.mean())
                    sem = float(stats.sem(difference)) if len(difference) > 1 else np.nan
                    t_critical = float(stats.t.ppf(0.975, len(difference) - 1)) if len(difference) > 1 else np.nan
                    try:
                        p_value = float(stats.ttest_1samp(difference, 0).pvalue)
                    except Exception:
                        p_value = np.nan
                    rows.append({"ensemble": ensemble, "source_size": source_size, "acquisitions": k, "baseline": baseline, "n_trials": len(difference), "mean_difference_disagreement_minus_baseline": mean, "ci95_low": mean - t_critical * sem, "ci95_high": mean + t_critical * sem, "win_rate_disagreement": float((difference < 0).mean()), "p_value": p_value})
    result = pd.DataFrame(rows)
    result["p_holm"] = _holm(result["p_value"].fillna(1.0))
    result["favors_disagreement"] = result["ci95_high"] < 0
    return result


def curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    grouped = curves.groupby(["ensemble", "source_size", "method", "acquisitions"])["rmse"]
    result = grouped.agg(mean="mean", std="std", n="count").reset_index()
    result["sem"] = result["std"] / np.sqrt(result["n"])
    result["ci95_low"] = result["mean"] - 1.96 * result["sem"]
    result["ci95_high"] = result["mean"] + 1.96 * result["sem"]
    return result


def _save_table(table: pd.DataFrame, output: Path, name: str) -> None:
    table.to_csv(output / f"{name}.csv", index=False)
    table.to_markdown(output / f"{name}.md", index=False, floatfmt=".3f")


def _plot_learning_curves(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    for ensemble in sorted(summary.ensemble.unique()):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
        for ax, source_size in zip(axes.flat, sorted(summary.source_size.unique())):
            part = summary.query("ensemble == @ensemble and source_size == @source_size")
            for method, group in part.groupby("method"):
                ax.plot(group.acquisitions, group["mean"], marker="o", linewidth=2, markersize=3, label=method)
                ax.fill_between(group.acquisitions, group.ci95_low, group.ci95_high, alpha=.12)
            ax.set_title(f"Source n={source_size}")
            ax.set_xlabel("Target observations acquired")
            ax.set_ylabel("Target RMSE")
            ax.axvspan(0.5, 3.5, color="gold", alpha=.08)
            ax.grid(alpha=.25)
        handles, labels = axes.flat[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        fig.suptitle(f"Target adaptation learning curves — {ensemble} committee", y=.98)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output / f"learning_curves_{ensemble}.png", dpi=220)
        plt.close(fig)


def _plot_paired_heatmap(paired: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    early = paired[paired.acquisitions.isin([1, 2, 3])].copy()
    for baseline in BASELINES:
        part = early[early.baseline == baseline]
        matrix = part.pivot_table(index=["ensemble", "source_size"], columns="acquisitions", values="mean_difference_disagreement_minus_baseline")
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-max(abs(matrix.min().min()), abs(matrix.max().max())), vmax=max(abs(matrix.min().min()), abs(matrix.max().max())))
        ax.set_xticks(range(len(matrix.columns)), labels=[str(x) for x in matrix.columns])
        ax.set_yticks(range(len(matrix.index)), labels=[f"{ensemble}, n={size}" for ensemble, size in matrix.index])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="RMSE difference (disagreement − baseline)")
        ax.set_title(f"Early paired advantage of disagreement vs {baseline}")
        ax.set_xlabel("Acquisitions"); ax.set_ylabel("Committee / source size")
        fig.tight_layout(); fig.savefig(output / f"early_heatmap_vs_{baseline}.png", dpi=220); plt.close(fig)


def _plot_relative_reduction(curves: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    baseline = curves[curves.acquisitions == 0][["trial", "ensemble", "source_size", "method", "rmse"]].rename(columns={"rmse": "rmse0"})
    joined = curves.merge(baseline, on=["trial", "ensemble", "source_size", "method"], suffixes=("", "_baseline"))
    joined["reduction_pct"] = 100 * (joined.rmse0 - joined.rmse) / joined.rmse0
    summary = joined.groupby(["ensemble", "source_size", "method", "acquisitions"])["reduction_pct"].mean().reset_index()
    summary.to_csv(output / "relative_rmse_reduction.csv", index=False)
    for ensemble in sorted(summary.ensemble.unique()):
        fig, ax = plt.subplots(figsize=(11, 6))
        part = summary.query("ensemble == @ensemble")
        for method, group in part.groupby("method"):
            ax.plot(group.acquisitions, group.reduction_pct, marker="o", label=method)
        ax.axvspan(.5, 3.5, color="gold", alpha=.1)
        ax.set(title=f"RMSE reduction relative to source-only — {ensemble}", xlabel="Acquisitions", ylabel="Mean RMSE reduction (%)")
        ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(output / f"relative_reduction_{ensemble}.png", dpi=220); plt.close(fig)


def _plot_trial_curves(curves: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    focus = curves.query("ensemble == 'all20' and source_size == 30")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, method in zip(axes, ["disagreement", "random"]):
        part = focus[focus.method == method]
        for trial, group in part.groupby("trial"):
            ax.plot(group.acquisitions, group.rmse, color="tab:blue" if method == "disagreement" else "tab:gray", alpha=.18)
        mean = part.groupby("acquisitions").rmse.mean()
        ax.plot(mean.index, mean.values, color="tab:blue" if method == "disagreement" else "black", linewidth=3)
        ax.set_title(f"{method}: all20, source n=30"); ax.set_xlabel("Acquisitions"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Target RMSE")
    fig.suptitle("Individual trial learning curves: variability behind the mean", y=.98)
    fig.tight_layout(); fig.savefig(output / "trial_curves_all20_source30.png", dpi=220); plt.close(fig)


def _plot_acquisition_locations(curves: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for _, row in curves.dropna(subset=["selected_inputs"]).iterrows():
        try:
            inputs = ast.literal_eval(row["selected_inputs"])
        except (ValueError, SyntaxError):
            continue
        rows.append({"method": row.method, "ensemble": row.ensemble, "acquisitions": row.acquisitions + 1, "P": inputs.get("PulseEnergy (microJ)"), "F": inputs.get("Frequency (kHz)"), "SS": inputs.get("ScanningSpeed (mm/s)")})
    locations = pd.DataFrame(rows)
    locations.to_csv(output / "selected_acquisition_locations.csv", index=False)
    focus = locations.query("ensemble == 'all20' and acquisitions <= 3")
    methods = ["random", "maximin", "gp_uncertainty", "disagreement", "hybrid"]
    fig, axes = plt.subplots(1, len(methods), figsize=(18, 4), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        part = focus[focus.method == method]
        scatter = ax.scatter(part.P, part.F, c=part.SS, s=18, alpha=.35, cmap="viridis")
        ax.set_title(method); ax.set_xlabel("Pulse energy")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Frequency"); fig.colorbar(scatter, ax=axes, label="Scanning speed")
    fig.suptitle("Where methods sample the target domain during the first three acquisitions — all20")
    fig.tight_layout(); fig.savefig(output / "acquisition_locations_all20_early.png", dpi=220); plt.close(fig)


def generate_report(experiment_path: str | Path, output_dir: str | Path) -> Path:
    experiment_path = Path(experiment_path)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    data = json.loads(experiment_path.read_text(encoding="utf-8"))
    curves = pd.read_csv(experiment_path.parent / "learning_curves.csv")
    summaries = curve_summary(curves)
    paired = paired_table(curves)
    _save_table(summaries, output, "rmse_summary")
    _save_table(paired, output, "paired_early_comparisons")
    _plot_learning_curves(summaries, output)
    _plot_paired_heatmap(paired, output)
    _plot_relative_reduction(curves, output)
    _plot_trial_curves(curves, output)
    _plot_acquisition_locations(curves, output)
    committee_table = pd.DataFrame([{"ensemble": name, "committee_size": len(ids), "equation_ids": ", ".join(ids)} for name, ids in data["settings"]["ensembles"].items()])
    _save_table(committee_table, output, "committee_definitions")
    report = output / "research_group_report.md"
    report.write_text("""# FLIPMM source-to-target adaptation: research-group report\n\n## What this report tests\n\nThe experiment compares whether disagreement among fixed literature-informed equations provides an early sample-efficiency advantage during adaptation to the water-speed = 8 m/s target domain. All ensemble types use identical trial splits and acquisition methods within each trial.\n\n## Ensemble definitions\n\nSee `committee_definitions.md`. The `diverse5` and `diverse9` committees were selected from source-validation prediction diversity and intentionally include lower-adjusted-R² candidates.\n\n## Recommended presentation order\n\n1. `learning_curves_all20.png`: primary result; emphasizes whether allowing less accurate models strengthens early disagreement.\n2. `early_heatmap_vs_random.png`: compact overview of paired disagreement advantages at acquisitions 1–3. Negative values favor disagreement.\n3. `trial_curves_all20_source30.png`: shows whether the mean advantage is consistent across trials.\n4. `relative_reduction_all20.png`: communicates practical improvement relative to the source-only model.\n5. `acquisition_locations_all20_early.png`: shows whether strategies investigate different target regions.\n\n## Statistical interpretation\n\nThe paired tables use within-trial RMSE differences, 95% t intervals, and Holm-adjusted p-values over the reported comparison family. These intervals quantify conditional uncertainty from repeated dataset splits and are not independent physical experimental replicates.\n\nA negative disagreement-minus-baseline difference favors disagreement. Focus interpretation on acquisitions 1–3; acquisition 10 is a convergence diagnostic rather than the primary claim.\n\n## Reproducibility\n\nThe source experiment is stored in the sibling `adaptation_ensembles/experiment.json`, including splits, selected inputs, fitted parameters, and failure counts.\n""", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research-group tables and figures from an adaptation experiment.")
    parser.add_argument("--experiment", default="output/adaptation_ensembles/experiment.json")
    parser.add_argument("--output-dir", default="output/adaptation_report")
    args = parser.parse_args()
    print(generate_report(args.experiment, args.output_dir))


if __name__ == "__main__":
    main()
