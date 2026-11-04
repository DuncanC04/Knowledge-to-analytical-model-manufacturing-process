"""Focused experiments on sequential and diversity-aware equation disagreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ..config_loader import load_config, resolve_path
from .adaptation import _fit_committee, _matrix, _predict, load_equation_committee


METHODS = ("random", "maximin", "disagreement", "hybrid")
BATCH_METHODS = ("random", "maximin", "disagreement", "diversity_disagreement")
SHIFTS = {
    "ws8_all": lambda df: np.isclose(df["WaterSpeed (m/s)"], 8.0),
    "ws8_high_energy": lambda df: np.isclose(df["WaterSpeed (m/s)"], 8.0) & (df["PulseEnergy (microJ)"] >= 23.33333333333334),
    "ws8_high_frequency": lambda df: np.isclose(df["WaterSpeed (m/s)"], 8.0) & (df["Frequency (kHz)"] >= 12.66666666666667),
    "ws8_high_energy_or_frequency": lambda df: np.isclose(df["WaterSpeed (m/s)"], 8.0) & ((df["PulseEnergy (microJ)"] >= 23.33333333333334) | (df["Frequency (kHz)"] >= 12.66666666666667)),
}


def _scores(method: str, candidate_x: np.ndarray, observed_x: np.ndarray, predictions: np.ndarray, scaler: StandardScaler, rng: np.random.Generator) -> tuple[int, dict]:
    disagreement = predictions.std(axis=0)
    scaled_candidates = scaler.transform(candidate_x)
    scaled_observed = scaler.transform(observed_x)
    distance = np.sqrt(((scaled_candidates[:, None, :] - scaled_observed[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    if method == "random":
        index = int(rng.integers(len(candidate_x)))
        score = np.zeros(len(candidate_x))
    elif method == "maximin":
        score = distance
        index = int(np.argmax(score))
    elif method == "disagreement":
        score = disagreement
        index = int(np.argmax(score))
    elif method in {"hybrid", "diversity_disagreement"}:
        def norm(values: np.ndarray) -> np.ndarray:
            spread = values.max() - values.min()
            return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
        if method == "diversity_disagreement":
            score = 0.3 * norm(distance) + 0.7 * norm(disagreement)
        else:
            score = 0.5 * norm(distance) + 0.5 * norm(disagreement)
        index = int(np.argmax(score))
    else:
        raise ValueError(method)
    return index, {"score": float(score[index]), "disagreement": float(disagreement[index]), "distance": float(distance[index])}


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    return {"rmse": float(mean_squared_error(y_true, prediction) ** 0.5)}


def run_study(
    frame: pd.DataFrame,
    records: list[dict],
    input_columns: list[str],
    output_column: str,
    *,
    source_size: int = 30,
    trials: int = 50,
    sequential_budget: int = 5,
    batch_budget: int = 10,
    target_test_size: int = 10,
    seed: int = 50,
) -> dict:
    source = frame[frame["WaterSpeed (m/s)"] < 8.0].copy()
    rows, ranking_rows, splits = [], [], []
    for shift_name, target_filter in SHIFTS.items():
        target = frame[target_filter(frame)].copy()
        if len(target) < target_test_size + sequential_budget:
            raise ValueError(f"Shift {shift_name} has too few target observations")
        for trial in range(trials):
            rng = np.random.default_rng(seed + trial)
            target_indices = target.index.to_numpy(copy=True); rng.shuffle(target_indices)
            test_indices = target_indices[:target_test_size]
            pool_indices = target_indices[target_test_size:]
            source_indices = rng.choice(source.index.to_numpy(), size=source_size, replace=False)
            test, pool, initial = target.loc[test_indices], target.loc[pool_indices], source.loc[source_indices]
            scaler = StandardScaler().fit(pd.concat([source[input_columns], pool[input_columns]])[input_columns].to_numpy(float))
            splits.append({"shift": shift_name, "trial": trial, "source_indices": [int(i) for i in source_indices], "test_indices": [int(i) for i in test_indices], "pool_indices": [int(i) for i in pool_indices]})
            for method in METHODS:
                observed = initial.copy(); available = pool.copy()
                for k in range(sequential_budget + 1):
                    fitted, parameters, failures = _fit_committee(records, _matrix(observed, input_columns), observed[output_column].to_numpy(float))
                    test_pred = _predict(records, fitted, _matrix(test, input_columns)).mean(axis=0)
                    rows.append({"experiment": "sequential", "shift": shift_name, "trial": trial, "method": method, "acquisitions": k, "rmse": _metrics(test[output_column].to_numpy(float), test_pred)["rmse"], "fit_failures": sum(bool(x) for x in failures)})
                    if k == sequential_budget:
                        break
                    candidate_x = _matrix(available, input_columns)
                    candidate_pred = _predict(records, fitted, candidate_x)
                    mean_pred = candidate_pred.mean(axis=0)
                    true_error = np.abs(available[output_column].to_numpy(float) - mean_pred)
                    score_index, score_info = _scores(method, candidate_x, _matrix(observed, input_columns), candidate_pred, scaler, rng)
                    score_values = candidate_pred.std(axis=0) if method == "disagreement" else np.asarray([score_info["score"]])
                    rho = spearmanr(candidate_pred.std(axis=0), true_error).statistic if len(candidate_pred) > 1 else np.nan
                    top_n = min(10, len(available)); top_error = set(np.argsort(true_error)[-top_n:]); top_score = set(np.argsort(candidate_pred.std(axis=0))[-top_n:])
                    ranking_rows.append({"shift": shift_name, "trial": trial, "method": method, "experiment": "sequential", "acquisitions": k, "spearman_disagreement_error": float(rho) if np.isfinite(rho) else np.nan, "top10_error_enrichment": len(top_error & top_score) / top_n, "selected_error_rank": int(np.sum(true_error > true_error[score_index]) + 1), **score_info})
                    selected_index = available.index.to_numpy()[score_index]
                    observed = pd.concat([observed, available.loc[[selected_index]]]); available = available.drop(index=selected_index)
            for method in BATCH_METHODS:
                observed = initial.copy(); available = pool.copy(); selected = []
                for _ in range(batch_budget):
                    fitted, _, _ = _fit_committee(records, _matrix(initial, input_columns), initial[output_column].to_numpy(float))
                    candidate_x = _matrix(available, input_columns); candidate_pred = _predict(records, fitted, candidate_x)
                    observed_x = _matrix(pd.concat([initial, pool.loc[selected]]) if selected else initial, input_columns)
                    score_index, score_info = _scores(method, candidate_x, observed_x, candidate_pred, scaler, rng)
                    selected_index = available.index.to_numpy()[score_index]; selected.append(selected_index); available = available.drop(index=selected_index)
                acquired = pool.loc[selected]
                adapted = pd.concat([initial, acquired]); fitted, _, failures = _fit_committee(records, _matrix(adapted, input_columns), adapted[output_column].to_numpy(float))
                test_pred = _predict(records, fitted, _matrix(test, input_columns)).mean(axis=0)
                rows.append({"experiment": "batch", "shift": shift_name, "trial": trial, "method": method, "acquisitions": batch_budget, "rmse": _metrics(test[output_column].to_numpy(float), test_pred)["rmse"], "fit_failures": sum(bool(x) for x in failures), "selected_indices": [int(i) for i in selected]})
    return {"settings": {"source_size": source_size, "trials": trials, "sequential_budget": sequential_budget, "batch_budget": batch_budget, "target_test_size": target_test_size, "seed": seed, "shifts": {name: int(SHIFTS[name](frame).sum()) for name in SHIFTS}, "equation_ids": [r.get("candidate_id") for r in records], "protocol": "target labels are used only for locked evaluation and retrospective score-quality diagnostics"}, "splits": splits, "results": rows, "ranking": ranking_rows}


def save_study(result: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(result["results"]).to_csv(output / "results.csv", index=False)
    pd.DataFrame(result["ranking"]).to_csv(output / "ranking_quality.csv", index=False)
    pd.DataFrame(result["splits"]).to_json(output / "splits.json", orient="records", indent=2)
    results = pd.DataFrame(result["results"])
    summary = results.groupby(["experiment", "shift", "method", "acquisitions"])["rmse"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(output / "rmse_summary.csv", index=False)
    paired_rows = []
    for experiment, shift, acquisitions in results[["experiment", "shift", "acquisitions"]].drop_duplicates().itertuples(index=False):
        part = results[(results["experiment"] == experiment) & (results["shift"] == shift) & (results["acquisitions"] == acquisitions)]
        baseline = "random"
        if baseline not in set(part["method"]):
            continue
        base = part.set_index("trial")["rmse"]
        comparisons = [m for m in part["method"].unique() if m != baseline]
        for method in comparisons:
            candidate = part[part["method"] == method].set_index("trial")["rmse"]
            joined = pd.concat([candidate.rename("candidate"), base.rename("baseline")], axis=1).dropna()
            differences = joined["candidate"] - joined["baseline"]
            n = len(differences); mean_diff = float(differences.mean())
            se = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            half = float(student_t.ppf(0.975, n - 1) * se) if n > 1 else np.nan
            paired_rows.append({"experiment": experiment, "shift": shift, "acquisitions": acquisitions, "method": method, "baseline": baseline, "n": n, "mean_rmse_difference": mean_diff, "ci95_low": mean_diff - half if np.isfinite(half) else np.nan, "ci95_high": mean_diff + half if np.isfinite(half) else np.nan})
    pd.DataFrame(paired_rows).to_csv(output / "paired_rmse_comparisons.csv", index=False)
    ranking = pd.DataFrame(result["ranking"])
    if not ranking.empty:
            ranking.groupby(["experiment", "shift", "method", "acquisitions"])[["spearman_disagreement_error", "top10_error_enrichment", "selected_error_rank"]].mean().reset_index().to_csv(output / "ranking_summary.csv", index=False)
    report_lines = [
        "# Focused disagreement study",
        "",
        "This study reuses the fixed 20-equation literature-informed committee. Source observations have WaterSpeed < 8 m/s; target observations have WaterSpeed = 8 m/s. Target labels are used only for locked evaluation and retrospective ranking diagnostics.",
        "",
        f"- Trials: {result['settings']['trials']} per shift and method",
        f"- Source size: {result['settings']['source_size']}",
        f"- Sequential budget: {result['settings']['sequential_budget']}; batch budget: {result['settings']['batch_budget']}",
        f"- Equations: {len(result['settings']['equation_ids'])}",
        "",
        "## Target shifts",
        "",
        "| Shift | Target observations |",
        "|---|---:|",
    ]
    report_lines.extend(f"| {name} | {count} |" for name, count in result["settings"]["shifts"].items())
    report_lines.extend(["", "## Interpretation convention", "", "For paired RMSE differences, negative values favor the listed method over random sampling. Confidence intervals are paired t-intervals over split/trial replicates; they describe conditional split-based uncertainty, not independent physical experiments.", "", "See `rmse_summary.csv`, `paired_rmse_comparisons.csv`, `ranking_summary.csv`, and the shift-specific PNG figures for full results."])
    (output / "study_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        for shift in results["shift"].unique():
            part = summary[(summary["shift"] == shift) & (summary["experiment"] == "sequential")]
            fig, ax = plt.subplots(figsize=(9, 5))
            for method, group in part.groupby("method"):
                ax.plot(group.acquisitions, group["mean"], marker="o", label=method)
            ax.set(title=f"Sequential disagreement study — {shift}", xlabel="Acquisitions", ylabel="Target RMSE"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / f"sequential_{shift}.png", dpi=200); plt.close(fig)
        ranking_summary = pd.read_csv(output / "ranking_summary.csv") if (output / "ranking_summary.csv").exists() else pd.DataFrame()
        if not ranking_summary.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            part = ranking_summary[ranking_summary["experiment"] == "sequential"]
            for method, group in part.groupby("method"):
                axes[0].plot(group.acquisitions, group.spearman_disagreement_error, marker="o", label=method)
                axes[1].plot(group.acquisitions, group.top10_error_enrichment, marker="o", label=method)
            axes[0].axhline(0, color="black", lw=.8); axes[0].set_ylabel("Spearman(disagreement, error)"); axes[1].set_ylabel("Top-10 error enrichment");
            for ax in axes: ax.set_xlabel("Acquisition step"); ax.grid(alpha=.25)
            axes[1].legend(); fig.tight_layout(); fig.savefig(output / "disagreement_error_ranking_quality.png", dpi=200); plt.close(fig)
    except ImportError:
        pass
    return output / "experiment.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused equation-disagreement experiments.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--equations", default="output/active_learning/equations/MRR__WaterSpeed__m_s__candidates.json")
    parser.add_argument("--output-dir", default="output/disagreement_study")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--source-size", type=int, default=30)
    args = parser.parse_args()
    cfg = load_config(args.config); frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    records = load_equation_committee(resolve_path(cfg, args.equations), count=20)
    result = run_study(frame, records, cfg["input_columns"], cfg["output_column"], source_size=args.source_size, trials=args.trials, seed=50)
    print(f"Saved {save_study(result, resolve_path(cfg, args.output_dir))}")


if __name__ == "__main__":
    main()
