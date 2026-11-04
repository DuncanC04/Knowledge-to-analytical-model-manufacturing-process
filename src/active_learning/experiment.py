"""Active-learning replay built on the existing equation-generation pipeline.

Each task is an ``(output, x_axis)`` pair.  Equations still use all configured
process inputs; the axis only defines the extrapolative pool and makes the
question "where should we measure next?" explicit.  Labels in that pool are
used only after acquisition for retrospective evaluation.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .. import modeling
from ..config_loader import resolve_path
from ..research import single_axis_split
from .metrics import rank_by_disagreement


def _task_data(frame: pd.DataFrame, input_columns: Sequence[str], output_column: str, axis: str, *, holdout_levels: int, validation_fraction: float, seed: int) -> tuple[dict, object]:
    split = single_axis_split(frame, axis, holdout_levels=holdout_levels, validation_fraction=validation_fraction, random_state=seed)

    def extract(indices: Sequence[int]) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        subset = frame.loc[list(indices)]
        return tuple(subset[column].to_numpy(float) for column in input_columns), subset[output_column].to_numpy(float)

    return {
        "train": extract(split.train_indices),
        "validation": extract(split.validation_indices),
        "test": extract(split.test_indices),
        "train_df": frame.loc[list(split.train_indices)],
    }, split


def pca_target_diagnostic(data: dict) -> dict:
    """Describe X-only PCA and its association with y; never fit equations on PCA.

    PCA is useful for visualizing correlated input directions.  It should not
    be performed on ``[X, y]`` when y is the target: that leaks the response
    into the representation and changes the meaning of the x-axis.
    """
    x, y = data["train"]
    matrix = np.column_stack(x)
    transformed = StandardScaler().fit_transform(matrix)
    pca = PCA().fit(transformed)
    scores = pca.transform(transformed)
    correlations = []
    for index in range(scores.shape[1]):
        correlations.append(float(np.corrcoef(scores[:, index], y)[0, 1]) if np.std(scores[:, index]) else np.nan)
    return {
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "target_correlation_by_component": correlations,
    }


def build_pca_disagreement_frame(
    data: dict,
    predictions: np.ndarray,
    input_columns: Sequence[str],
    output_column: str,
    axis: str,
    *,
    tau: float = 1.0,
) -> pd.DataFrame:
    """Build a plotting table for domain shift and committee disagreement.

    PCA is fit on training X only, then applied to the held-out test pool.
    ``y`` is included for retrospective evaluation; it must not be used by an
    acquisition function. The returned table supports both PCA1-vs-PCA2 and
    PCA1-vs-y plots.
    """
    train_x, _ = data["train"]
    test_x, test_y = data["test"]
    train_matrix = np.column_stack(train_x).astype(float)
    test_matrix = np.column_stack(test_x).astype(float)
    scaler = StandardScaler().fit(train_matrix)
    pca = PCA(n_components=min(2, train_matrix.shape[1])).fit(scaler.transform(train_matrix))
    train_projection = pca.transform(scaler.transform(train_matrix))
    test_projection = pca.transform(scaler.transform(test_matrix))
    ranking = rank_by_disagreement(predictions, metric="paper_ibmd", tau=tau)
    predictions = np.asarray(predictions, dtype=float)
    test_table = pd.DataFrame({
        "domain": "test_pool",
        "pca1": test_projection[:, 0],
        "pca2": test_projection[:, 1] if test_projection.shape[1] > 1 else 0.0,
        "axis_value": test_matrix[:, list(input_columns).index(axis)],
        "y": np.asarray(test_y, dtype=float),
        "y_hat_mean": predictions.mean(axis=0),
        "y_hat_spread": predictions.std(axis=0),
        "paper_cv": ranking.set_index("pool_index")["paper_cv"].reindex(range(len(test_matrix))).to_numpy(),
        "paper_ibmd": ranking.set_index("pool_index")["paper_ibmd"].reindex(range(len(test_matrix))).to_numpy(),
    })
    train_table = pd.DataFrame({
        "domain": "train",
        "pca1": train_projection[:, 0],
        "pca2": train_projection[:, 1] if train_projection.shape[1] > 1 else 0.0,
        "axis_value": train_matrix[:, list(input_columns).index(axis)],
        "y": np.asarray(data["train"][1], dtype=float),
        "y_hat_mean": np.nan,
        "y_hat_spread": np.nan,
        "paper_cv": np.nan,
        "paper_ibmd": np.nan,
    })
    return pd.concat([train_table, test_table], ignore_index=True)


def plot_pca_disagreement(
    view: pd.DataFrame,
    output_dir: str | Path,
    *,
    output_label: str = "MRR",
    top_queries: int = 5,
) -> list[Path]:
    """Save input-domain and PCA1-vs-output disagreement figures."""
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    test = view[view["domain"] == "test_pool"].copy()
    top = test.nlargest(top_queries, "paper_ibmd", keep="all")
    paths = []

    fig, ax = plt.subplots(figsize=(7, 5))
    train = view[view["domain"] == "train"]
    ax.scatter(train["pca1"], train["pca2"], color="lightgray", label="Training domain")
    points = ax.scatter(test["pca1"], test["pca2"], c=test["paper_ibmd"], cmap="viridis", s=35, label="Held-out pool")
    ax.scatter(top["pca1"], top["pca2"], facecolors="none", edgecolors="red", s=110, label="Top IBMD queries")
    ax.set(xlabel="PCA 1 of input space", ylabel="PCA 2 of input space", title="Input-domain shift and committee disagreement")
    ax.legend()
    fig.colorbar(points, ax=ax, label="IBMD")
    fig.tight_layout()
    path = destination / "pca_domain_shift_disagreement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train["pca1"], train["y"], color="lightgray", label="Observed training MRR")
    points = ax.scatter(test["pca1"], test["y"], c=test["paper_ibmd"], cmap="viridis", s=35, label="Held-out MRR")
    ax.scatter(test["pca1"], test["y_hat_mean"], marker="x", color="black", label="Committee mean ŷ")
    ax.vlines(test["pca1"], test["y_hat_mean"] - test["y_hat_spread"], test["y_hat_mean"] + test["y_hat_spread"], color="black", alpha=0.2)
    ax.set(xlabel="PCA 1 of input space", ylabel=output_label, title=f"{output_label} versus committee prediction")
    ax.legend()
    fig.colorbar(points, ax=ax, label="IBMD")
    fig.tight_layout()
    path = destination / "pca1_vs_output_disagreement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _target_config(cfg: dict, output: dict) -> dict:
    task_cfg = copy.deepcopy(cfg)
    task_cfg["variables"]["output"] = copy.deepcopy(output)
    task_cfg["output_symbol"] = output["symbol"]
    task_cfg["output_column"] = output["column"]
    return task_cfg


def run_active_learning_grid(
    client,
    cfg: dict,
    retrieve: Callable[[], str] | None,
    frame: pd.DataFrame,
    *,
    committee_sizes: Sequence[int] = (3, 5, 9),
    generation_count: int = 20,
    criterion: str = "adjusted_r2",
    holdout_levels: int = 1,
    validation_fraction: float = 0.25,
    seed: int = 50,
    tau: float = 1.0,
    retrieve_factory: Callable[[dict], Callable[[], str]] | None = None,
    target_output: str | None = None,
    all_outputs: bool = False,
    use_multistage: bool = True,
    make_plots: bool = True,
    log=print,
) -> dict:
    """Generate committees and disagreement maps for every output and axis.

    One candidate archive is generated per task, then reused for every
    committee size. This makes committee-size comparisons computationally
    comparable and avoids treating repeated LLM generations as independent
    measurements.
    """
    sizes = sorted({int(size) for size in committee_sizes})
    if not sizes or sizes[0] < 1:
        raise ValueError("committee_sizes must contain positive integers")
    results = {}
    checkpoint_path = resolve_path(cfg, cfg["output"]["dir"]) / "active_learning" / "grid_checkpoint.json"
    outputs = cfg["variables"].get("outputs", [cfg["variables"]["output"]])
    if not all_outputs and target_output is None:
        target_output = cfg["variables"]["output"]["symbol"]
    if not all_outputs:
        outputs = [item for item in outputs if target_output in {item["symbol"], item["name"], item["column"]}]
        if not outputs:
            raise ValueError(f"Unknown configured output: {target_output}")
    inputs = cfg["variables"]["inputs"]
    input_columns = [item["column"] for item in inputs]
    for output in outputs:
        for axis in input_columns:
            task_id = f"{output['symbol']}__{axis}"
            task_cfg = _target_config(cfg, output)
            task_retrieve = retrieve_factory(task_cfg) if retrieve_factory is not None else retrieve
            if task_retrieve is None:
                raise ValueError("Provide retrieve or retrieve_factory for literature-conditioned generation")
            data, split = _task_data(frame, input_columns, output["column"], axis, holdout_levels=holdout_levels, validation_fraction=validation_fraction, seed=seed)
            log(f"[active learning] {task_id}: running {'Phase 1 -> Phase 3' if use_multistage else 'initial'} generation")
            if use_multistage:
                summary_p1, phase1_results = modeling.phase1(client, task_cfg, task_retrieve, data, log=log)
                _, raw_candidates = modeling.phase3(client, task_cfg, task_retrieve, data, summary_p1, list(phase1_results), log=log)
                fitted_candidates = []
                failures = []
                for index, candidate in enumerate(raw_candidates):
                    try:
                        fitted_candidates.append(modeling.fit_candidate_record(
                            candidate["model_code"], data, candidate_id=f"{output['symbol']}-{index:03d}"
                        ))
                    except Exception as exc:
                        failures.append({"candidate_id": index, "error": type(exc).__name__, "message": str(exc)})
                archive = {"all_candidates": fitted_candidates, "failures": failures, "criterion": criterion, "use_multistage": True}
            else:
                _, archive = modeling.generate_and_select_committee(
                    client, task_cfg, task_retrieve, data, use_literature=True,
                    generation_count=generation_count, committee_size=max(sizes),
                    criterion=criterion, seed=seed,
                    selection_config=task_cfg.get("active_learning", {}), log=log,
                )
            candidates = archive["all_candidates"]
            task_result = {
                "output": output,
                "axis": axis,
                "split": {"train": list(split.train_indices), "validation": list(split.validation_indices), "test": list(split.test_indices)},
                "pca_diagnostic": pca_target_diagnostic(data),
                "archive": archive,
                "committee_sizes": {},
            }
            for size in sizes:
                committee = modeling.select_committee(candidates, size, criterion, task_cfg.get("active_learning", {}))
                if not committee:
                    continue
                evaluation = modeling.evaluate_committee_on_test(committee, data)
                predictions = np.asarray(evaluation["predictions"], dtype=float)
                disagreement = rank_by_disagreement(predictions, metric="paper_ibmd", tau=tau)
                disagreement_view = build_pca_disagreement_frame(
                    data, predictions, input_columns, output["column"], axis, tau=tau
                )
                if make_plots:
                    plot_paths = plot_pca_disagreement(
                        disagreement_view,
                        resolve_path(cfg, cfg["output"]["dir"]) / "active_learning" / "plots" / task_id / f"committee_{size}",
                        output_label=output["name"],
                    )
                else:
                    plot_paths = []
                task_result["committee_sizes"][str(size)] = {
                    "committee": [item["candidate_id"] for item in committee],
                    "evaluation": evaluation,
                    "pca_disagreement_view": disagreement_view.to_dict("records"),
                    "plot_paths": [str(path) for path in plot_paths],
                    "top_ibmd_queries": disagreement.head(min(10, len(disagreement))).to_dict("records"),
                    "top_cv_queries": rank_by_disagreement(predictions, metric="paper_cv", tau=tau).head(min(10, len(disagreement))).to_dict("records"),
                }
            results[task_id] = task_result
            # Persist after every completed output/axis task. This makes an
            # interrupted multi-output run recoverable without regenerating
            # equations that have already been fitted.
            save_active_learning_result(
                {
                    "settings": {
                        "committee_sizes": sizes,
                        "generation_count": generation_count,
                        "criterion": criterion,
                        "holdout_levels": holdout_levels,
                        "seed": seed,
                        "tau": tau,
                        "target_output": target_output,
                    },
                    "tasks": results,
                },
                checkpoint_path,
            )
            log(f"[active learning] checkpoint saved: {checkpoint_path}")
    return {"settings": {"committee_sizes": sizes, "generation_count": generation_count, "criterion": criterion, "holdout_levels": holdout_levels, "seed": seed, "tau": tau}, "tasks": results}


def run_active_learning_from_config(client, cfg: dict, retrieve: Callable[[], str] | None = None, **kwargs) -> dict:
    """Load the configured spreadsheet and run :func:`run_active_learning_grid`."""
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    required = [item["column"] for item in cfg["variables"]["inputs"]]
    required.extend(item["column"] for item in cfg["variables"].get("outputs", [cfg["variables"]["output"]]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Configured active-learning columns are missing from the dataset: {missing}")
    return run_active_learning_grid(client, cfg, retrieve, frame, **kwargs)


def save_active_learning_result(result: dict, path: str | Path) -> Path:
    """Write the result and a reusable equation archive.

    The JSON contains the complete candidate records.  In addition, each task
    gets a readable Python file containing every fitted equation and a JSON
    metadata file containing parameters and validation/IC metrics.  These
    files are deliberately written after generation so later analysis does
    not require another LLM call.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    equation_dir = destination.parent / "equations"
    equation_dir.mkdir(parents=True, exist_ok=True)
    for task_id, task in result.get("tasks", {}).items():
        safe_task_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in task_id)
        candidates = task.get("archive", {}).get("all_candidates", [])
        (equation_dir / f"{safe_task_id}_candidates.json").write_text(
            json.dumps(candidates, indent=2, default=str), encoding="utf-8"
        )
        with (equation_dir / f"{safe_task_id}_equations.py").open("w", encoding="utf-8") as stream:
            stream.write(f"# Fitted equation archive for {task_id}\n")
            stream.write("# Parameters and metrics are stored in the matching *_candidates.json file.\n\n")
            function_names = []
            for candidate in candidates:
                function_name = f"{safe_task_id}_candidate_{len(function_names):03d}"
                function_names.append(function_name)
                stream.write(f"# {candidate.get('candidate_id', 'candidate')}\n")
                stream.write(f"# parameters = {candidate.get('parameters', [])}\n")
                stream.write(f"# validation_mse = {candidate.get('validation_mse')}\n")
                stream.write(f"# validation_r2 = {candidate.get('validation_r2')}\n")
                code = candidate.get("model_code", "").strip()
                code = re.sub(r"^def\s+\w+\s*\(", f"def {function_name}(", code, count=1, flags=re.MULTILINE)
                stream.write(code + "\n\n")
            stream.write(f"GENERATED_FUNCTIONS = {function_names!r}\n")
    return destination
