"""Create disagreement plots from a saved active-learning checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from ..config_loader import load_config, resolve_path
from .. import modeling
from .experiment import _task_data, build_pca_disagreement_frame, plot_pca_disagreement


def plot_four_input_slices(
    frame: pd.DataFrame,
    candidates: list[dict],
    input_variables: list[dict],
    output: dict,
    test_indices: list[int],
    output_path: Path,
) -> Path:
    """Plot true observations and saved model slices for every input."""
    import matplotlib.pyplot as plt

    input_columns = [item["column"] for item in input_variables]
    train_indices = [index for index in frame.index if index not in set(test_indices)]
    baseline = frame.loc[train_indices, input_columns].to_numpy(float).mean(axis=0)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(candidates), 1)))
    fig, axes = plt.subplots(len(input_variables), 1, figsize=(10, 13), sharey=True)
    axes = np.atleast_1d(axes)

    for axis_index, (axis_variable, ax) in enumerate(zip(input_variables, axes)):
        column = axis_variable["column"]
        x_values = np.linspace(frame[column].min(), frame[column].max(), 250)
        model_inputs = np.tile(baseline, (len(x_values), 1))
        model_inputs[:, axis_index] = x_values
        model_x = tuple(model_inputs[:, index] for index in range(model_inputs.shape[1]))

        ax.scatter(
            frame.loc[train_indices, column],
            frame.loc[train_indices, output["column"]],
            color="black", s=18, alpha=0.55, label="Training observations",
        )
        if test_indices:
            ax.scatter(
                frame.loc[test_indices, column],
                frame.loc[test_indices, output["column"]],
                color="tab:orange", edgecolors="black", s=28, alpha=0.8, label="Held-out observations",
            )

        for candidate, color in zip(candidates, colors):
            function = modeling._extract_function(candidate["model_code"])
            with np.errstate(all="ignore"):
                prediction = np.asarray(function(model_x, *candidate["parameters"]), dtype=float)
            ax.plot(x_values, prediction, color=color, linewidth=1.7, alpha=0.85, label=candidate["candidate_id"])

        ax.set_xlabel(axis_variable["name"])
        ax.set_ylabel(output["name"])
        ax.grid(alpha=0.25)
        ax.set_title(f"{output['name']} vs {axis_variable['name']}\n(other inputs fixed at training means)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Saved equation committee: {output['name']}", y=1.02, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved active-learning equations without regeneration.")
    parser.add_argument("--checkpoint", default="output/active_learning/grid_checkpoint.json")
    parser.add_argument("--task-id", default="MRR__WaterSpeed (m/s)")
    parser.add_argument("--committee-size", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    checkpoint_path = resolve_path(cfg, args.checkpoint)
    result = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    task = result["tasks"][args.task_id]
    output = task["output"]
    axis = task["axis"]
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    input_columns = [item["column"] for item in cfg["variables"]["inputs"]]
    data, _ = _task_data(frame, input_columns, output["column"], axis, holdout_levels=result["settings"].get("holdout_levels", 1), validation_fraction=0.25, seed=result["settings"].get("seed", 50))
    candidates = task["archive"]["all_candidates"]
    selected_ids = task["committee_sizes"][str(args.committee_size)]["committee"]
    selected = [candidate for candidate in candidates if candidate["candidate_id"] in selected_ids]
    evaluation = modeling.evaluate_committee_on_test(selected, data)
    view = build_pca_disagreement_frame(data, evaluation["predictions"], input_columns, output["column"], axis, tau=result["settings"].get("tau", 1.0))
    safe_task = re.sub(r"[^A-Za-z0-9_-]", "_", args.task_id)
    output_dir = resolve_path(cfg, cfg["output"]["dir"]) / "active_learning" / "plots" / safe_task / f"committee_{args.committee_size}"
    for path in plot_pca_disagreement(view, output_dir, output_label=output["name"]):
        print(f"Wrote: {path}")
    slice_path = resolve_path(cfg, cfg["output"]["dir"]) / "active_learning" / "plots" / safe_task / f"committee_{args.committee_size}" / "mrr_four_input_slices.png"
    print(f"Wrote: {plot_four_input_slices(frame, selected, cfg['variables']['inputs'], output, list(task['split']['test']), slice_path)}")


if __name__ == "__main__":
    main()
