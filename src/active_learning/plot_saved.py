"""Create disagreement plots from a saved active-learning checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ..config_loader import load_config, resolve_path
from .. import modeling
from .experiment import _task_data, build_pca_disagreement_frame, plot_pca_disagreement


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


if __name__ == "__main__":
    main()
