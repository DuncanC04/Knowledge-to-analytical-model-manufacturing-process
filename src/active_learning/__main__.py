"""Command-line runner for the axis-wise active-learning experiment."""

from __future__ import annotations

import argparse
import os

import openai

from ..config_loader import load_config, require_env, resolve_path
from ..knowledge import build_retriever
from .experiment import run_active_learning_from_config, save_active_learning_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate equation committees and rank disagreement queries.")
    parser.add_argument("--config", default=None, help="Path to config YAML; defaults to config/config.yaml")
    parser.add_argument("--generation-count", type=int, default=None)
    parser.add_argument("--committee-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--holdout-levels", type=int, default=1)
    parser.add_argument("--target-output", default=None, help="Configured output symbol/name/column; defaults to variables.output")
    parser.add_argument("--all-outputs", action="store_true", help="Run every configured output instead of only variables.output")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    active_cfg = cfg.get("active_learning", {})
    generation_count = args.generation_count or int(active_cfg.get("generation_count", 20))
    committee_sizes = args.committee_sizes or list(active_cfg.get("committee_sizes", [3, 5, 9]))
    criterion = str(active_cfg.get("selection_criterion", "adjusted_r2"))
    target_output = args.target_output or cfg["variables"]["output"]["symbol"]
    client_kwargs = {"api_key": require_env("OPENAI_API_KEY")}
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)
    result = run_active_learning_from_config(
        client,
        cfg,
        retrieve_factory=build_retriever,
        committee_sizes=committee_sizes,
        generation_count=generation_count,
        criterion=criterion,
        target_output=target_output,
        all_outputs=args.all_outputs,
        use_multistage=True,
        holdout_levels=args.holdout_levels,
    )
    output = args.output or str(resolve_path(cfg, cfg["output"]["dir"]) / "active_learning" / "grid.json")
    destination = save_active_learning_result(result, output)
    print(f"Active-learning result written to {destination}")


if __name__ == "__main__":
    main()
