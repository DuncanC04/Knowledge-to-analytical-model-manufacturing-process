"""End-to-end pipeline: knowledge -> initial equations -> refinement -> output.

This ties the modules together and writes the result text files into the
configured output directory.
"""

from pathlib import Path
import json
import numpy as np

import openai

from .config_loader import load_config, require_env, resolve_path
from .dataset import load_and_split, split_summary
from .knowledge import build_retriever
from . import modeling
from .research import committee_disagreement


def _write_equation_file(path: Path, title: str, best: dict, test_mse, test_r2):
    """Write a single best-equation report (code + validation + test metrics)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(best["model_code"].strip() + "\n\n")
        f.write(f"Validation MSE: {best['validation_mse']}\n")
        f.write(f"Validation R^2: {best['validation_r2']}\n")
        if test_mse is not None:
            f.write(f"Test MSE: {test_mse}\n")
            f.write(f"Test R^2: {test_r2}\n")


def run(config_path=None, log=print) -> dict:
    """Run the full pipeline. Returns a dict with the key results."""
    cfg = load_config(config_path)
    out_cfg = cfg["output"]
    out_dir = resolve_path(cfg, out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    client = openai.OpenAI(api_key=require_env("OPENAI_API_KEY"))

    # --- 1. Knowledge extraction (RAG over the literature) -------------------
    log(f"Topic: {cfg['topic']}")
    log("Building the knowledge retriever ...")
    retrieve = build_retriever(cfg)

    # --- 2. Dataset ----------------------------------------------------------
    log("Loading and splitting the dataset ...")
    data = load_and_split(cfg)
    log(split_summary(data))

    # --- 3. Phase 1: initial equation generation -----------------------------
    log("=== Phase 1: initial equation generation ===")
    summary_p1, results_p1 = modeling.phase1(client, cfg, retrieve, data, log=log)
    best_initial = results_p1[0]
    init_test_mse, init_test_r2, _ = modeling.evaluate_on_test(results_p1, data)

    initial_path = out_dir / out_cfg["initial_equation_file"]
    _write_equation_file(
        initial_path, "Initial best equation (Phase 1)",
        best_initial, init_test_mse, init_test_r2,
    )
    log(f"Initial equation written to {initial_path}")

    # --- 4. Phase 3: refinement ----------------------------------------------
    log("=== Phase 3: refinement ===")
    summary_p3, results_p3 = modeling.phase3(
        client, cfg, retrieve, data, summary_p1, list(results_p1), log=log
    )
    best_final = results_p3[0]
    final_test_mse, final_test_r2, test_summary = modeling.evaluate_on_test(results_p3, data)

    final_path = out_dir / out_cfg["final_equation_file"]
    _write_equation_file(
        final_path, "Final best equation (after refinement)",
        best_final, final_test_mse, final_test_r2,
    )
    log(f"Final equation written to {final_path}")

    # --- 5. Full run summary -------------------------------------------------
    summary_path = out_dir / out_cfg["summary_file"]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Topic: {cfg['topic']}\n")
        f.write(f"{split_summary(data)}\n\n")
        f.write("==================== PHASE 1 (INITIAL) ====================\n")
        f.write(summary_p1 + "\n")
        f.write("==================== PHASE 3 (REFINED) ====================\n")
        f.write(summary_p3 + "\n")
        f.write("==================== TEST RESULTS =========================\n")
        f.write(test_summary + "\n")
    log(f"Run summary written to {summary_path}")

    return {
        "initial": best_initial,
        "final": best_final,
        "initial_test": (init_test_mse, init_test_r2),
        "final_test": (final_test_mse, final_test_r2),
        "output_dir": out_dir,
    }


def run_literature_comparison(config_path=None, log=print) -> dict:
    """Compare identical equation-generation budgets with and without literature.

    Literature is allowed to influence candidate generation only. Both arms
    use the same data split, generation count, information-criterion rule, and
    committee size. Predictions are evaluated only after candidate selection.
    """
    cfg = load_config(config_path)
    out_dir = resolve_path(cfg, cfg["output"]["dir"]) / "literature_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(api_key=require_env("OPENAI_API_KEY"))
    data = load_and_split(cfg)
    comparison_cfg = cfg.get("reliability_comparison", {})
    generation_count = int(comparison_cfg.get("generation_count", 20))
    committee_size = int(comparison_cfg.get("committee_size", 9))
    criterion = str(comparison_cfg.get("selection_criterion", "bic"))
    seed = int(comparison_cfg.get("seed", 50))
    arms = {}
    literature_context = None
    for use_literature in (False, True):
        if use_literature:
            if literature_context is None:
                log("Building local literature retriever and retrieving one fixed context ...")
                literature_context = build_retriever(cfg)()
            arm_retrieve = lambda context=literature_context: context
        else:
            arm_retrieve = lambda: ""
        selected, archive = modeling.generate_and_select_committee(
            client, cfg, arm_retrieve, data,
            use_literature=use_literature,
            generation_count=generation_count,
            committee_size=committee_size,
            criterion=criterion,
            seed=seed,
            log=log,
        )
        archive_payload = {"selected": selected, "archive": archive}
        (out_dir / f"{'literature' if use_literature else 'no_literature'}_archive.json").write_text(
            json.dumps(archive_payload, indent=2, default=str), encoding="utf-8"
        )
        if not selected:
            raise ValueError(
                f"No finite candidates survived the {('literature' if use_literature else 'no-literature')} arm. "
                f"Inspect {out_dir} archive for generation failures."
            )
        evaluation = modeling.evaluate_committee_on_test(selected, data)
        disagreement = committee_disagreement(np.asarray(evaluation["predictions"]), tau=1.0)
        arms["literature" if use_literature else "no_literature"] = {
            "selected_candidates": selected,
            "archive": archive,
            "evaluation": evaluation,
            "disagreement": disagreement.to_dict(orient="records"),
        }
    result = {
        "comparison_contract": {
            "same_data_split": True,
            "same_generation_count": generation_count,
            "same_committee_size": committee_size,
            "selection_criterion": criterion,
            "literature_changes_predictions_via_generation": True,
            "literature_is_not_a_calibrated_probability": True,
        },
        "arms": arms,
        "output_path": str(out_dir / "comparison.json"),
    }
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log(f"Literature comparison written to {out_dir / 'comparison.json'}")
    return result
