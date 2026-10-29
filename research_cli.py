"""Command-line audit and evaluation entry points for the research plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.research import (
    audit_table,
    corpus_manifest,
    make_data_dictionary,
    development_threshold_grid,
    task_manifests,
    single_axis_split,
    rank_metrics,
)


def _paths(cfg: dict) -> tuple[Path, Path]:
    data_path = resolve_path(cfg, cfg["dataset"]["file"])
    corpus_path = resolve_path(cfg, "data/literature")
    return data_path, corpus_path


def audit(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_path, corpus_path = _paths(cfg)
    frame = pd.read_excel(data_path)
    report = audit_table(frame, cfg["input_columns"], cfg["output_columns"])
    report["data_file"] = str(data_path)
    report["data_sha256"] = __import__("src.research", fromlist=["sha256"]).sha256(data_path)
    report["corpus_files"] = int(len(list(corpus_path.glob("*.pdf"))))
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Audited {len(frame)} rows; wrote {out}")


def manifest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    _, corpus_path = _paths(cfg)
    table = corpus_manifest(corpus_path)
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Manifested {len(table)} PDFs; wrote {out}")


def splits(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_path, _ = _paths(cfg)
    frame = pd.read_excel(data_path)
    rows = []
    for axis in cfg["input_columns"]:
        split = single_axis_split(frame, axis, validation_fraction=args.validation_fraction, random_state=args.seed)
        rows.append({"axis": axis, **split.__dict__})
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=list), encoding="utf-8")
    print(f"Wrote {len(rows)} single-axis split manifests to {out}")


def tasks(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_path, _ = _paths(cfg)
    frame = pd.read_excel(data_path)
    table = task_manifests(frame, cfg["input_columns"], cfg["output_columns"], seed=args.seed)
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} task manifests to {out}")


def data_dictionary(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_path, _ = _paths(cfg)
    frame = pd.read_excel(data_path)
    dictionary = make_data_dictionary(frame, cfg["input_columns"], cfg["output_columns"])
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dictionary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote data dictionary to {out}")


def thresholds(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_path, _ = _paths(cfg)
    frame = pd.read_excel(data_path)
    fractions = cfg.get("evaluation", {}).get("statistical_threshold_fractions", [0.1, 0.25, 0.5, 1.0])
    table = development_threshold_grid(frame[cfg["output_column"]].dropna(), fractions)
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote development-only statistical thresholds to {out}")


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a frozen score/prediction CSV; no model selection occurs here."""
    cfg = load_config(args.config)
    frame = pd.read_csv(resolve_path(cfg, args.predictions))
    required = {args.score, args.absolute_error}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing evaluation columns: {sorted(missing)}")
    result = rank_metrics(frame[args.score], frame[args.absolute_error], args.tau)
    out = resolve_path(cfg, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote frozen-score evaluation to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("audit", audit), ("manifest", manifest), ("splits", splits), ("tasks", tasks), ("data-dictionary", data_dictionary), ("thresholds", thresholds), ("evaluate", evaluate)):
        command = sub.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--output", default=f"output/{name}.json" if name != "manifest" else "output/corpus_manifest.csv")
    sub.choices["splits"].add_argument("--seed", type=int, default=50)
    sub.choices["splits"].add_argument("--validation-fraction", type=float, default=0.25)
    sub.choices["tasks"].add_argument("--seed", type=int, default=50)
    sub.choices["data-dictionary"].set_defaults(output="output/data_dictionary.json")
    sub.choices["thresholds"].set_defaults(output="output/statistical_thresholds.csv")
    sub.choices["evaluate"].add_argument("predictions")
    sub.choices["evaluate"].add_argument("--score", default="score")
    sub.choices["evaluate"].add_argument("--absolute-error", default="absolute_error")
    sub.choices["evaluate"].add_argument("--tau", type=float, required=True)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
