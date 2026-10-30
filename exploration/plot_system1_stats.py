"""Create presentation charts from the two-stage System 1 output."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output" / "system1_labels.json"
OUTPUT_DIR = ROOT / "output"


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    decisions = data["decisions"]
    retained = [decision for decision in decisions if decision["retained"]]
    total = len(decisions)

    label_names = [
        "physical_relationship",
        "mathematical_relationship",
        "physical_mechanism",
        "quantitative_evidence",
        "operating_conditions",
    ]
    label_display = [name.replace("_", " ").title() for name in label_names]
    label_rates = [
        100 * sum(bool(d["labels"].get(name)) for d in decisions) / total
        for name in label_names
    ]

    score_counts = Counter(float(d["score"]) for d in decisions)
    score_values = sorted(score_counts)
    retained_by_score = [
        sum(float(d["score"]) == score and d["retained"] for d in decisions)
        for score in score_values
    ]
    rejected_by_score = [
        sum(float(d["score"]) == score and not d["retained"] for d in decisions)
        for score in score_values
    ]

    relationship = Counter(
        str(d.get("relationship_facts", {}).get("relationship_form", "not_stated"))
        for d in retained
    )
    direction = Counter(
        str(d.get("relationship_facts", {}).get("relationship_direction", "not_stated"))
        for d in retained
    )
    range_names = [
        "range_present",
        "fixed_conditions_present",
        "applicability_limit_present",
    ]
    range_display = [
        "Explicit range",
        "Fixed conditions",
        "Applicability limit",
    ]
    range_counts = [
        sum(bool(d.get("range_facts", {}).get(name, {}).get("yes")) for d in retained)
        for name in range_names
    ]

    colors = {"retained": "#2a9d8f", "rejected": "#e76f51"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Two-Stage System 1 Filter Statistics", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.barh(label_display, label_rates, color="#457b9d")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Positive passages (%)")
    ax.set_title("Laya screening labels")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, label_rates):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")

    ax = axes[0, 1]
    width = 0.38
    x = range(len(score_values))
    ax.bar([i - width / 2 for i in x], retained_by_score, width, label="Retained", color=colors["retained"])
    ax.bar([i + width / 2 for i in x], rejected_by_score, width, label="Rejected", color=colors["rejected"])
    ax.set_xticks(list(x), [f"{score:.1f}" for score in score_values])
    ax.set_xlabel("Fraction of positive Laya labels")
    ax.set_ylabel("Passage count")
    ax.set_title("Retention by aggregate filter score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    categories = ["Relationship form", "Direction"]
    category_counts = [relationship, direction]
    category_names = sorted(set(relationship) | set(direction))
    y = range(len(category_names))
    left = [0] * len(category_names)
    palette = plt.cm.tab10.colors
    for index, category in enumerate(categories):
        values = [category_counts[index].get(name, 0) for name in category_names]
        ax.barh(
            [position + (index - 0.5) * 0.36 for position in y],
            values,
            height=0.32,
            label=category,
            color=palette[index],
        )
    ax.set_yticks(list(y), [name.replace("_", " ") for name in category_names])
    ax.set_xlabel("Retained passages")
    ax.set_title("Structured relationship findings")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    bars = ax.bar(range(len(range_counts)), range_counts, color="#f4a261")
    ax.set_xticks(range(len(range_counts)), range_display, rotation=20, ha="right")
    ax.set_ylim(0, len(retained))
    ax.set_ylabel("Retained passages")
    ax.set_title("Range and applicability evidence")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, range_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value), ha="center")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "system1_filter_stats.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(label_display, label_rates, color="#457b9d")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Positive passages (%)")
    ax.set_title(f"Laya Filter Positive Rates (n={total})", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, label_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    label_path = OUTPUT_DIR / "system1_label_positive_rates.png"
    fig.savefig(label_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    confidence_values = [
        [
            float(
                decision["raw_answers"][name].get(
                    "answer_confidence",
                    decision["raw_answers"][name].get("confidence", 0.0),
                )
            )
            for decision in decisions
        ]
        for name in label_names
    ]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    box = ax.boxplot(
        confidence_values,
        tick_labels=label_display,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#8ecae6")
        patch.set_alpha(0.85)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Laya answer confidence")
    ax.set_title(
        f"Confidence Distribution by Noul Test (n={total})",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.01,
        "Note: confidence is uncalibrated model confidence; labels use noul > 0.5.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    confidence_path = OUTPUT_DIR / "system1_noul_confidence_boxplot.png"
    fig.savefig(confidence_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(summary_path)
    print(label_path)
    print(confidence_path)


if __name__ == "__main__":
    main()
