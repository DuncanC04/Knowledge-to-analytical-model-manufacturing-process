"""Create a presentation visual for the second-stage LLM descriptions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output" / "system1_labels.json"
OUTPUT = ROOT / "output" / "system1_stage2_results.png"


def present(value: object) -> bool:
    return value is not None and str(value) not in {"", "[]", "{}", "None"}


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    decisions = [d for d in data["decisions"] if d.get("retained")]
    total = len(decisions)
    fields = [
        ("summary", "Summary"),
        ("variables", "Variables"),
        ("relationship", "Relationship"),
        ("mechanism", "Mechanism"),
        ("ranges", "Ranges"),
        ("conditions", "Conditions"),
        ("evidence_quote", "Evidence quote"),
    ]
    counts = [
        sum(present(d.get("llm_description", {}).get(key)) for d in decisions)
        for key, _ in fields
    ]
    rates = [100 * count / total for count in counts]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])
    fig.suptitle("Second-Stage LLM Evidence Descriptions", fontsize=18, fontweight="bold")

    flow = fig.add_subplot(grid[0, 0])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    boxes = [
        (0.08, 0.72, 0.84, 0.16, "Laya-retained chunks", f"{total} chunks\nstructured signals"),
        (0.08, 0.42, 0.84, 0.16, "LLM description stage", "Evidence-grounded\nJSON extraction"),
        (0.08, 0.12, 0.84, 0.16, "Reusable evidence records", "Summary · variables · relationship\nmechanism · ranges · conditions · quote"),
    ]
    colors = ["#2a9d8f", "#457b9d", "#f4a261"]
    for (x, y, width, height, title, detail), color in zip(boxes, colors):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.95,
        )
        flow.add_patch(patch)
        flow.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        flow.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", color="white", fontsize=10)
    for y in (0.68, 0.38):
        flow.add_patch(FancyArrowPatch((0.5, y), (0.5, y - 0.08), arrowstyle="-|>", mutation_scale=18, linewidth=2, color="#555555"))
    flow.text(0.5, 0.03, "Only retained chunks consume second-stage LLM calls", ha="center", fontsize=10, color="#555555")

    chart = fig.add_subplot(grid[0, 1])
    labels = [label for _, label in fields]
    bars = chart.barh(labels, rates, color="#5dade2")
    chart.set_xlim(0, 100)
    chart.set_xlabel("Retained chunks with populated field (%)")
    chart.set_title("Description field completeness", fontsize=13)
    chart.invert_yaxis()
    chart.grid(axis="x", alpha=0.25)
    for bar, rate in zip(bars, rates):
        chart.text(rate + 1, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center", fontsize=10)
    chart.text(0.0, -0.16, "n = " + str(total) + " retained chunks; model: " + str(data.get("description_model")), transform=chart.transAxes, fontsize=9, color="#555555")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
