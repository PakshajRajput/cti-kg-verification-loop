"""
Results visualization script.

Reads results/evaluation_results.json and generates:
  1. Grouped bar chart: Precision/Recall/F1 for baseline vs. full pipeline
  2. Hallucination rate comparison bar chart
  3. Entity-level F1 comparison

Saves all plots to results/figures/ as high-res PNG.

Usage:
    python src/evaluation/plot_results.py
    python src/evaluation/plot_results.py --results-file results/evaluation_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import typer

app = typer.Typer()

# ─── Style Configuration ──────────────────────────────────────────────────────
COLORS = {
    "baseline": "#E07B54",       # warm orange
    "full_pipeline": "#4C9BE8",  # blue
}
CONDITION_LABELS = {
    "baseline": "Baseline\n(No Verification)",
    "full_pipeline": "Full Pipeline\n(+ Verifier)",
}
FONT_FAMILY = "DejaVu Sans"


def _setup_style():
    """Configure matplotlib for paper-quality plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": FONT_FAMILY,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


@app.command()
def main(
    results_file: str = typer.Option("results/evaluation_results.json", help="Path to evaluation results JSON"),
    figures_dir: str = typer.Option("results/figures", help="Output directory for figures"),
):
    """Generate evaluation comparison plots."""
    results_path = Path(results_file)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run: python -m src.evaluation.run_evaluation --condition both")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    metadata = data.get("metadata", {})

    if not results:
        print("No results found in file.")
        sys.exit(1)

    print(f"Loaded {len(results)} condition(s) from {results_path}")
    _setup_style()

    # ── Plot 1: Triplet P/R/F1 Comparison ────────────────────────────────────
    _plot_prf1(results, figures_path, metadata)

    # ── Plot 2: Hallucination Rate ────────────────────────────────────────────
    _plot_hallucination_rate(results, figures_path, metadata)

    # ── Plot 3: Entity F1 ────────────────────────────────────────────────────
    _plot_entity_f1(results, figures_path, metadata)

    print(f"\nAll plots saved to {figures_path}/")


def _plot_prf1(results: list[dict], figures_path: Path, metadata: dict):
    """Grouped bar chart: Precision, Recall, F1 per condition."""
    conditions = [r["condition"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    x = np.arange(len(conditions))
    width = 0.22
    offsets = [-width, 0, width]
    metric_data = [precision, recall, f1]
    metric_labels = ["Precision", "Recall", "F1"]
    metric_colors = ["#5B9BD5", "#70AD47", "#FF7C43"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric, label, color) in enumerate(zip(metric_data, metric_labels, metric_colors)):
        bars = ax.bar(x + offsets[i], metric, width, label=label, color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, metric):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Triplet-Level Precision / Recall / F1\n"
        f"Model: {metadata.get('model', 'GPT-4o')} | "
        f"Test reports: {metadata.get('num_test_reports', '?')}",
        pad=10,
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = figures_path / "prf1_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_hallucination_rate(results: list[dict], figures_path: Path, metadata: dict):
    """Bar chart: hallucination rate per condition."""
    conditions = [r["condition"] for r in results]
    hall_rates = [r["hallucination_rate"] * 100 for r in results]  # Convert to %

    colors = [COLORS.get(c, "#888") for c in conditions]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, hall_rates, color=colors, alpha=0.88, width=0.4, zorder=3)

    for bar, val in zip(bars, hall_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title(
        "Hallucination Rate: Baseline vs. Full Pipeline\n"
        "(Lower is better — fraction of triplets rejected by verifier)",
        pad=10,
    )
    ax.set_ylim(0, max(hall_rates + [10]) * 1.3)

    # Annotation for full pipeline bar
    if len(conditions) == 2 and conditions[0] == "baseline" and conditions[1] == "full_pipeline":
        reduction = hall_rates[0] - hall_rates[1]
        if reduction > 0:
            ax.annotate(
                f"−{reduction:.1f}pp reduction",
                xy=(1, hall_rates[1]),
                xytext=(0.5, max(hall_rates) * 0.7),
                fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green"),
            )

    plt.tight_layout()
    out = figures_path / "hallucination_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_entity_f1(results: list[dict], figures_path: Path, metadata: dict):
    """Bar chart: entity-level F1 comparison."""
    conditions = [r["condition"] for r in results]
    entity_f1 = [r.get("entity_f1", 0.0) for r in results]
    triplet_f1 = [r["f1"] for r in results]

    x = np.arange(len(conditions))
    width = 0.3
    colors_e = "#9B59B6"
    colors_t = "#2ECC71"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, entity_f1, width, label="Entity F1", color=colors_e, alpha=0.88, zorder=3)
    bars2 = ax.bar(x + width / 2, triplet_f1, width, label="Triplet F1", color=colors_t, alpha=0.88, zorder=3)

    for bars, vals in [(bars1, entity_f1), (bars2, triplet_f1)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Entity F1 vs. Triplet F1 by Condition", pad=10)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = figures_path / "entity_vs_triplet_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    app()
