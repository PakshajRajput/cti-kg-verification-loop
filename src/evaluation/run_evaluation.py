"""
Evaluation runner CLI.

Runs the CTI extraction pipeline in two conditions:
  (a) baseline — Extractor + Canonicalizer only (no verifier)
  (b) full pipeline — Extractor + Canonicalizer + Verifier

Computes P/R/F1 for both conditions on the held-out test split
and saves results to results/evaluation_results.json.

Usage:
    python -m src.evaluation.run_evaluation --subset 20 --condition both
    python -m src.evaluation.run_evaluation --subset 20 --condition baseline
    python -m src.evaluation.run_evaluation --subset 20 --condition full
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Ensure project root is on path when run as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from src.evaluation.evaluator import evaluate_pipeline_run
from src.ingestion.dataset_loader import load_ctinexus_dataset
from src.ingestion.models import CTIReport, PipelineResult
from src.pipeline.runner import PipelineConfig, PipelineRunner
from src.rag.hybrid_retriever import HybridRetriever

console = Console()
app = typer.Typer()


def _run_condition(
    runner: PipelineRunner,
    test_reports: list[CTIReport],
    condition: str,
) -> tuple[dict, dict, dict, dict]:
    """
    Run the pipeline on all test reports for one condition.

    Returns:
        predicted_by_report, ground_truth_by_report,
        rejected_by_report, extracted_counts_by_report
    """
    predicted_by_report = {}
    ground_truth_by_report = {}
    rejected_by_report = {}
    extracted_counts_by_report = {}

    for i, report in enumerate(test_reports):
        console.print(f"\n[cyan]  [{i+1}/{len(test_reports)}] Report: {report.id}[/cyan]")

        result: PipelineResult = runner.run(report)

        predicted_by_report[report.id] = result.final_triplets
        ground_truth_by_report[report.id] = report.ground_truth_triplets
        rejected_by_report[report.id] = result.rejected_triplets
        # Total extracted before verification = final + rejected
        extracted_counts_by_report[report.id] = (
            len(result.final_triplets) + len(result.rejected_triplets)
        )

        console.print(
            f"  [green]✓ {len(result.final_triplets)} final, "
            f"{len(result.rejected_triplets)} rejected, "
            f"{len(report.ground_truth_triplets)} GT[/green]"
        )

    return (
        predicted_by_report,
        ground_truth_by_report,
        rejected_by_report,
        extracted_counts_by_report,
    )


@app.command()
def main(
    subset: int = typer.Option(20, help="Number of test reports to evaluate (0 = all)"),
    condition: str = typer.Option("both", help="baseline | full | both"),
    data_dir: str = typer.Option("data", help="Root data directory"),
    results_dir: str = typer.Option("results", help="Results output directory"),
    model: str = typer.Option(None, help="LLM model override (default: from env/config)"),
):
    """Evaluate the CTI extraction pipeline and save P/R/F1 results."""
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    llm_model = model or os.environ.get("LLM_MODEL", "gpt-4o")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    console.rule("[bold cyan]CTI Pipeline Evaluation")
    console.print(f"  Model: {llm_model} | Subset: {subset if subset > 0 else 'all'} | Condition: {condition}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    console.rule("[cyan]Loading Dataset")
    train_reports, test_reports = load_ctinexus_dataset(data_path)

    if subset > 0:
        test_reports = test_reports[:subset]
    console.print(f"[green]Evaluating on {len(test_reports)} test reports.[/green]")

    # ── Build retriever (needed for full pipeline) ─────────────────────────────
    retriever = None
    if condition in ("full", "both"):
        console.rule("[cyan]Loading RAG Retriever")
        retriever = HybridRetriever.from_disk(
            data_path,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    all_results = []
    start_time = time.time()

    # ── Baseline condition ────────────────────────────────────────────────────
    if condition in ("baseline", "both"):
        console.rule("[bold yellow]Running BASELINE Condition (no verifier)")

        baseline_config = PipelineConfig(
            model=llm_model,
            embedding_model=embedding_model,
            skip_verifier=True,
            data_dir=data_path,
        )
        baseline_runner = PipelineRunner.build(baseline_config, train_reports)

        pred, gt, rej, cnt = _run_condition(baseline_runner, test_reports, "baseline")
        baseline_metrics = evaluate_pipeline_run("baseline", pred, gt)
        all_results.append(baseline_metrics.model_dump())
        console.print(f"[green]Baseline F1: {baseline_metrics.f1:.4f}[/green]")

    # ── Full pipeline condition ───────────────────────────────────────────────
    if condition in ("full", "both"):
        console.rule("[bold green]Running FULL PIPELINE Condition (with verifier)")

        full_config = PipelineConfig(
            model=llm_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            skip_verifier=False,
            data_dir=data_path,
        )
        full_runner = PipelineRunner.build(full_config, train_reports, retriever=retriever)

        pred, gt, rej, cnt = _run_condition(full_runner, test_reports, "full_pipeline")
        full_metrics = evaluate_pipeline_run(
            "full_pipeline", pred, gt,
            rejected_by_report=rej,
            extracted_counts_by_report=cnt,
        )
        all_results.append(full_metrics.model_dump())
        console.print(f"[green]Full Pipeline F1: {full_metrics.f1:.4f}[/green]")

    elapsed = time.time() - start_time

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "results": all_results,
        "metadata": {
            "model": llm_model,
            "num_test_reports": len(test_reports),
            "condition": condition,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    output_path = results_path / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    # ── Print summary table ───────────────────────────────────────────────────
    console.rule("[bold]Results Summary")
    table = Table(title="CTI Extraction Evaluation")
    table.add_column("Condition", style="cyan")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1", style="bold green")
    table.add_column("Entity F1", style="yellow")
    table.add_column("Halluc. Rate", style="red")
    table.add_column("Reports", style="dim")

    for r in all_results:
        table.add_row(
            r["condition"],
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['entity_f1']:.4f}",
            f"{r['hallucination_rate']:.4f}",
            str(r["num_reports"]),
        )

    console.print(table)
    console.print(f"\n[dim]Elapsed: {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    app()
