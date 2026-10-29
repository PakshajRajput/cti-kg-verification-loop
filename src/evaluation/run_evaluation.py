"""
Evaluation runner CLI.

Runs the CTI extraction pipeline in two conditions:
  (a) baseline — Extractor + Canonicalizer only (no verifier)
  (b) full pipeline — Extractor + Canonicalizer + Verifier

Writes aggregate metrics to evaluation_results.json and a per-report JSONL
audit record to per_report_audit.jsonl for every report/condition run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from src.evaluation.audit_log import (
    build_audit_record,
    build_experiment_manifest,
    write_json,
    write_jsonl_record,
)
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
    audit_path: Path | None = None,
    model: str | None = None,
    dataset_mode: str = "unknown",
) -> tuple[dict, dict, dict, dict]:
    """Run one condition, emitting a JSONL audit record for each completed report."""
    predicted_by_report = {}
    ground_truth_by_report = {}
    rejected_by_report = {}
    extracted_counts_by_report = {}

    for i, report in enumerate(test_reports):
        console.print(f"\n[cyan]  [{i+1}/{len(test_reports)}] Report: {report.id}[/cyan]")
        result: PipelineResult = runner.run(report)

        # Persist an error record too so a failed experiment is not silently lost.
        if audit_path is not None:
            record = build_audit_record(
                report=report,
                result=result,
                condition=condition,
                model=model or getattr(getattr(runner, "config", None), "model", "unknown"),
                dataset_mode=dataset_mode,
            )
            write_jsonl_record(audit_path, record)

        if result.error:
            raise RuntimeError(
                f"Pipeline failed for report {report.id!r} during {condition}: {result.error}. "
                f"See audit record at {audit_path if audit_path else 'not configured'}."
            )

        predicted_by_report[report.id] = result.final_triplets
        ground_truth_by_report[report.id] = report.ground_truth_triplets
        rejected_by_report[report.id] = result.rejected_triplets
        # Rejection rate describes the final verifier pass, not all raw outputs
        # accumulated over re-extraction attempts. Raw totals remain in the audit.
        extracted_counts_by_report[report.id] = (
            len(result.verified_triplets) + len(result.rejected_triplets)
            if condition == "full_pipeline"
            else len(result.final_triplets)
        )

        console.print(
            f"  [green]✓ {len(result.final_triplets)} final, "
            f"{len(result.rejected_triplets)} rejected, "
            f"{len(report.ground_truth_triplets)} GT[/green]"
        )

    return predicted_by_report, ground_truth_by_report, rejected_by_report, extracted_counts_by_report


@app.command()
def main(
    subset: int = typer.Option(20, help="Number of test reports to evaluate (0 = all)"),
    condition: str = typer.Option("both", help="baseline | full | both"),
    data_dir: str = typer.Option("data", help="Root data directory"),
    results_dir: str = typer.Option("results", help="Results output directory"),
    model: str = typer.Option(None, help="LLM model override (default: from env/config)"),
    allow_synthetic_data: bool = typer.Option(
        False, help="Allow synthetic supplementation for demo only; never use for reported research results"
    ),
    seed: int = typer.Option(42, min=0, help="Seed for the reproducible train/test split"),
    test_split: float = typer.Option(0.2, min=0.01, max=0.99, help="Fraction held out for testing (0.01-0.99)"),
):
    """Evaluate the CTI extraction pipeline and save metrics plus per-report audit records."""
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    llm_model = model or os.environ.get("LLM_MODEL", "gpt-4o")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    console.rule("[bold cyan]CTI Pipeline Evaluation")
    console.print(f"  Model: {llm_model} | Subset: {subset if subset > 0 else 'all'} | Condition: {condition}")

    if condition not in {"baseline", "full", "both"}:
        raise typer.BadParameter("condition must be one of: baseline, full, both")

    train_reports, all_test_reports = load_ctinexus_dataset(
        data_path,
        allow_synthetic=allow_synthetic_data,
        seed=seed,
        test_split=test_split,
    )
    # Use the same frozen report IDs for every requested condition; subset only
    # affects evaluation, not the dataset provenance accounting.
    test_reports = all_test_reports[:subset] if subset > 0 else all_test_reports
    all_dataset_reports = train_reports + all_test_reports
    synthetic_report_ids = sorted(
        report.id for report in all_dataset_reports if report.id.startswith("synthetic_")
    )
    dataset_mode = "synthetic_demo" if synthetic_report_ids else "real_data"
    console.print(
        f"[green]Evaluating on {len(test_reports)} test reports "
        f"(split seed={seed}, test fraction={test_split}).[/green]"
    )

    # Start clean per-run artifacts; stale lines from previous runs must not mix
    # with this experiment's per-report audit records.
    audit_path = results_path / "per_report_audit.jsonl"
    manifest_path = results_path / "experiment_manifest.json"
    if audit_path.exists():
        audit_path.unlink()

    started_at_utc = datetime.now(timezone.utc).isoformat()
    manifest = build_experiment_manifest(
        started_at_utc=started_at_utc,
        condition=condition,
        model=llm_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        seed=seed,
        test_split=test_split,
        subset=subset,
        train_reports=train_reports,
        test_reports=test_reports,
        dataset_mode=dataset_mode,
        allow_synthetic_data=allow_synthetic_data,
        project_root=Path(__file__).resolve().parents[2],
    )
    write_json(manifest_path, manifest)
    console.print(f"[green]Experiment manifest saved to {manifest_path}[/green]")

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

    if condition in ("baseline", "both"):
        console.rule("[bold yellow]Running BASELINE Condition (no verifier)")
        baseline_runner = PipelineRunner.build(
            PipelineConfig(
                model=llm_model, embedding_model=embedding_model,
                skip_verifier=True, data_dir=data_path,
            ),
            train_reports,
        )
        pred, gt, rej, cnt = _run_condition(
            baseline_runner, test_reports, "baseline", audit_path, llm_model, dataset_mode
        )
        baseline_metrics = evaluate_pipeline_run("baseline", pred, gt, extracted_counts_by_report=cnt)
        all_results.append(baseline_metrics.model_dump())
        console.print(f"[green]Baseline F1: {baseline_metrics.f1:.4f}[/green]")

    if condition in ("full", "both"):
        console.rule("[bold green]Running FULL PIPELINE Condition (with verifier)")
        full_runner = PipelineRunner.build(
            PipelineConfig(
                model=llm_model, embedding_model=embedding_model,
                reranker_model=reranker_model, skip_verifier=False, data_dir=data_path,
            ),
            train_reports, retriever=retriever,
        )
        pred, gt, rej, cnt = _run_condition(
            full_runner, test_reports, "full_pipeline", audit_path, llm_model, dataset_mode
        )
        full_metrics = evaluate_pipeline_run(
            "full_pipeline", pred, gt, rejected_by_report=rej, extracted_counts_by_report=cnt
        )
        all_results.append(full_metrics.model_dump())
        console.print(f"[green]Full Pipeline F1: {full_metrics.f1:.4f}[/green]")

    elapsed = time.time() - start_time
    output = {
        "results": all_results,
        "metadata": {
            "model": llm_model,
            "embedding_model": embedding_model,
            "reranker_model": reranker_model if condition in {"full", "both"} else None,
            "num_test_reports": len(test_reports),
            "condition": condition,
            "seed": seed,
            "test_split_fraction": test_split,
            "started_at_utc": started_at_utc,
            "dataset_mode": dataset_mode,
            "allow_synthetic_data": allow_synthetic_data,
            "real_report_count": len(all_dataset_reports) - len(synthetic_report_ids),
            "synthetic_report_count": len(synthetic_report_ids),
            "synthetic_report_ids": synthetic_report_ids,
            "test_report_ids": [report.id for report in test_reports],
            "elapsed_seconds": round(elapsed, 1),
            "per_report_audit_file": str(audit_path),
            "experiment_manifest_file": str(manifest_path),
            "dataset_fingerprint_sha256": manifest["dataset"]["fingerprint_sha256"],
            "prompt_sha256": manifest["prompt_sha256"],
            "implementation_sha256": manifest["implementation_sha256"],
        },
    }
    output_path = results_path / "evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")
    console.print(f"[green]Per-report audit saved to {audit_path}[/green]")

    table = Table(title="CTI Extraction Evaluation")
    table.add_column("Condition", style="cyan")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1", style="bold green")
    table.add_column("Entity F1", style="yellow")
    table.add_column("Reject Rate", style="red")
    table.add_column("Reports", style="dim")
    for r in all_results:
        table.add_row(
            r["condition"], f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['f1']:.4f}", f"{r['entity_f1']:.4f}",
            f"{r.get('rejection_rate', r.get('hallucination_rate', 0.0)):.4f}",
            str(r["num_reports"]),
        )
    console.print(table)
    console.print(f"\n[dim]Elapsed: {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    app()
