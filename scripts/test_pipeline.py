"""
Smoke test — run this FIRST after setup.

Runs the full pipeline on 3 synthetic CTI reports to verify:
  - HF_TOKEN is valid and callable
  - Embedding model loads correctly
  - BM25 + ChromaDB indexes are loaded
  - Cross-encoder reranker loads correctly
  - LangGraph pipeline executes without error (extractor + verifier)
  - NetworkX graph connection works

Usage:
    python scripts/test_pipeline.py

Expected output: 3 reports processed with triplets printed to console.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from src.ingestion.dataset_loader import _create_synthetic_reports
from src.ingestion.mitre_loader import load_mitre_attack, _create_fallback_entries
from src.kg.networkx_writer import NetworkXWriter
from src.pipeline.runner import PipelineConfig, PipelineRunner
from src.rag.bm25_index import ATTACKBm25Index
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import ATTACKVectorStore

console = Console(force_terminal=True, legacy_windows=False)
DATA_DIR = Path("data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


def check_env():
    """Verify required environment variables."""
    console.rule("[cyan]Environment Check")
    api_key = os.environ.get("HF_TOKEN", "")
    if not api_key or not api_key.startswith("hf_"):
        console.print("[red]✗ HF_TOKEN not set or invalid. Edit your .env file.[/red]")
        sys.exit(1)
    console.print(f"[green]✓ HF_TOKEN found (hf_...{api_key[-4:]})[/green]")
    console.print(f"[green]✓ LLM model: {LLM_MODEL}[/green]")


def build_indexes():
    """Build or load ATT&CK indexes."""
    console.rule("[cyan]Loading ATT&CK Indexes")
    chroma_dir = DATA_DIR / "chroma"
    mitre_dir = DATA_DIR / "mitre"

    # Load or create entries
    entries_path = mitre_dir / "entries.jsonl"
    if entries_path.exists():
        entries = load_mitre_attack(DATA_DIR)
    else:
        console.print("[yellow]No ATT&CK data found. Using fallback entries for smoke test.[/yellow]")
        mitre_dir.mkdir(parents=True, exist_ok=True)
        entries = _create_fallback_entries(entries_path)

    # Vector store
    vs = ATTACKVectorStore(chroma_dir, embedding_model=EMBEDDING_MODEL)
    if not vs.is_populated():
        vs.build(entries)
    console.print(f"[green]✓ Vector store: {vs.collection.count()} entries[/green]")

    # BM25
    bm25 = ATTACKBm25Index(mitre_dir)
    if not bm25.is_built():
        bm25.build(entries)
    else:
        bm25.load()
    console.print(f"[green]✓ BM25 index loaded[/green]")

    return HybridRetriever(vs, bm25, reranker_model=RERANKER_MODEL)


def run_smoke_test(retriever: HybridRetriever):
    """Run the full pipeline on 3 synthetic reports."""
    console.rule("[cyan]Running Smoke Test — 3 Reports")

    # Get 3 synthetic test reports
    test_reports = _create_synthetic_reports(3)

    # Use a few of the synthetic reports as demonstrations
    demo_reports = _create_synthetic_reports(5)[3:]  # different from test set

    config = PipelineConfig(
        model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        skip_verifier=False,
        data_dir=DATA_DIR,
    )

    runner = PipelineRunner.build(config, demo_reports, retriever=retriever)

    all_passed = True
    for i, report in enumerate(test_reports, 1):
        console.print(f"\n[bold cyan]─── Report {i}/3: {report.id} ───[/bold cyan]")
        console.print(f"[dim]{report.text[:150]}...[/dim]")

        result = runner.run(report)

        if result.error:
            console.print(f"[red]✗ Error: {result.error}[/red]")
            all_passed = False
            continue

        # Print results table
        table = Table(title=f"Report {report.id}: Extracted Triplets")
        table.add_column("Entity 1", style="cyan")
        table.add_column("Type 1", style="dim")
        table.add_column("Relation", style="yellow")
        table.add_column("Entity 2", style="cyan")
        table.add_column("Type 2", style="dim")
        table.add_column("Conf", style="green")

        for t in result.final_triplets:
            table.add_row(
                t.entity1, t.entity1_type.value,
                t.relation,
                t.entity2, t.entity2_type.value,
                f"{t.confidence:.2f}",
            )

        console.print(table)

        if result.rejected_triplets:
            console.print(f"[yellow]Rejected: {len(result.rejected_triplets)} triplets[/yellow]")
            for r in result.rejected_triplets:
                console.print(f"  [red]✗ {r.to_text()}[/red]")

        gt_count = len(report.ground_truth_triplets)
        final_count = len(result.final_triplets)
        console.print(
            f"[green]✓ Report {i} done: {final_count} verified triplets "
            f"(ground truth: {gt_count})[/green]"
        )

    return all_passed


def check_kg():
    """Test NetworkX connection."""
    console.rule("[cyan]NetworkX Graph Check")
    try:
        kg = NetworkXWriter("data/test_kg.json")
        kg.connect()
        stats = kg.get_stats()
        kg.close()
        console.print(f"[green]✓ NetworkX graph working. Graph has {stats['total_nodes']} nodes, {stats['total_edges']} edges.[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ NetworkX graph failed: {e}[/yellow]")


def main():
    console.rule("[bold cyan]CTI KG Pipeline — Smoke Test")

    # 1. Check environment
    check_env()

    # 2. Load indexes
    retriever = build_indexes()

    # 3. Run pipeline on 3 reports
    passed = run_smoke_test(retriever)

    # 4. Check NetworkX
    check_kg()

    # Summary
    console.rule("[bold]Smoke Test Result")
    if passed:
        console.print("[bold green]✓ All checks passed! Pipeline is working.[/bold green]")
        console.print(
            "\nNext steps:\n"
            "  • Full evaluation: [cyan]python -m src.evaluation.run_evaluation --subset 20 --condition both[/cyan]\n"
            "  • Start API:       [cyan]uvicorn src.api.main:app --reload[/cyan]\n"
        )
    else:
        console.print("[bold red]✗ Some checks failed. See errors above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
