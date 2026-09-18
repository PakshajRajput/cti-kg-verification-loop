"""
One-time data setup script.

Run this ONCE before running any pipeline or evaluation:
  python scripts/setup_data.py

What it does:
  1. Downloads + parses CTINexus dataset
  2. Downloads MITRE ATT&CK STIX2 enterprise bundle
  3. Parses ATT&CK STIX → flat entries
  4. Builds ChromaDB vector store (sentence-transformer embeddings)
  5. Builds BM25 index over ATT&CK entries

Estimated time:
  - First run: 5-15 minutes (downloads + embedding inference)
  - Subsequent runs: < 30 seconds (loads from cache)
"""
from __future__ import annotations

import os
import sys

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from rich.console import Console

console = Console(force_terminal=True, legacy_windows=False)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.rule import Rule

from src.ingestion.dataset_loader import download_ctinexus_dataset, load_ctinexus_dataset
from src.ingestion.mitre_loader import load_mitre_attack
from src.rag.bm25_index import ATTACKBm25Index
from src.rag.vector_store import ATTACKVectorStore

console = Console()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = DATA_DIR / "chroma"
MITRE_DIR = DATA_DIR / "mitre"


def main():
    console.rule("[bold cyan]CTI KG — Data Setup")
    console.print(f"Data directory: [green]{DATA_DIR.resolve()}[/green]")

    # ── Step 1: Download CTINexus dataset ─────────────────────────────────────
    console.rule("[cyan]Step 1: CTINexus Dataset")
    raw_dir = download_ctinexus_dataset(DATA_DIR)
    train, test = load_ctinexus_dataset(DATA_DIR)
    console.print(f"[green]Dataset ready: {len(train)} train / {len(test)} test reports.[/green]")

    # ── Step 2: Download + parse MITRE ATT&CK ─────────────────────────────────
    console.rule("[cyan]Step 2: MITRE ATT&CK STIX2")
    entries = load_mitre_attack(DATA_DIR)
    console.print(f"[green]ATT&CK entries ready: {len(entries)} entries.[/green]")

    # ── Step 3: Build ChromaDB vector store ───────────────────────────────────
    console.rule("[cyan]Step 3: Building ChromaDB Vector Store")
    vector_store = ATTACKVectorStore(CHROMA_DIR, embedding_model=EMBEDDING_MODEL)
    vector_store.build(entries)
    console.print(f"[green]Vector store ready: {vector_store.collection.count()} entries.[/green]")

    # ── Step 4: Build BM25 index ──────────────────────────────────────────────
    console.rule("[cyan]Step 4: Building BM25 Index")
    bm25 = ATTACKBm25Index(MITRE_DIR)
    bm25.build(entries)
    console.print("[green]BM25 index ready.[/green]")

    # ── Done ──────────────────────────────────────────────────────────────────
    console.rule("[bold green]Setup Complete!")
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Ensure .env has HF_TOKEN set\n"
        "  2. Run the smoke test: [cyan]python scripts/test_pipeline.py[/cyan]\n"
        "  3. Run evaluation: [cyan]python -m src.evaluation.run_evaluation --subset 20 --condition both[/cyan]\n"
        "  4. Start the API: [cyan]uvicorn src.api.main:app --reload[/cyan]\n"
    )


if __name__ == "__main__":
    main()
