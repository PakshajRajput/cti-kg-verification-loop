"""
BM25 sparse index over MITRE ATT&CK entries.

Uses rank-bm25 (Okapi BM25) for keyword-based lexical retrieval.
Complements dense retrieval in hybrid search.
"""
from __future__ import annotations

import json
import pickle
import re
import string
from pathlib import Path

from rank_bm25 import BM25Okapi
from rich.console import Console

from src.ingestion.models import ATT_CK_Entry

console = Console()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    # Replace hyphens with spaces (important for CVE-XXXX-XXXX style tokens)
    text = text.replace("-", " ")
    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


class ATTACKBm25Index:
    """
    BM25 sparse index over MITRE ATT&CK entry texts.

    Persists the index + corpus to disk so it only needs to be built once.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[ATT_CK_Entry] = []
        self._index_path = index_dir / "bm25_index.pkl"
        self._corpus_path = index_dir / "bm25_corpus.jsonl"

    def is_built(self) -> bool:
        return self._index_path.exists() and self._corpus_path.exists()

    def build(self, entries: list[ATT_CK_Entry]) -> None:
        """Build and persist BM25 index from ATT&CK entries."""
        if self.is_built():
            console.print("[green]BM25 index already built. Skipping.[/green]")
            self._load()
            return

        console.print(f"[cyan]Building BM25 index over {len(entries)} ATT&CK entries...[/cyan]")
        self._corpus = entries
        tokenized = [_tokenize(e.to_text()) for e in entries]
        self._bm25 = BM25Okapi(tokenized)

        # Persist
        with open(self._index_path, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(self._corpus_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(e.model_dump_json() + "\n")

        console.print(f"[green]BM25 index built and saved.[/green]")

    def _load(self) -> None:
        """Load persisted BM25 index from disk."""
        with open(self._index_path, "rb") as f:
            self._bm25 = pickle.load(f)
        self._corpus = []
        with open(self._corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._corpus.append(ATT_CK_Entry.model_validate_json(line))

    def load(self) -> None:
        """Public load method — call before search if not built in this session."""
        if self._bm25 is None:
            if self.is_built():
                self._load()
                console.print(f"[green]BM25 index loaded ({len(self._corpus)} entries).[/green]")
            else:
                console.print("[yellow]BM25 index not built yet. Call build() first.[/yellow]")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        BM25 keyword search.

        Returns list of dicts with keys: attack_id, name, type, document, score.
        """
        if self._bm25 is None:
            self.load()
        if self._bm25 is None or not self._corpus:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        hits = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            entry = self._corpus[idx]
            hits.append({
                "attack_id": entry.id,
                "name": entry.name,
                "type": entry.type,
                "document": entry.to_text(),
                "score": float(scores[idx]),
            })

        return hits
