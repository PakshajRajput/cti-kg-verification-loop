"""
ChromaDB vector store for MITRE ATT&CK dense retrieval.

Builds and persists a ChromaDB collection of ATT&CK entry embeddings
using sentence-transformers (local, CPU-friendly).
"""
from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rich.console import Console
from sentence_transformers import SentenceTransformer

from src.ingestion.models import ATT_CK_Entry

console = Console()


class ATTACKVectorStore:
    """
    ChromaDB-backed dense vector store over MITRE ATT&CK entries.

    Collection name: "attack_entries"
    Embedding model: configurable via EMBEDDING_MODEL env var (default: all-MiniLM-L6-v2)
    """

    COLLECTION_NAME = "attack_entries"

    def __init__(self, persist_dir: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model

        console.print(f"[cyan]Loading embedding model: {embedding_model}[/cyan]")
        self.embedder = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection (using cosine similarity)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def build(self, entries: list[ATT_CK_Entry], batch_size: int = 128) -> None:
        """
        Embed all ATT&CK entries and insert into ChromaDB.
        Skips if collection is already populated.
        """
        if self.is_populated():
            console.print(
                f"[green]Vector store already populated ({self.collection.count()} entries). Skipping.[/green]"
            )
            return

        console.print(f"[cyan]Building vector store for {len(entries)} ATT&CK entries...[/cyan]")

        # Prepare texts, IDs, and metadata in batches
        texts = [e.to_text() for e in entries]
        ids = [f"{e.id}_{i}" for i, e in enumerate(entries)]  # ensure uniqueness
        metadatas = [
            {
                "attack_id": e.id,
                "name": e.name,
                "type": e.type,
                "tactics": ",".join(e.tactics),
            }
            for e in entries
        ]

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]

            embeddings = self.embedder.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )
            console.print(
                f"[cyan]  Indexed {min(start + batch_size, len(texts))}/{len(texts)} entries...[/cyan]"
            )

        console.print(f"[green]Vector store built: {self.collection.count()} entries.[/green]")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Dense similarity search.

        Returns list of dicts with keys: id, document, metadata, distance.
        """
        if not self.is_populated():
            console.print("[yellow]Vector store is empty. Run build() first.[/yellow]")
            return []

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids_ = results.get("ids", [[]])[0]

        for doc, meta, dist, id_ in zip(docs, metas, dists, ids_):
            hits.append({
                "id": id_,
                "attack_id": meta.get("attack_id", ""),
                "name": meta.get("name", ""),
                "type": meta.get("type", ""),
                "document": doc,
                "score": 1.0 - dist,  # cosine similarity (1 = identical)
            })

        return hits

    def get_embedder(self) -> SentenceTransformer:
        return self.embedder
