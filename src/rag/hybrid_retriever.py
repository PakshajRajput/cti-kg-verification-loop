"""
Hybrid retriever: BM25 + Dense (ChromaDB) + Cross-Encoder Reranker.

Pipeline:
  1. BM25 sparse retrieval (top-k candidates)
  2. Dense embedding retrieval via ChromaDB (top-k candidates)
  3. Reciprocal Rank Fusion (RRF) to merge ranked lists
  4. Cross-encoder reranking of top-N fused candidates
  5. Return final top-k results

References:
  - RRF: Cormack et al. (2009) "Reciprocal rank fusion outperforms condorcet"
  - Cross-encoder: ms-marco-MiniLM-L-6-v2 (fast, accurate passage reranker)
"""
from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from sentence_transformers import CrossEncoder

from src.ingestion.models import ATT_CK_Entry
from src.rag.bm25_index import ATTACKBm25Index
from src.rag.vector_store import ATTACKVectorStore

console = Console()


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    score(d) = sum over lists: 1 / (k + rank(d))

    Args:
        ranked_lists: List of ranked result lists. Each result dict must have "attack_id".
        k: Constant (60 recommended in original paper).

    Returns:
        Single merged and re-ranked list of result dicts.
    """
    scores: dict[str, float] = {}
    # Store the best doc dict for each attack_id
    best_doc: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            aid = doc.get("attack_id", doc.get("id", ""))
            if not aid:
                continue
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
            # Keep the first (dense) doc dict as canonical
            if aid not in best_doc:
                best_doc[aid] = doc

    # Sort by RRF score descending
    sorted_aids = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
    merged = []
    for aid in sorted_aids:
        doc = best_doc[aid].copy()
        doc["rrf_score"] = scores[aid]
        merged.append(doc)

    return merged


class HybridRetriever:
    """
    Hybrid ATT&CK retriever combining sparse (BM25) + dense (ChromaDB) + reranking.

    Usage:
        retriever = HybridRetriever.from_disk(data_dir, embedding_model, reranker_model)
        results = retriever.retrieve("APT29 uses Cobalt Strike for C2", top_k=5)
    """

    def __init__(
        self,
        vector_store: ATTACKVectorStore,
        bm25_index: ATTACKBm25Index,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 20,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.fetch_k = fetch_k

        console.print(f"[cyan]Loading cross-encoder reranker: {reranker_model}[/cyan]")
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        console.print("[green]Hybrid retriever ready.[/green]")

    @classmethod
    def from_disk(
        cls,
        data_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 20,
    ) -> "HybridRetriever":
        """Load a pre-built retriever from disk (both indexes must already be built)."""
        chroma_dir = data_dir / "chroma"
        bm25_dir = data_dir / "mitre"

        vector_store = ATTACKVectorStore(chroma_dir, embedding_model)
        bm25_index = ATTACKBm25Index(bm25_dir)
        bm25_index.load()

        return cls(vector_store, bm25_index, reranker_model, fetch_k)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the most relevant ATT&CK entries for a query.

        Args:
            query: The search query (typically a triplet description).
            top_k: Number of results to return after reranking.

        Returns:
            List of result dicts with keys:
                attack_id, name, type, document, rrf_score, rerank_score
        """
        # Step 1: Dense retrieval
        dense_results = self.vector_store.search(query, k=self.fetch_k)

        # Step 2: Sparse BM25 retrieval
        sparse_results = self.bm25_index.search(query, k=self.fetch_k)

        # Step 3: RRF fusion
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])

        # Take top-N for reranking (avoid reranking hundreds of docs)
        candidates = fused[:min(self.fetch_k, len(fused))]

        if not candidates:
            return []

        # Step 4: Cross-encoder reranking
        pairs = [(query, c["document"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(rerank_scores[i])

        # Sort by reranker score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]
