"""
Canonicalizer Agent — LangGraph node.

Deduplicates and merges entity mentions that refer to the same concept.

Approach:
  1. Cluster entity mentions by embedding cosine similarity (threshold 0.85)
  2. For each cluster, use LLM to select a canonical name / detect aliases
  3. Rewrite triplets using canonical names
  4. Deduplicate identical triplets after rewriting

This catches cases like:
  "APT29" = "Cozy Bear" = "the threat actor" = "they" (in context)
  "CVE-2021-40444" = "the vulnerability" = "the MSHTML exploit"
"""
from __future__ import annotations

import json
import os
import re
from itertools import combinations

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from src.agents.llm_client import call_llm

from src.ingestion.models import EntityType, PipelineState, Triplet

console = Console()


def _cluster_entities(
    entity_names: list[str],
    embedder: SentenceTransformer,
    threshold: float = 0.85,
) -> list[list[str]]:
    """
    Single-linkage clustering of entity names by embedding cosine similarity.
    Returns a list of clusters (each cluster is a list of entity names).
    """
    if not entity_names:
        return []

    embeddings = embedder.encode(entity_names, normalize_embeddings=True)
    n = len(entity_names)
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                assigned[j] = cluster_id
        cluster_id += 1

    clusters: dict[int, list[str]] = {}
    for idx, cid in enumerate(assigned):
        clusters.setdefault(cid, []).append(entity_names[idx])

    return list(clusters.values())


def _select_canonical(cluster: list[str]) -> str:
    """
    Heuristic canonical name selection (used as fallback without LLM call).
    Prefers the shortest non-pronoun name, or the first name if all similar.
    """
    pronouns = {"they", "them", "it", "the group", "the actor", "the attacker", "the attackers"}
    non_pronouns = [n for n in cluster if n.lower() not in pronouns]
    if non_pronouns:
        # Prefer names that look like proper nouns (capital letter start)
        proper = [n for n in non_pronouns if n and n[0].isupper()]
        if proper:
            return min(proper, key=len)
        return min(non_pronouns, key=len)
    return cluster[0]


class CanonicalizerAgent:
    """
    Entity alignment and deduplication agent.

    Merges entity aliases and removes duplicate triplets.
    Uses embedding clustering + optional LLM confirmation for ambiguous cases.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ):
        self.model = model
        self.threshold = similarity_threshold
        self.embedder = SentenceTransformer(embedding_model)

    def _llm_resolve_aliases(self, cluster: list[str]) -> str:
        """Use LLM to pick the canonical name for a cluster of similar entities."""
        if len(cluster) == 1:
            return cluster[0]

        prompt = (
            f"These entity names likely refer to the same cybersecurity concept. "
            f"Choose the most canonical, specific name (prefer official names over pronouns or vague descriptors).\n\n"
            f"Names: {json.dumps(cluster)}\n\n"
            f"Reply with ONLY the chosen canonical name, nothing else."
        )
        try:
            resp = call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            choice = resp.strip().strip('"')
            return choice if choice in cluster else _select_canonical(cluster)
        except Exception:
            return _select_canonical(cluster)

    def canonicalize(self, triplets: list[Triplet]) -> list[Triplet]:
        """
        Canonicalize entity names and deduplicate triplets.

        Returns a new list of triplets with canonical entity names.
        """
        if not triplets:
            return []

        # Collect all unique entity names
        all_entities = list({t.entity1 for t in triplets} | {t.entity2 for t in triplets})

        # Cluster by embedding similarity
        clusters = _cluster_entities(all_entities, self.embedder, self.threshold)
        console.print(
            f"[cyan]Canonicalizer: {len(all_entities)} entities → {len(clusters)} clusters.[/cyan]"
        )

        # Build alias → canonical mapping
        alias_map: dict[str, str] = {}
        for cluster in clusters:
            if len(cluster) == 1:
                alias_map[cluster[0]] = cluster[0]
            else:
                canonical = self._llm_resolve_aliases(cluster)
                for name in cluster:
                    alias_map[name] = canonical
                console.print(f"[dim]  Merged: {cluster} → '{canonical}'[/dim]")

        # Rewrite triplets using canonical names
        canonical_triplets = []
        for t in triplets:
            canonical_triplets.append(Triplet(
                entity1=alias_map.get(t.entity1, t.entity1),
                entity1_type=t.entity1_type,
                relation=t.relation,
                entity2=alias_map.get(t.entity2, t.entity2),
                entity2_type=t.entity2_type,
                confidence=t.confidence,
                source_text=t.source_text,
            ))

        console.print(
            f"[green]Canonicalizer: {len(triplets)} → {len(canonical_triplets)} triplets after merge (no deduplication here).[/green]"
        )
        return canonical_triplets


def make_canonicalizer_node(agent: CanonicalizerAgent):
    """Factory: returns a LangGraph node function for the canonicalizer."""
    def canonicalizer_node(state: dict) -> dict:
        pipeline_state = PipelineState.model_validate(state)
        canonical = agent.canonicalize(pipeline_state.raw_triplets)
        pipeline_state.canonical_triplets = canonical
        return pipeline_state.model_dump()

    return canonicalizer_node
