"""
LangGraph Pipeline Graph.

Wires the Extractor → Canonicalizer → Verifier nodes into a StateGraph
with a conditional feedback edge: if verification rejects too many triplets
and we haven't exceeded max iterations, route back to the Extractor.

Graph topology:
    START → extractor → canonicalizer → verifier
                ▲                           │
                │    [needs_reextraction]    │
                └───────────────────────────┘
                         [otherwise]
                              ↓
                         kg_builder → END
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from rich.console import Console

from src.agents.canonicalizer import CanonicalizerAgent, make_canonicalizer_node
from src.agents.extractor import ExtractorAgent, make_extractor_node
from src.agents.verifier import VerifierAgent, make_verifier_node
from src.ingestion.models import CTIReport, PipelineState

console = Console()

MAX_ITERATIONS = 2  # Maximum extractor→verifier loop iterations


def _make_kg_builder_node():
    """
    KG builder node: finalizes the state by setting final_triplets.
    The actual graph write happens in the runner, not here, so the
    graph stays decoupled from storage.
    """
    def kg_builder_node(state: dict) -> dict:
        pipeline_state = PipelineState.model_validate(state)

        # Baseline mode never runs the verifier node, so verified_triplets
        # stays empty by default. Source from canonical_triplets in that case
        # so baseline results reflect what was actually extracted, not zero.
        source_triplets = (
            pipeline_state.verified_triplets
            if pipeline_state.verifier_ran
            else pipeline_state.canonical_triplets
        )

        # Deduplicate for the final graph
        seen: set[tuple[str, str, str]] = set()
        deduped = []
        for t in source_triplets:
            key = (t.entity1.lower(), t.relation.lower(), t.entity2.lower())
            if key not in seen:
                seen.add(key)
                deduped.append(t)
                
        pipeline_state.final_triplets = deduped
        console.print(
            f"[bold green]KG Builder: {len(source_triplets)} "
            f"{'verified' if pipeline_state.verifier_ran else 'canonical (baseline)'} "
            f"→ {len(pipeline_state.final_triplets)} unique triplets ready for graph.[/bold green]"
        )
        return pipeline_state.model_dump()

    return kg_builder_node


def _make_router(skip_verifier: bool = False):
    """
    Conditional edge router.

    After the verifier node, decide:
      - "extractor" → loop back (too many rejections, under iteration cap)
      - "kg_builder" → proceed to KG storage
    """
    def route_after_verification(state: dict) -> Literal["extractor", "kg_builder"]:
        pipeline_state = PipelineState.model_validate(state)

        if skip_verifier:
            return "kg_builder"

        if (
            pipeline_state.needs_reextraction
            and pipeline_state.iteration < MAX_ITERATIONS
        ):
            console.print(
                f"[yellow]Router: Re-extracting (iteration {pipeline_state.iteration}/{MAX_ITERATIONS}).[/yellow]"
            )
            return "extractor"

        return "kg_builder"

    return route_after_verification


def build_pipeline_graph(
    extractor: ExtractorAgent,
    canonicalizer: CanonicalizerAgent,
    verifier: VerifierAgent | None = None,
    skip_verifier: bool = False,
) -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Compile the LangGraph pipeline.

    Args:
        extractor: Configured ExtractorAgent instance.
        canonicalizer: Configured CanonicalizerAgent instance.
        verifier: Configured VerifierAgent instance. Required unless skip_verifier=True.
        skip_verifier: If True, skips the verifier node (baseline condition).

    Returns:
        Compiled LangGraph that accepts a state dict and returns an updated state dict.
    """
    # Build node functions
    extractor_fn = make_extractor_node(extractor)
    canonicalizer_fn = make_canonicalizer_node(canonicalizer)
    kg_builder_fn = _make_kg_builder_node()

    # Create graph with dict state (LangGraph native)
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("extractor", extractor_fn)
    graph.add_node("canonicalizer", canonicalizer_fn)
    graph.add_node("kg_builder", kg_builder_fn)

    if not skip_verifier:
        if verifier is None:
            raise ValueError("verifier must be provided when skip_verifier=False")
        verifier_fn = make_verifier_node(verifier)
        graph.add_node("verifier", verifier_fn)

    # Wire edges
    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "canonicalizer")

    if skip_verifier:
        # Baseline: extractor → canonicalizer → kg_builder
        graph.add_edge("canonicalizer", "kg_builder")
    else:
        # Full pipeline: canonicalizer → verifier → (conditional) → extractor | kg_builder
        graph.add_edge("canonicalizer", "verifier")
        graph.add_conditional_edges(
            "verifier",
            _make_router(skip_verifier=False),
            {"extractor": "extractor", "kg_builder": "kg_builder"},
        )

    graph.add_edge("kg_builder", END)

    compiled = graph.compile()
    console.print(
        f"[green]Pipeline graph compiled. "
        f"Mode: {'baseline (no verifier)' if skip_verifier else 'full pipeline'}.[/green]"
    )
    return compiled
