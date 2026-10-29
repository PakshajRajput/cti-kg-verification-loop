"""
Pipeline Runner.

High-level interface for running the compiled LangGraph pipeline
on a single CTI report, returning a structured PipelineResult.

Also exposes a factory function to build the full pipeline from config,
which handles agent construction and dependency injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.agents.canonicalizer import CanonicalizerAgent
from src.agents.extractor import ExtractorAgent
from src.agents.verifier import VerifierAgent
from src.ingestion.models import CTIReport, PipelineResult, PipelineState
from src.pipeline.graph import build_pipeline_graph
from src.rag.hybrid_retriever import HybridRetriever

console = Console()


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    model: str = "gpt-4o"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    few_shot_k: int = 5
    rag_top_k: int = 5
    similarity_threshold: float = 0.85
    skip_verifier: bool = False
    data_dir: Path = field(default_factory=lambda: Path("data"))


class PipelineRunner:
    """End-to-end pipeline runner; call build() once and run() per report."""

    def __init__(self, compiled_graph, config: PipelineConfig):
        self.graph = compiled_graph
        self.config = config

    @classmethod
    def build(
        cls,
        config: PipelineConfig,
        demonstrations: list[CTIReport],
        retriever: Optional[HybridRetriever] = None,
    ) -> "PipelineRunner":
        """Build a configured runner from demonstrations and an optional retriever."""
        console.print(
            f"[cyan]Building pipeline: model={config.model}, "
            f"skip_verifier={config.skip_verifier}[/cyan]"
        )
        extractor = ExtractorAgent(
            demonstrations=demonstrations,
            model=config.model,
            embedding_model=config.embedding_model,
            few_shot_k=config.few_shot_k,
        )
        canonicalizer = CanonicalizerAgent(
            model=config.model,
            embedding_model=config.embedding_model,
            similarity_threshold=config.similarity_threshold,
        )

        verifier: Optional[VerifierAgent] = None
        if not config.skip_verifier:
            if retriever is None:
                raise ValueError("retriever required for full pipeline")
            verifier = VerifierAgent(
                retriever=retriever,
                model=config.model,
                rag_top_k=config.rag_top_k,
            )

        graph = build_pipeline_graph(
            extractor=extractor,
            canonicalizer=canonicalizer,
            verifier=verifier,
            skip_verifier=config.skip_verifier,
        )
        return cls(graph, config)

    def run(self, report: CTIReport) -> PipelineResult:
        """Run pipeline for a report and return an auditable structured result."""
        console.print(f"\n[bold cyan]─── Processing Report: {report.id} ───[/bold cyan]")
        initial_state = PipelineState(report=report)
        initial_state_dict = initial_state.model_dump()

        try:
            final_state_dict = self.graph.invoke(initial_state_dict)
            final_state = PipelineState.model_validate(final_state_dict)
            return PipelineResult(
                report_id=report.id,
                final_triplets=final_state.final_triplets,
                revised_triplets=final_state.revised_triplets,
                rejected_triplets=final_state.rejected_triplets,
                verification_decisions=final_state.verification_decisions,
                iterations=final_state.iteration,
                total_raw_triplets=final_state.total_raw_triplets_all_iterations,
                raw_triplets=final_state.raw_triplets,
                raw_triplets_by_iteration=final_state.raw_triplets_by_iteration,
                canonical_triplets_by_iteration=final_state.canonical_triplets_by_iteration,
                canonical_triplets=final_state.canonical_triplets,
                verified_triplets=final_state.verified_triplets,
                error=final_state.error,
            )
        except Exception as e:
            console.print(f"[red]Pipeline error for report {report.id}: {e}[/red]")
            return PipelineResult(
                report_id=report.id,
                final_triplets=[],
                revised_triplets=[],
                rejected_triplets=[],
                verification_decisions=[],
                iterations=0,
                error=str(e),
            )
