"""
Pipeline Runner.

High-level interface for running the compiled LangGraph pipeline
on a single CTI report, returning a structured PipelineResult.

Also exposes a factory function to build the full pipeline from config,
which handles agent construction and dependency injection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.agents.canonicalizer import CanonicalizerAgent
from src.agents.extractor import ExtractorAgent
from src.agents.verifier import VerifierAgent
from src.ingestion.models import CTIReport, PipelineResult, PipelineState, Triplet
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
    skip_verifier: bool = False  # True = baseline condition
    data_dir: Path = field(default_factory=lambda: Path("data"))


class PipelineRunner:
    """
    End-to-end pipeline runner.

    Encapsulates all agents and the compiled graph.
    Use build() to construct from config, then call run() for each report.
    """

    def __init__(
        self,
        compiled_graph,
        config: PipelineConfig,
    ):
        self.graph = compiled_graph
        self.config = config

    @classmethod
    def build(
        cls,
        config: PipelineConfig,
        demonstrations: list[CTIReport],
        retriever: Optional[HybridRetriever] = None,
    ) -> "PipelineRunner":
        """
        Build a complete pipeline runner from config.

        Args:
            config: PipelineConfig with model and path settings.
            demonstrations: Training reports used as few-shot examples by the extractor.
            retriever: Pre-built HybridRetriever. Required unless skip_verifier=True.
        """
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
        """
        Run the pipeline on a single CTI report.

        Args:
            report: The CTI report to process.

        Returns:
            PipelineResult with final verified triplets and metadata.
        """
        console.print(f"\n[bold cyan]─── Processing Report: {report.id} ───[/bold cyan]")

        # Build initial state
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
