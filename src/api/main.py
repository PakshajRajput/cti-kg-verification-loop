"""
FastAPI Application.

Exposes the CTI extraction pipeline as a REST API.

Endpoints:
  POST /extract          — Run full pipeline (extractor + verifier + KG write)
  POST /extract/baseline — Run baseline (extractor only, no verifier)
  GET  /graph/stats      — Graph node/edge counts
  GET  /health           — Pipeline component health check

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

load_dotenv()

from src.api.schemas import (
    ExtractRequest,
    ExtractResponse,
    GraphStatsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TripletResponse,
    VerificationDecisionResponse,
)
from src.ingestion.models import CTIReport, Triplet, VerificationDecision
from src.ingestion.dataset_loader import load_ctinexus_dataset
from src.kg.networkx_writer import NetworkXWriter
from src.pipeline.runner import PipelineConfig, PipelineRunner
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.bm25_index import ATTACKBm25Index
from src.rag.vector_store import ATTACKVectorStore

console = Console()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


# ─── Global State (loaded at startup) ─────────────────────────────────────────

class AppState:
    full_runner: PipelineRunner | None = None
    baseline_runner: PipelineRunner | None = None
    kg_writer: NetworkXWriter | None = None
    vector_store: ATTACKVectorStore | None = None
    bm25: ATTACKBm25Index | None = None
    ready: bool = False


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all pipeline components at startup."""
    console.print("[cyan]Starting CTI KG API — loading components...[/cyan]")

    try:
        # Load dataset (training split only needed for demonstrations)
        train_reports, _ = load_ctinexus_dataset(DATA_DIR)

        # Build retriever
        retriever = HybridRetriever.from_disk(
            DATA_DIR,
            embedding_model=EMBEDDING_MODEL,
            reranker_model=RERANKER_MODEL,
        )
        _state.vector_store = retriever.vector_store
        _state.bm25 = retriever.bm25_index

        # Build pipeline runners
        full_config = PipelineConfig(
            model=LLM_MODEL,
            embedding_model=EMBEDDING_MODEL,
            reranker_model=RERANKER_MODEL,
            skip_verifier=False,
            data_dir=DATA_DIR,
        )
        _state.full_runner = PipelineRunner.build(full_config, train_reports, retriever=retriever)

        baseline_config = PipelineConfig(
            model=LLM_MODEL,
            embedding_model=EMBEDDING_MODEL,
            skip_verifier=True,
            data_dir=DATA_DIR,
        )
        _state.baseline_runner = PipelineRunner.build(baseline_config, train_reports)

        # Connect to NetworkX
        _state.kg_writer = NetworkXWriter("data/kg.json")
        try:
            _state.kg_writer.connect()
        except Exception as e:
            console.print(f"[yellow]Graph load warning: {e}. KG writes will be skipped.[/yellow]")
            _state.kg_writer = None

        _state.ready = True
        console.print("[bold green]API ready.[/bold green]")

    except Exception as e:
        console.print(f"[red]Startup error: {e}[/red]")
        _state.ready = False

    yield  # App runs here

    # Cleanup
    if _state.kg_writer:
        _state.kg_writer.close()
    console.print("[cyan]API shutdown complete.[/cyan]")


# ─── App Instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CTI Knowledge Graph Pipeline API",
    description=(
        "Verification-in-the-Loop CTI extraction pipeline. "
        "Extracts entity-relation triplets from CTI reports and stores them in a NetworkX graph."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _triplet_to_response(t: Triplet) -> TripletResponse:
    return TripletResponse(
        entity1=t.entity1,
        entity1_type=t.entity1_type.value,
        relation=t.relation,
        entity2=t.entity2,
        entity2_type=t.entity2_type.value,
        confidence=t.confidence,
    )


def _decision_to_response(d: VerificationDecision) -> VerificationDecisionResponse:
    return VerificationDecisionResponse(
        original=_triplet_to_response(d.original_triplet),
        verdict=d.verdict.value,
        corrected=_triplet_to_response(d.corrected_triplet) if d.corrected_triplet else None,
        reasoning=d.reasoning,
        attack_context_ids=d.retrieved_context_ids,
    )


def _run_and_respond(
    request: ExtractRequest,
    runner: PipelineRunner,
    skip_kg: bool = False,
) -> ExtractResponse:
    """Run pipeline and build response dict."""
    if not _state.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Check server logs.")

    report_id = request.report_id or f"api_{uuid.uuid4().hex[:8]}"
    report = CTIReport(
        id=report_id,
        text=request.text,
        source=request.source or "API",
        title=request.title or "",
    )

    result = runner.run(report)

    graph_stats = None
    if not skip_kg and _state.kg_writer and not result.error:
        try:
            _state.kg_writer.write_triplets(result.final_triplets)
            graph_stats = _state.kg_writer.get_stats()
        except Exception as e:
            console.print(f"[yellow]Graph write warning: {e}[/yellow]")

    total_extracted = result.total_raw_triplets
    hall_rate = (
        len(result.rejected_triplets) / total_extracted if total_extracted > 0 else 0.0
    )

    return ExtractResponse(
        report_id=report_id,
        final_triplets=[_triplet_to_response(t) for t in result.final_triplets],
        revised_triplets=[_triplet_to_response(t) for t in result.revised_triplets],
        rejected_triplets=[_triplet_to_response(t) for t in result.rejected_triplets],
        verification_decisions=[_decision_to_response(d) for d in result.verification_decisions],
        iterations=result.iterations,
        total_extracted=total_extracted,
        hallucination_rate=round(hall_rate, 4),
        graph_stats=graph_stats,
        error=result.error,
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractResponse, tags=["Pipeline"])
async def extract_full(request: ExtractRequest) -> ExtractResponse:
    """
    Run the full pipeline (Extractor → Canonicalizer → Verifier) on a CTI report.
    Writes verified triplets to NetworkX and returns the result.
    """
    if not _state.full_runner:
        raise HTTPException(status_code=503, detail="Full pipeline not initialized.")
    return _run_and_respond(request, _state.full_runner)


@app.post("/extract/baseline", response_model=ExtractResponse, tags=["Pipeline"])
async def extract_baseline(request: ExtractRequest) -> ExtractResponse:
    """
    Run the baseline pipeline (Extractor + Canonicalizer only, no verifier).
    Useful for comparing against the full pipeline.
    """
    if not _state.baseline_runner:
        raise HTTPException(status_code=503, detail="Baseline pipeline not initialized.")
    return _run_and_respond(request, _state.baseline_runner, skip_kg=True)


@app.get("/graph/stats", response_model=GraphStatsResponse, tags=["Knowledge Graph"])
async def graph_stats() -> GraphStatsResponse:
    """Return current graph statistics (node/edge counts)."""
    if not _state.kg_writer:
        raise HTTPException(status_code=503, detail="Graph not connected.")
    stats = _state.kg_writer.get_stats()
    return GraphStatsResponse(**stats)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Check the status of all pipeline components."""
    neo4j_ok = _state.kg_writer is not None

    vs_populated = _state.vector_store.is_populated() if _state.vector_store else False
    bm25_built = _state.bm25.is_built() if _state.bm25 else False

    return HealthResponse(
        status="ready" if _state.ready else "initializing",
        neo4j_connected=neo4j_ok,
        vector_store_populated=vs_populated,
        bm25_built=bm25_built,
        available_queries=[],
    )


@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "name": "CTI Knowledge Graph Pipeline API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
