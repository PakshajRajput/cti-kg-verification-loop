"""
Pydantic request/response schemas for the FastAPI API.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw CTI report text to process", min_length=20)
    report_id: Optional[str] = Field(None, description="Optional report ID (auto-generated if not provided)")
    source: Optional[str] = Field(None, description="Report source (e.g., 'Trend Micro')")
    title: Optional[str] = Field(None, description="Report title")


class QueryRequest(BaseModel):
    query_name: str = Field(..., description="Named pre-built Cypher query to run")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional query parameters")


# ─── Response Models ──────────────────────────────────────────────────────────

class TripletResponse(BaseModel):
    entity1: str
    entity1_type: str
    relation: str
    entity2: str
    entity2_type: str
    confidence: float


class VerificationDecisionResponse(BaseModel):
    original: TripletResponse
    verdict: str
    corrected: Optional[TripletResponse] = None
    reasoning: str
    attack_context_ids: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    report_id: str
    final_triplets: list[TripletResponse]
    revised_triplets: list[TripletResponse]
    rejected_triplets: list[TripletResponse]
    verification_decisions: list[VerificationDecisionResponse]
    iterations: int
    total_extracted: int
    hallucination_rate: float
    graph_stats: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    entity_nodes: int = 0
    report_nodes: int = 0


class QueryResponse(BaseModel):
    query_name: str
    results: list[dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    vector_store_populated: bool
    bm25_built: bool
    available_queries: list[str]
