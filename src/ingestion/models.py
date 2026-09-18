"""
Shared Pydantic models used across the entire pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Entity Types ─────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    MALWARE = "Malware"
    THREAT_ACTOR = "ThreatActor"
    TOOL = "Tool"
    VULNERABILITY = "Vulnerability"
    TARGET = "Target"
    TECHNIQUE = "Technique"
    TACTIC = "Tactic"
    CAMPAIGN = "Campaign"
    IOC = "IOC"
    UNKNOWN = "Unknown"


# ─── Core Data Models ─────────────────────────────────────────────────────────

class Triplet(BaseModel):
    """A single subject–relation–object triplet extracted from a CTI report."""
    entity1: str
    entity1_type: EntityType = EntityType.UNKNOWN
    relation: str
    entity2: str
    entity2_type: EntityType = EntityType.UNKNOWN
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = None  # snippet from the report that supports this

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triplet):
            return False
        return (
            self.entity1.lower() == other.entity1.lower()
            and self.relation.lower() == other.relation.lower()
            and self.entity2.lower() == other.entity2.lower()
        )

    def __hash__(self) -> int:
        return hash((self.entity1.lower(), self.relation.lower(), self.entity2.lower()))

    def to_text(self) -> str:
        return f"{self.entity1} {self.relation} {self.entity2}"


class CTIReport(BaseModel):
    """A single CTI report with optional ground-truth annotations."""
    id: str
    source: str = ""          # e.g. "Trend Micro", "Symantec"
    title: str = ""
    text: str
    ground_truth_entities: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_relations: list[dict[str, Any]] = Field(default_factory=list)
    ground_truth_triplets: list[Triplet] = Field(default_factory=list)
    split: str = "train"      # "train" | "test"


class ATT_CK_Entry(BaseModel):
    """A single MITRE ATT&CK object (technique, group, software, etc.)."""
    id: str                   # e.g. "T1566", "G0016", "S0002"
    name: str
    description: str
    type: str                 # "technique" | "group" | "software" | "mitigation" | "tactic"
    external_references: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    tactics: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Concatenated text used for embedding and BM25 indexing."""
        parts = [f"{self.type.upper()}: {self.name} ({self.id})"]
        if self.aliases:
            parts.append(f"Also known as: {', '.join(self.aliases)}")
        if self.tactics:
            parts.append(f"Tactics: {', '.join(self.tactics)}")
        if self.platforms:
            parts.append(f"Platforms: {', '.join(self.platforms)}")
        parts.append(self.description[:1000])  # cap to avoid token bloat
        return "\n".join(parts)


# ─── Verification Models ──────────────────────────────────────────────────────

class VerificationVerdict(str, Enum):
    VALID = "valid"
    CORRECTED = "corrected"
    HALLUCINATED = "hallucinated"


class VerificationDecision(BaseModel):
    """Verifier agent's decision for a single triplet."""
    original_triplet: Triplet
    verdict: VerificationVerdict
    corrected_triplet: Optional[Triplet] = None
    reasoning: str = ""
    retrieved_context_ids: list[str] = Field(default_factory=list)  # ATT&CK IDs used


class EvalMetrics(BaseModel):
    """Evaluation results for a single condition."""
    condition: str            # "baseline" | "full_pipeline"
    precision: float
    recall: float
    f1: float
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    hallucination_rate: float = 0.0
    total_extracted: int = 0
    total_verified: int = 0
    total_rejected: int = 0
    total_ground_truth: int = 0
    num_reports: int = 0


# ─── Pipeline State ───────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """Mutable state passed between LangGraph nodes."""
    report: CTIReport
    # Extractor outputs
    raw_triplets: list[Triplet] = Field(default_factory=list)
    total_raw_triplets_all_iterations: int = 0
    # Canonicalizer outputs
    canonical_triplets: list[Triplet] = Field(default_factory=list)
    # Verifier outputs
    verified_triplets: list[Triplet] = Field(default_factory=list)
    revised_triplets: list[Triplet] = Field(default_factory=list)
    rejected_triplets: list[Triplet] = Field(default_factory=list)
    verification_decisions: list[VerificationDecision] = Field(default_factory=list)
    # Control flow
    iteration: int = 0
    needs_reextraction: bool = False
    reextraction_feedback: str = ""
    verifier_ran: bool = False
    error: Optional[str] = None
    # Final result
    final_triplets: list[Triplet] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Final output of a single pipeline run."""
    report_id: str
    final_triplets: list[Triplet]
    revised_triplets: list[Triplet]
    rejected_triplets: list[Triplet]
    verification_decisions: list[VerificationDecision]
    iterations: int
    total_raw_triplets: int = 0
    error: Optional[str] = None

    @property
    def hallucination_rate(self) -> float:
        total = len(self.final_triplets) + len(self.rejected_triplets)
        if total == 0:
            return 0.0
        return len(self.rejected_triplets) / total
