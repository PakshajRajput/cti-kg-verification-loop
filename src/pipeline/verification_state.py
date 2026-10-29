"""Small, dependency-light helpers for applying verifier outputs to pipeline state."""
from __future__ import annotations

from src.ingestion.models import (
    PipelineState,
    Triplet,
    VerificationDecision,
    VerificationVerdict,
)


def apply_verification_outputs(
    state: PipelineState,
    *,
    verified: list[Triplet],
    rejected: list[Triplet],
    decisions: list[VerificationDecision],
) -> PipelineState:
    """Store only the current attempt's outputs and retain an iteration-tagged history."""
    if not state.canonical_triplets:
        # An empty retry is a real empty final output, not permission to reuse
        # accepted/rejected results from an earlier attempt.
        state.verified_triplets = []
        state.rejected_triplets = []
        state.revised_triplets = []
        state.needs_reextraction = False
        state.reextraction_feedback = ""
        return state

    current_iteration_decisions = [
        decision.model_copy(update={"iteration": state.iteration})
        for decision in decisions
    ]
    state.verified_triplets = verified
    state.rejected_triplets = rejected
    state.revised_triplets = [
        decision.corrected_triplet
        for decision in current_iteration_decisions
        if decision.verdict == VerificationVerdict.CORRECTED and decision.corrected_triplet
    ]
    state.verification_decisions.extend(current_iteration_decisions)
    return state
