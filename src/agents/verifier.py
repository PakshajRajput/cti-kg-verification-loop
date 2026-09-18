"""
Verifier Agent — LangGraph node.

For each extracted triplet:
  1. Constructs a query string from the triplet
  2. Retrieves relevant MITRE ATT&CK context via HybridRetriever
  3. Calls LLM to decide: valid / corrected / hallucinated
  4. Aggregates results into verified_triplets and rejected_triplets
  5. Sets needs_reextraction flag if too many triplets are rejected

The verifier is the core research contribution — it grounds LLM output
in a structured, verifiable knowledge base rather than relying on
uncorroborated LLM memory.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rich.console import Console

from src.agents.llm_client import call_llm

from src.ingestion.models import (
    EntityType,
    PipelineState,
    Triplet,
    VerificationDecision,
    VerificationVerdict,
)
from src.rag.hybrid_retriever import HybridRetriever

console = Console()

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "verifier_system.txt"

# If >50% of triplets are rejected, flag for re-extraction
REEXTRACTION_THRESHOLD = 0.5


def _load_prompt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _format_context(retrieved: list[dict]) -> str:
    """Format retrieved ATT&CK entries as context for the verifier LLM."""
    if not retrieved:
        return "(No relevant MITRE ATT&CK entries found for this triplet.)"
    parts = []
    for i, hit in enumerate(retrieved, 1):
        parts.append(
            f"**[{i}] {hit.get('name', 'Unknown')} ({hit.get('attack_id', 'N/A')})**\n"
            f"{hit.get('document', '')[:600]}"
        )
    return "\n\n".join(parts)


def _parse_verification_response(raw: str) -> dict:
    """Parse verifier LLM JSON output. Robust to markdown fences and truncation."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    
    # Try standard JSON parse first
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    # Fallback: manually extract fields using regex if JSON is malformed or truncated
    lower_raw = raw.lower()
    
    # Extract verdict
    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).lower()
        if verdict not in ["valid", "corrected", "hallucinated"]:
            verdict = "valid"
    else:
        # Fallback to keyword matching
        if "hallucinated" in lower_raw or "invalid" in lower_raw or "rejected" in lower_raw:
            verdict = "hallucinated"
        elif "corrected" in lower_raw:
            verdict = "hallucinated" # Safer fallback if we can't parse the corrected triplet
        else:
            verdict = "valid"
            
    # Extract reasoning
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1)
    else:
        # Just use the raw string (truncated if too long)
        reasoning = raw[:200] + "..." if len(raw) > 200 else raw
        
    return {
        "verdict": verdict,
        "corrected_triplet": None,  # Difficult to parse reliably without valid JSON
        "reasoning": f"(Recovered via fallback) {reasoning}"
    }


class VerifierAgent:
    """
    RAG-grounded verification agent.

    Checks each triplet against MITRE ATT&CK context retrieved via HybridRetriever.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = "gpt-4o",
        rag_top_k: int = 5,
    ):
        self.retriever = retriever
        self.model = model
        self.rag_top_k = rag_top_k
        self.system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH)

    def _call_llm(self, triplet: Triplet, context: str) -> str:
        """Call verifier LLM for a single triplet."""
        user_message = (
            f"## Triplet to Verify\n"
            f"- **Entity 1**: {triplet.entity1} (type: {triplet.entity1_type.value})\n"
            f"- **Relation**: {triplet.relation}\n"
            f"- **Entity 2**: {triplet.entity2} (type: {triplet.entity2_type.value})\n"
            f"- **Extraction Confidence**: {triplet.confidence:.2f}\n\n"
            f"## Relevant MITRE ATT&CK Context\n{context}\n\n"
            f"Verify this triplet and return your verdict as JSON."
        )
        return call_llm(
            messages=[
                {"role": "system", "content": self.system_prompt + "\nReturn ONLY valid JSON."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )

    def _parse_corrected_triplet(self, data: dict | None) -> Triplet | None:
        """Parse corrected_triplet dict from verifier response."""
        if not data or not isinstance(data, dict):
            return None
        e1 = str(data.get("entity1", "")).strip()
        e2 = str(data.get("entity2", "")).strip()
        rel = str(data.get("relation", "")).strip()
        if not e1 or not e2 or not rel:
            return None

        def safe_type(val) -> EntityType:
            try:
                return EntityType(val)
            except (ValueError, KeyError):
                return EntityType.UNKNOWN

        return Triplet(
            entity1=e1,
            entity1_type=safe_type(data.get("entity1_type", "Unknown")),
            relation=rel,
            entity2=e2,
            entity2_type=safe_type(data.get("entity2_type", "Unknown")),
            confidence=float(data.get("confidence", 0.9)),
        )

    def verify_triplets(
        self, triplets: list[Triplet]
    ) -> tuple[list[Triplet], list[Triplet], list[VerificationDecision], bool]:
        """
        Verify a list of triplets.

        Returns:
            verified_triplets: Triplets that are valid or corrected
            rejected_triplets: Hallucinated/invalid triplets
            decisions: Full decision record for each triplet
            needs_reextraction: True if rejection rate > REEXTRACTION_THRESHOLD
        """
        verified: list[Triplet] = []
        rejected: list[Triplet] = []
        decisions: list[VerificationDecision] = []

        for triplet in triplets:
            if triplet.entity1_type == EntityType.UNKNOWN or triplet.entity2_type == EntityType.UNKNOWN:
                console.print(f"[dim yellow]  ⚠ Unknown entity type in: '{triplet.to_text()}' — sending to LLM verifier[/dim yellow]")

            # Build query from triplet
            query = f"{triplet.entity1} {triplet.relation} {triplet.entity2}"
            query += f" {triplet.entity1_type.value} {triplet.entity2_type.value}"

            # Retrieve ATT&CK context
            retrieved = self.retriever.retrieve(query, top_k=self.rag_top_k)
            context = _format_context(retrieved)
            attack_ids = [r.get("attack_id", "") for r in retrieved if r.get("attack_id")]

            # LLM verification
            try:
                raw = self._call_llm(triplet, context)
                parsed = _parse_verification_response(raw)
            except Exception as e:
                console.print(f"[yellow]Verifier LLM failed for triplet '{triplet.to_text()}': {e}[/yellow]")
                parsed = {"verdict": "valid", "corrected_triplet": None, "reasoning": f"LLM error: {e}"}

            verdict_str = parsed.get("verdict", "valid").lower()
            try:
                verdict = VerificationVerdict(verdict_str)
            except ValueError:
                verdict = VerificationVerdict.VALID

            corrected = self._parse_corrected_triplet(parsed.get("corrected_triplet"))
            reasoning = str(parsed.get("reasoning", ""))

            decision = VerificationDecision(
                original_triplet=triplet,
                verdict=verdict,
                corrected_triplet=corrected,
                reasoning=reasoning,
                retrieved_context_ids=attack_ids,
            )
            decisions.append(decision)

            if verdict == VerificationVerdict.HALLUCINATED:
                rejected.append(triplet)
                console.print(f"[red]  ✗ HALLUCINATED: '{triplet.to_text()}'[/red]")
            elif verdict == VerificationVerdict.CORRECTED and corrected:
                verified.append(corrected)
                console.print(f"[yellow]  ~ CORRECTED: '{triplet.to_text()}' → '{corrected.to_text()}'[/yellow]")
            else:
                verified.append(triplet)
                console.print(f"[green]  ✓ VALID: '{triplet.to_text()}'[/green]")

        total = len(triplets)
        
        valid_count = sum(1 for d in decisions if d.verdict == VerificationVerdict.VALID)
        revised_count = sum(1 for d in decisions if d.verdict == VerificationVerdict.CORRECTED)
        rejected_count = sum(1 for d in decisions if d.verdict == VerificationVerdict.HALLUCINATED)
        
        assert valid_count + revised_count + rejected_count == total, (
            f"Triplet tracking mismatch! Valid: {valid_count}, Revised: {revised_count}, Rejected: {rejected_count}, Total: {total}"
        )

        rejection_rate = len(rejected) / total if total > 0 else 0.0
        needs_reextraction = rejection_rate > REEXTRACTION_THRESHOLD

        console.print(
            f"[cyan]Verifier: {len(verified)} verified, {len(rejected)} rejected "
            f"({rejection_rate:.0%} rejection rate). "
            f"Re-extract: {needs_reextraction}[/cyan]"
        )

        return verified, rejected, decisions, needs_reextraction


def _build_reextraction_feedback(decisions: list[VerificationDecision], max_items: int = 6) -> str:
    """
    Summarize why triplets were rejected/corrected so the extractor can
    actually change its output on retry, instead of regenerating the
    same deterministic result.
    """
    bad = [
        d for d in decisions
        if d.verdict in (VerificationVerdict.HALLUCINATED, VerificationVerdict.CORRECTED)
    ]
    if not bad:
        return ""

    lines = [
        "The previous extraction attempt had the following triplets flagged by the verifier. "
        "Avoid repeating these exact mistakes — either omit unsupported claims entirely, or "
        "phrase them more precisely and conservatively:"
    ]
    for d in bad[:max_items]:
        tag = "REJECTED" if d.verdict == VerificationVerdict.HALLUCINATED else "CORRECTED"
        lines.append(f"- [{tag}] '{d.original_triplet.to_text()}' — reason: {d.reasoning}")
    return "\n".join(lines)


def make_verifier_node(agent: VerifierAgent):
    """Factory: returns a LangGraph node function for the verifier."""
    def verifier_node(state: dict) -> dict:
        pipeline_state = PipelineState.model_validate(state)
        triplets = pipeline_state.canonical_triplets
        pipeline_state.verifier_ran = True

        if not triplets:
            pipeline_state.needs_reextraction = False
            return pipeline_state.model_dump()

        verified, rejected, decisions, needs_reextraction = agent.verify_triplets(triplets)

        pipeline_state.verified_triplets.extend(verified)
        pipeline_state.rejected_triplets.extend(rejected)
        pipeline_state.verification_decisions.extend(decisions)
        
        revised = [d.corrected_triplet for d in decisions if d.verdict == VerificationVerdict.CORRECTED and d.corrected_triplet]
        pipeline_state.revised_triplets.extend(revised)

        pipeline_state.needs_reextraction = (
            needs_reextraction and pipeline_state.iteration < 2
        )
        pipeline_state.reextraction_feedback = (
            _build_reextraction_feedback(decisions) if pipeline_state.needs_reextraction else ""
        )
        return pipeline_state.model_dump()

    return verifier_node
