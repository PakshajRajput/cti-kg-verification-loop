"""
Extractor Agent — LangGraph node.

Performs few-shot ICL-based triplet extraction from a CTI report.
Retrieves k similar demonstration reports using embedding cosine similarity,
constructs the prompt, calls GPT-4o, and parses structured JSON output.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from src.agents.llm_client import call_llm_json

from src.ingestion.models import CTIReport, EntityType, PipelineState, Triplet

console = Console()

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "extractor_system.txt"
_FEW_SHOT_PROMPT_PATH = Path(__file__).parent / "prompts" / "extractor_few_shot.txt"


def _load_prompt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _retrieve_demonstrations(
    report: CTIReport,
    demos: list[CTIReport],
    embedder: SentenceTransformer,
    k: int = 5,
) -> list[CTIReport]:
    """Retrieve the k most similar demonstration reports by text embedding."""
    if not demos:
        return []

    query_emb = embedder.encode([report.text], normalize_embeddings=True)[0]
    demo_texts = [d.text for d in demos]
    demo_embs = embedder.encode(demo_texts, normalize_embeddings=True)

    sims = [_cosine_similarity(query_emb, de) for de in demo_embs]
    top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]

    return [demos[i] for i in top_indices]


def _format_demonstrations(demos: list[CTIReport], max_triplets_per_demo: int = 6) -> str:
    """Format demonstration reports as numbered examples in the prompt.

    Two things are capped to keep prompt size predictable and bounded:
      - demo report text (truncated to 300 chars, handled by caller)
      - triplets shown per demo (capped here to max_triplets_per_demo)
    Some CTINexus reports have 15-20+ ground-truth triplets; showing all of
    them for every demo made prompt size effectively unbounded and was the
    real cause of repeated rate-limit failures, independent of few_shot_k.
    Compact (non-indented) JSON is used instead of pretty-printed JSON for
    the same reason — indentation roughly doubles token cost per triplet
    with no benefit to the model's ability to parse the pattern.
    """
    parts = []
    for i, demo in enumerate(demos, 1):
        demo_triplets = demo.ground_truth_triplets[:max_triplets_per_demo]
        triplets_json = json.dumps(
            [
                {
                    "entity1": t.entity1,
                    "entity1_type": t.entity1_type.value,
                    "relation": t.relation,
                    "entity2": t.entity2,
                    "entity2_type": t.entity2_type.value,
                    "confidence": t.confidence,
                }
                for t in demo_triplets
            ],
            separators=(",", ":"),
        )
        parts.append(
            f"### Example {i}\n"
            f"**Text**: {demo.text[:300]}{'...' if len(demo.text) > 300 else ''}\n\n"
            f"**Extracted Triplets**:\n```json\n{triplets_json}\n```"
        )
    return "\n\n---\n\n".join(parts)


def _find_triplet_array(obj: Any, _depth: int = 0) -> list | None:
    """
    Recursively search a parsed JSON structure for a list of triplet-shaped
    dicts, regardless of what key(s) it's nested under.

    Reasoning models (e.g. gpt-oss) sometimes wrap the array in an object
    under an unpredictable key (or nest it under commentary/analysis
    fields), so matching by structure — dicts that actually have the
    expected triplet fields — is more robust than guessing key names.
    """
    if _depth > 4:
        return None
    if isinstance(obj, list):
        if not obj:
            return obj  # Found an empty list (possible empty triplets array)
        if all(isinstance(item, dict) for item in obj):
            sample = obj[0]
            if "entity1" in sample and "entity2" in sample and "relation" in sample:
                return obj
        return None
    if isinstance(obj, dict):
        empty_list = None
        for value in obj.values():
            found = _find_triplet_array(value, _depth + 1)
            if found is not None:
                if not found:
                    empty_list = found
                else:
                    return found
        return empty_list
    return None


def _parse_triplets(data: Any) -> list[Triplet]:
    """Parse JSON data into Triplet objects. Robust to markdown fences. Rejects malformed objects."""
    if isinstance(data, dict):
        if "entity1" in data and "relation" in data and "entity2" in data:
            data = [data]
        else:
            found = _find_triplet_array(data)
            if found is not None:
                console.print("[dim]Extractor: unwrapped JSON object to find the triplet array.[/dim]")
                data = found
            else:
                console.print(f"[yellow]Expected JSON array but got dict: {str(data)[:250]}[/yellow]")

    if not isinstance(data, list):
        if not isinstance(data, dict):
            console.print(f"[yellow]Expected JSON array but got: {type(data)}[/yellow]")
        return []

    triplets = []
    for item in data:
        if not isinstance(item, dict):
            continue
            
        # Extraction Quality Guard
        if "entity1" not in item or "relation" not in item or "entity2" not in item:
            continue
            
        e1 = str(item.get("entity1", "")).strip()
        e2 = str(item.get("entity2", "")).strip()
        rel = str(item.get("relation", "")).strip()
        if not e1 or not e2 or not rel:
            continue

        def safe_entity_type(val: Any) -> EntityType:
            if not val:
                return EntityType.UNKNOWN
            raw = str(val).strip()
            # Try exact match first
            try:
                return EntityType(raw)
            except (ValueError, KeyError):
                pass
            # Fuzzy match: normalize to lowercase, strip spaces/underscores/hyphens
            normalized = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
            lookup = {
                # Malware
                "malware": EntityType.MALWARE,
                "ransomware": EntityType.MALWARE,
                "trojan": EntityType.MALWARE,
                "backdoor": EntityType.MALWARE,
                "rat": EntityType.MALWARE,
                "worm": EntityType.MALWARE,
                "virus": EntityType.MALWARE,
                "rootkit": EntityType.MALWARE,
                "botnet": EntityType.MALWARE,
                "maliciouscode": EntityType.MALWARE,
                "payload": EntityType.MALWARE,
                # Threat Actor
                "threatactor": EntityType.THREAT_ACTOR,
                "aptgroup": EntityType.THREAT_ACTOR,
                "apt": EntityType.THREAT_ACTOR,
                "actor": EntityType.THREAT_ACTOR,
                "group": EntityType.THREAT_ACTOR,
                "hackergroup": EntityType.THREAT_ACTOR,
                "attacker": EntityType.THREAT_ACTOR,
                "adversary": EntityType.THREAT_ACTOR,
                "cybercriminal": EntityType.THREAT_ACTOR,
                "nationstate": EntityType.THREAT_ACTOR,
                "nationstateactor": EntityType.THREAT_ACTOR,
                # Tool
                "tool": EntityType.TOOL,
                "software": EntityType.TOOL,
                "utility": EntityType.TOOL,
                "framework": EntityType.TOOL,
                "hackingtool": EntityType.TOOL,
                "pentest": EntityType.TOOL,
                # Vulnerability
                "vulnerability": EntityType.VULNERABILITY,
                "cve": EntityType.VULNERABILITY,
                "exploit": EntityType.VULNERABILITY,
                "weakness": EntityType.VULNERABILITY,
                "zerodayexploit": EntityType.VULNERABILITY,
                "zeroday": EntityType.VULNERABILITY,
                "0day": EntityType.VULNERABILITY,
                "bug": EntityType.VULNERABILITY,
                # Target
                "target": EntityType.TARGET,
                "victim": EntityType.TARGET,
                "organization": EntityType.TARGET,
                "company": EntityType.TARGET,
                "sector": EntityType.TARGET,
                "industry": EntityType.TARGET,
                "country": EntityType.TARGET,
                "region": EntityType.TARGET,
                "system": EntityType.TARGET,
                "product": EntityType.TARGET,
                "application": EntityType.TARGET,
                "infrastructure": EntityType.TARGET,
                "network": EntityType.TARGET,
                "service": EntityType.TARGET,
                "platform": EntityType.TARGET,
                "server": EntityType.TARGET,
                "device": EntityType.TARGET,
                "data": EntityType.TARGET,
                "asset": EntityType.TARGET,
                "identity": EntityType.TARGET,
                "credential": EntityType.TARGET,
                "securityresearcher": EntityType.TARGET,
                "researcher": EntityType.TARGET,
                "vendor": EntityType.TARGET,
                # Technique
                "technique": EntityType.TECHNIQUE,
                "attacktechnique": EntityType.TECHNIQUE,
                "method": EntityType.TECHNIQUE,
                "attackmethod": EntityType.TECHNIQUE,
                "attackpattern": EntityType.TECHNIQUE,
                "attackvector": EntityType.TECHNIQUE,
                # Tactic
                "tactic": EntityType.TACTIC,
                "attacktactic": EntityType.TACTIC,
                "phase": EntityType.TACTIC,
                "killchainphase": EntityType.TACTIC,
                # Campaign
                "campaign": EntityType.CAMPAIGN,
                "operation": EntityType.CAMPAIGN,
                "incident": EntityType.CAMPAIGN,
                # IOC
                "ioc": EntityType.IOC,
                "indicator": EntityType.IOC,
                "indicatorofcompromise": EntityType.IOC,
                "ipaddress": EntityType.IOC,
                "domain": EntityType.IOC,
                "url": EntityType.IOC,
                "hash": EntityType.IOC,
                "filehash": EntityType.IOC,
                # Unknown
                "unknown": EntityType.UNKNOWN,
                "other": EntityType.UNKNOWN,
            }
            result = lookup.get(normalized, EntityType.UNKNOWN)
            if result == EntityType.UNKNOWN and normalized != "unknown":
                console.print(f"[dim yellow]  ⚠ Unrecognized entity_type '{raw}' → defaulting to Unknown[/dim yellow]")
            return result

        triplets.append(Triplet(
            entity1=e1,
            entity1_type=safe_entity_type(item.get("entity1_type", "Unknown")),
            relation=rel.lower().replace(" ", "-"),
            entity2=e2,
            entity2_type=safe_entity_type(item.get("entity2_type", "Unknown")),
            confidence=float(item.get("confidence", 1.0)),
        ))

    return triplets


class ExtractorAgent:
    """
    LangGraph-compatible extractor agent.

    Wraps the LLM call, demonstration retrieval, and prompt assembly.
    Instantiate once and reuse across pipeline runs.
    """

    def __init__(
        self,
        demonstrations: list[CTIReport],
        model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        few_shot_k: int = 5,
        max_report_chars: int = 2500,
    ):
        self.demonstrations = demonstrations
        self.model = model
        self.few_shot_k = few_shot_k
        self.max_report_chars = max_report_chars
        self.embedder = SentenceTransformer(embedding_model)
        self.system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH)
        self.few_shot_template = _load_prompt(_FEW_SHOT_PROMPT_PATH)

    def _call_llm(self, user_message: str) -> Any:
        """Call LLM."""
        return call_llm_json(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

    def extract(self, report: CTIReport, feedback: str = "") -> list[Triplet]:
        """Extract triplets from a single CTI report.

        Args:
            feedback: Optional verifier feedback from a prior failed iteration.
                When present, it's appended to the prompt so a retry can
                actually differ from the first attempt instead of
                regenerating identical output at temperature=0.0.
        """
        import time
        start_time = time.time()
        
        # Retrieve k similar demonstrations
        demos = _retrieve_demonstrations(report, self.demonstrations, self.embedder, k=self.few_shot_k)
        demo_text = _format_demonstrations(demos)

        # Truncate very long report text so the prompt stays within the
        # account's per-minute token budget. Most CTI reports fit well
        # under this; only unusually long ones are affected.
        report_text = report.text
        if len(report_text) > self.max_report_chars:
            report_text = report_text[: self.max_report_chars] + "\n[...truncated for length...]"

        # Build user message
        user_message = self.few_shot_template.format(
            k=len(demos),
            demonstrations=demo_text if demo_text else "(No demonstration examples available.)",
            report_id=report.id,
            source=report.source or "Unknown",
            report_text=report_text,
        )

        if feedback:
            user_message += f"\n\n## Verifier Feedback From Previous Attempt\n{feedback}"
            console.print("[magenta]Extractor: retrying with verifier feedback.[/magenta]")

        console.print(f"[cyan]Extractor: processing report {report.id} ({len(demos)} demos)...[/cyan]")

        try:
            data, recovery_method, _ = self._call_llm(user_message)
            if data is None:
                data = []
        except Exception as e:
            console.print(f"[red]Extractor LLM call failed: {e}[/red]")
            data = []
            recovery_method = "failed_exception"
            
        elapsed = time.time() - start_time
        
        triplets = _parse_triplets(data)
        console.print(f"[green]Extractor: extracted {len(triplets)} raw triplets.[/green]")
        
        # Write structured logging
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{report.id}.log"
        summary_file = log_dir / "pipeline_summary.log"
        
        log_content = (
            f"Report ID: {report.id}\n"
            f"Model: {self.model}\n"
            f"Response Time: {elapsed:.2f}s\n"
            f"Recovery Method: {recovery_method}\n"
            f"Parsing Status: {'success' if triplets else 'failed'}\n"
            f"Recovered Triplets Count: {len(triplets)}\n"
        )
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
            
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"{report.id} | {recovery_method} | {len(triplets)} triplets | {elapsed:.2f}s\n")
            
        return triplets


def make_extractor_node(agent: ExtractorAgent):
    """
    Factory: returns a LangGraph node function bound to an ExtractorAgent instance.

    The returned function signature matches what LangGraph expects:
        node(state: dict) -> dict
    """
    def extractor_node(state: dict) -> dict:
        pipeline_state = PipelineState.model_validate(state)
        report = pipeline_state.report
        triplets = agent.extract(report, feedback=pipeline_state.reextraction_feedback)
        pipeline_state.raw_triplets = triplets
        pipeline_state.total_raw_triplets_all_iterations += len(triplets)
        pipeline_state.iteration += 1
        return pipeline_state.model_dump()

    return extractor_node
