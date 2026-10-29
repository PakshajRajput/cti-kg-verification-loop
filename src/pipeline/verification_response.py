"""Dependency-light parsing and validation for verifier model responses."""
from __future__ import annotations

import json
import re


def parse_verification_response(raw: str) -> dict:
    """Parse a verifier response and reject ambiguous failures rather than fail open."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise ValueError("Verifier JSON response must be an object")
            verdict = str(parsed.get("verdict", "")).lower()
            if verdict not in {"valid", "corrected", "hallucinated"}:
                raise ValueError(f"Unsupported or missing verifier verdict: {verdict!r}")
            return parsed
        except json.JSONDecodeError:
            pass

    # Conservative fallback for truncated/malformed JSON.
    lower_raw = raw.lower()
    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).lower()
        if verdict not in {"valid", "corrected", "hallucinated"}:
            raise ValueError(f"Unsupported verifier verdict: {verdict!r}")
    elif "hallucinated" in lower_raw or "invalid" in lower_raw or "rejected" in lower_raw:
        verdict = "hallucinated"
    elif "corrected" in lower_raw:
        # Without a reliably parsed corrected triplet, do not accept as valid.
        verdict = "hallucinated"
    elif re.search(r"""["']?verdict["']?\s*[:=]\s*["']?valid\b""", raw, re.IGNORECASE):
        verdict = "valid"
    else:
        raise ValueError("Verifier response did not contain a recognizable verdict")

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    reasoning = reasoning_match.group(1) if reasoning_match else (
        raw[:200] + "..." if len(raw) > 200 else raw
    )
    return {
        "verdict": verdict,
        "corrected_triplet": None,
        "reasoning": f"(Recovered via fallback) {reasoning}",
    }
