"""Dependency-free audit records for LLM provider/model attribution."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def log_model_call(
    *,
    requested_model: str,
    actual_model: str,
    provider: str,
    outcome: str,
) -> None:
    """Append minimal request-level model attribution without prompt or secret data."""
    path = Path(os.environ.get("LLM_CALL_AUDIT_PATH", "results/llm_call_audit.jsonl"))
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "outcome": outcome,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\\n")
    except OSError as error:
        # Logging failures must not alter whether the actual model call succeeded.
        print(f"[WARN] Unable to write LLM call audit record: {error}")
