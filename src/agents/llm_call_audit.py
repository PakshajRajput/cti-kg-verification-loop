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
    report_id: str | None = None,
    stage: str | None = None,
) -> None:
    """Append minimal request-level model attribution without prompt or secret data."""
    path = Path(os.environ.get("LLM_CALL_AUDIT_PATH", "results/llm_call_audit.jsonl"))
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "outcome": outcome,
    }
    if report_id is not None:
        record["report_id"] = report_id
    if stage is not None:
        record["stage"] = stage
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as error:
        # Logging failures must not alter whether the actual model call succeeded.
        print(f"[WARN] Unable to write LLM call audit record: {error}")



def resolve_served_model(response: object, default_model: str) -> str:
    """Use the model ID reported in a provider response, falling back to requested ID."""
    served_model = getattr(response, "model", None)
    return str(served_model or default_model)
