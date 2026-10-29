"""Small, dependency-light helpers for per-report experiment audit logging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ingestion.models import CTIReport, PipelineResult


def _json_model(value: Any) -> Any:
    """Convert Pydantic models/enums and nested containers to JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_model(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_model(item) for key, item in value.items()}
    return value


def build_audit_record(
    *,
    report: CTIReport,
    result: PipelineResult,
    condition: str,
    model: str,
    dataset_mode: str,
) -> dict[str, Any]:
    """Build a report-level record without storing report text or credentials."""
    return {
        "schema_version": 1,
        "condition": condition,
        "report_id": report.id,
        "model": model,
        "dataset_mode": dataset_mode,
        "ground_truth_triplets": _json_model(report.ground_truth_triplets),
        "raw_triplets": _json_model(result.raw_triplets),
        "raw_triplets_by_iteration": _json_model(result.raw_triplets_by_iteration),
        "canonical_triplets_by_iteration": _json_model(result.canonical_triplets_by_iteration),
        "total_raw_triplets": result.total_raw_triplets,
        "canonical_triplets": _json_model(result.canonical_triplets),
        "final_triplets": _json_model(result.final_triplets),
        "verified_triplets": _json_model(result.verified_triplets),
        "revised_triplets": _json_model(result.revised_triplets),
        "rejected_triplets": _json_model(result.rejected_triplets),
        "verification_decisions": _json_model(result.verification_decisions),
        "iterations": result.iterations,
        "error": result.error,
    }


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one UTF-8 JSON object and newline to the audit file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
