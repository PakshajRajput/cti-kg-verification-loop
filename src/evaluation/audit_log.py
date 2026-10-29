"""Small, dependency-light helpers for per-report experiment audit logging."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

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


def sha256_file(path: Path) -> str | None:
    """Return a file hash, or None when the optional file is unavailable."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_configured_fallback_model() -> str:
    """Return the configured fallback model name from environment or project default."""
    env_model = os.environ.get("FALLBACK_MODEL")
    if env_model:
        return env_model
    try:
        from src.agents.llm_client import FALLBACK_MODEL
        return FALLBACK_MODEL
    except Exception:
        provider = os.environ.get("LLM_PROVIDER", "groq").lower()
        return "openai/gpt-oss-20b" if provider == "groq" else "gpt-4o-mini"


def dataset_fingerprint(reports: Iterable[CTIReport]) -> str:
    """Hash stable report/annotation content without exposing report text in output."""
    digest = hashlib.sha256()
    records = []
    for report in sorted(reports, key=lambda item: item.id):
        records.append({
            "id": report.id,
            "source": report.source,
            "text_sha256": hashlib.sha256(report.text.encode("utf-8")).hexdigest(),
            "ground_truth_triplets": _json_model(report.ground_truth_triplets),
        })
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(encoded)
    return digest.hexdigest()


def build_experiment_manifest(
    *,
    started_at_utc: str,
    condition: str,
    model: str,
    embedding_model: str,
    reranker_model: str,
    seed: int,
    test_split: float,
    subset: int,
    train_reports: list[CTIReport],
    test_reports: list[CTIReport],
    full_test_split_reports: list[CTIReport] | None = None,
    dataset_mode: str,
    allow_synthetic_data: bool,
    project_root: Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest without serializing report bodies."""
    data_root = data_dir or Path("data")
    attack_paths = {
        "raw_stix_bundle": data_root / "mitre/enterprise-attack.json",
        "normalized_entries": data_root / "mitre/entries.jsonl",
        "bm25_corpus": data_root / "mitre/bm25_corpus.jsonl",
    }
    attack_entry_count = None
    entry_path = attack_paths["normalized_entries"]
    if entry_path.is_file():
        with entry_path.open(encoding="utf-8") as handle:
            attack_entry_count = sum(1 for line in handle if line.strip())
    prompt_paths = {
        "extractor_system": project_root / "src/agents/prompts/extractor_system.txt",
        "extractor_few_shot_template": project_root / "src/agents/prompts/extractor_few_shot.txt",
        "verifier_system": project_root / "src/agents/prompts/verifier_system.txt",
    }
    code_paths = {
        "evaluator": project_root / "src/evaluation/evaluator.py",
        "evaluation_runner": project_root / "src/evaluation/run_evaluation.py",
        "audit_logger": project_root / "src/evaluation/audit_log.py",
        "dataset_loader": project_root / "src/ingestion/dataset_loader.py",
        "mitre_loader": project_root / "src/ingestion/mitre_loader.py",
        "data_setup_script": project_root / "scripts/setup_data.py",
        "data_models": project_root / "src/ingestion/models.py",
        "pipeline_runner": project_root / "src/pipeline/runner.py",
        "pipeline_graph": project_root / "src/pipeline/graph.py",
        "verification_state": project_root / "src/pipeline/verification_state.py",
        "verification_response_parser": project_root / "src/pipeline/verification_response.py",
        "dense_vector_store": project_root / "src/rag/vector_store.py",
        "bm25_index": project_root / "src/rag/bm25_index.py",
        "hybrid_retriever": project_root / "src/rag/hybrid_retriever.py",
        "canonicalizer": project_root / "src/agents/canonicalizer.py",
        "verifier": project_root / "src/agents/verifier.py",
        "llm_client": project_root / "src/agents/llm_client.py",
        "llm_call_audit": project_root / "src/agents/llm_call_audit.py",
        "extractor": project_root / "src/agents/extractor.py",
        "dependencies": project_root / "pyproject.toml",
        "research_preflight": project_root / "scripts/validate_research_inputs.py",
        "research_benchmark_workflow": project_root / ".github/workflows/research-benchmark.yml",
    }
    full_test_reports = full_test_split_reports if full_test_split_reports is not None else test_reports
    all_dataset_reports = [*train_reports, *full_test_reports]
    synthetic_ids = sorted(
        report.id for report in all_dataset_reports
        if report.id.startswith("synthetic_")
    )
    return {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "git_commit": os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT_SHA"),
        "python_version": platform.python_version(),
        "condition": condition,
        "models": {
            "provider": os.environ.get("LLM_PROVIDER", "groq").lower(),
            "llm_configured_primary": model,
            "llm_configured_fallback": resolve_configured_fallback_model(),
            "embedding": embedding_model,
            "reranker": reranker_model if condition in {"full", "both"} else None,
        },
        "llm_call_audit_file": os.environ.get("LLM_CALL_AUDIT_PATH", "results/llm_call_audit.jsonl"),
        "configuration": {
            "seed": seed,
            "test_split_fraction": test_split,
            "subset_limit": subset,
            "few_shot_k": 5,
            "rag_top_k": 5 if condition in {"full", "both"} else None,
            "similarity_threshold": 0.85,
            "llm_temperature": 0.0,
            "extractor_max_tokens": 2048,
            "canonicalizer_max_tokens": 64,
            "verifier_max_tokens": 512,
            "max_report_chars": 2500,
            "max_triplets_per_report": 10,
        },
        "attack_knowledge_base": {
            "entry_count": attack_entry_count,
            "files_sha256": {name: sha256_file(path) for name, path in attack_paths.items()},
        },
        "dataset": {
            "mode": dataset_mode,
            "allow_synthetic_data": allow_synthetic_data,
            "report_count": len(all_dataset_reports),
            "train_count": len(train_reports),
            "full_test_split_count": len(full_test_reports),
            "evaluated_test_count": len(test_reports),
            "synthetic_report_ids": synthetic_ids,
            "train_report_ids": [report.id for report in train_reports],
            "full_test_split_report_ids": sorted(report.id for report in full_test_reports),
            "evaluated_test_report_ids": sorted(report.id for report in test_reports),
            "fingerprint_sha256": dataset_fingerprint(all_dataset_reports),
        },
        "prompt_sha256": {name: sha256_file(path) for name, path in prompt_paths.items()},
        "implementation_sha256": {name: sha256_file(path) for name, path in code_paths.items()},
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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
        "llm_provider": os.environ.get("LLM_PROVIDER", "groq").lower(),
        "configured_fallback_model": resolve_configured_fallback_model(),
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
