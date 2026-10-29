"""Tests for safe request-level LLM model attribution audit logging."""
import json

from types import SimpleNamespace

from src.agents.llm_call_audit import (
    log_model_call as _log_model_call,
    resolve_served_model,
)


def test_model_call_audit_records_actual_fallback_model_without_prompts_or_secrets(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "models.jsonl"
    monkeypatch.setenv("LLM_CALL_AUDIT_PATH", str(path))

    _log_model_call(
        requested_model="model-primary",
        actual_model="model-fallback",
        provider="groq",
        outcome="fallback_success",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["requested_model"] == "model-primary"
    assert rows[0]["actual_model"] == "model-fallback"
    assert rows[0]["outcome"] == "fallback_success"
    assert "messages" not in rows[0]
    assert "api_key" not in rows[0]


def test_model_call_audit_records_one_json_object_per_line(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("LLM_CALL_AUDIT_PATH", str(path))

    _log_model_call(
        requested_model="model-a",
        actual_model="model-a",
        provider="openai",
        outcome="success",
    )
    _log_model_call(
        requested_model="model-a",
        actual_model="model-b",
        provider="openai",
        outcome="fallback_success",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[1]["actual_model"] == "model-b"


def test_served_model_id_prefers_provider_response_when_available():
    response = SimpleNamespace(model="provider-resolved-model-v2")
    assert resolve_served_model(response, "requested-alias") == "provider-resolved-model-v2"


def test_served_model_id_falls_back_to_requested_id_when_provider_omits_it():
    response = SimpleNamespace(choices=[])
    assert resolve_served_model(response, "requested-model") == "requested-model"
