"""Preflight tests ensure benchmark workflows reject synthetic/fallback inputs."""
import json

import pytest

from scripts.validate_research_inputs import validate_research_inputs


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_valid_inputs(root, *, synthetic=False, attack_count=100, annotated=10):
    reports = []
    for i in range(10):
        reports.append({
            "id": f"synthetic_{i}" if synthetic and i == 0 else f"report_{i}",
            "ground_truth_triplets": [{"entity1": "A", "relation": "uses", "entity2": "B"}]
            if i < annotated else [],
        })
    _write_jsonl(root / "processed/reports.jsonl", reports)

    entries = [
        {"id": f"T{i}", "name": f"Technique {i}", "description": "description"}
        for i in range(attack_count)
    ]
    _write_jsonl(root / "mitre/entries.jsonl", entries)
    _write_jsonl(root / "mitre/bm25_corpus.jsonl", entries)
    (root / "mitre/bm25_index.pkl").parent.mkdir(parents=True, exist_ok=True)
    (root / "mitre/bm25_index.pkl").write_bytes(b"test-index")
    (root / "chroma/chroma.sqlite3").parent.mkdir(parents=True, exist_ok=True)
    (root / "chroma/chroma.sqlite3").write_bytes(b"test-vector-store")


def test_research_preflight_accepts_real_annotated_inputs(tmp_path):
    _make_valid_inputs(tmp_path)

    result = validate_research_inputs(tmp_path, check_chroma_collection=False)

    assert result["real_reports"] == 10
    assert result["annotated_reports"] == 10
    assert result["attack_entries"] == 100


def test_research_preflight_rejects_synthetic_reports(tmp_path):
    _make_valid_inputs(tmp_path, synthetic=True)

    with pytest.raises(RuntimeError, match="synthetic"):
        validate_research_inputs(tmp_path, check_chroma_collection=False)


def test_research_preflight_rejects_fallback_attack_entries(tmp_path):
    _make_valid_inputs(tmp_path, attack_count=10)

    with pytest.raises(RuntimeError, match="fallback/demo"):
        validate_research_inputs(tmp_path, check_chroma_collection=False)


def test_research_preflight_requires_enough_annotated_reports(tmp_path):
    _make_valid_inputs(tmp_path, annotated=9)

    with pytest.raises(RuntimeError, match="ground-truth triplets"):
        validate_research_inputs(tmp_path, check_chroma_collection=False)
