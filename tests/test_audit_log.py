import json

from src.evaluation.audit_log import (
    build_audit_record,
    build_experiment_manifest,
    dataset_fingerprint,
    write_jsonl_record,
)
from src.ingestion.models import (
    CTIReport,
    EntityType,
    PipelineResult,
    Triplet,
    VerificationDecision,
    VerificationVerdict,
)


def make_triplet(subject="APT29", relation="uses", obj="Cobalt Strike"):
    return Triplet(
        entity1=subject,
        entity1_type=EntityType.THREAT_ACTOR,
        relation=relation,
        entity2=obj,
        entity2_type=EntityType.TOOL,
        source_text="supporting excerpt",
    )


def test_audit_record_serializes_all_pipeline_stages_without_report_text():
    raw = make_triplet()
    canonical = make_triplet("APT29", "uses", "Cobalt Strike")
    corrected = make_triplet("APT29", "deploys", "Cobalt Strike")
    decision = VerificationDecision(
        original_triplet=raw,
        verdict=VerificationVerdict.CORRECTED,
        corrected_triplet=corrected,
        reasoning="Relation corrected",
        retrieved_context_ids=["T1059"],
    )
    report = CTIReport(
        id="report-1",
        title="Sensitive report title",
        text="Full confidential report body should not be copied to the audit record.",
        ground_truth_triplets=[raw],
    )
    result = PipelineResult(
        report_id=report.id,
        raw_triplets=[raw],
        raw_triplets_by_iteration=[[raw]],
        total_raw_triplets=1,
        canonical_triplets_by_iteration=[[canonical]],
        canonical_triplets=[canonical],
        final_triplets=[corrected],
        verified_triplets=[corrected],
        revised_triplets=[corrected],
        rejected_triplets=[],
        verification_decisions=[decision],
        iterations=1,
    )

    record = build_audit_record(
        report=report,
        result=result,
        condition="full_pipeline",
        model="test-model",
        dataset_mode="real_data",
    )

    assert record["raw_triplets"][0]["entity1_type"] == "ThreatActor"
    assert record["canonical_triplets_by_iteration"][0][0]["entity2"] == "Cobalt Strike"
    assert record["verification_decisions"][0]["verdict"] == "corrected"
    assert record["ground_truth_triplets"][0]["entity1"] == "APT29"
    assert "text" not in record
    assert "Sensitive report title" not in json.dumps(record)


def test_write_jsonl_record_appends_independent_parseable_records(tmp_path):
    path = tmp_path / "nested" / "audit.jsonl"
    write_jsonl_record(path, {"report_id": "a", "error": None})
    write_jsonl_record(path, {"report_id": "b", "error": "sample"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["report_id"] for row in rows] == ["a", "b"]


def test_dataset_fingerprint_is_stable_and_changes_when_report_changes():
    base = CTIReport(
        id="r1",
        source="dataset",
        text="one report",
        ground_truth_triplets=[make_triplet()],
    )
    reordered = CTIReport(
        id="r1",
        source="dataset",
        text="one report",
        ground_truth_triplets=[make_triplet()],
    )
    changed = CTIReport(
        id="r1",
        source="dataset",
        text="different report",
        ground_truth_triplets=[make_triplet()],
    )

    assert dataset_fingerprint([base]) == dataset_fingerprint([reordered])
    assert dataset_fingerprint([base]) != dataset_fingerprint([changed])


def test_experiment_manifest_records_reproducibility_without_report_text(tmp_path):
    train = [CTIReport(id="train-1", text="training text", ground_truth_triplets=[make_triplet()])]
    test = [CTIReport(id="test-1", text="held out text", ground_truth_triplets=[make_triplet()])]
    manifest = build_experiment_manifest(
        started_at_utc="2026-10-29T00:00:00+00:00",
        condition="both",
        model="test-llm",
        embedding_model="test-embedding",
        reranker_model="test-reranker",
        seed=42,
        test_split=0.2,
        subset=1,
        train_reports=train,
        test_reports=test,
        dataset_mode="real_data",
        allow_synthetic_data=False,
        project_root=tmp_path,
    )

    assert manifest["configuration"]["seed"] == 42
    assert manifest["dataset"]["train_report_ids"] == ["train-1"]
    assert manifest["dataset"]["test_report_ids"] == ["test-1"]
    assert len(manifest["dataset"]["fingerprint_sha256"]) == 64
    assert "training text" not in json.dumps(manifest)
    assert manifest["prompt_sha256"]["extractor_system"] is None
