import json

from src.evaluation.audit_log import build_audit_record, write_jsonl_record
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
