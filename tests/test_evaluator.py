from src.evaluation.evaluator import (
    compute_entity_f1,
    compute_hallucination_rate,
    compute_rejection_rate,
    compute_triplet_f1,
    evaluate_pipeline_run,
)
from src.ingestion.models import EntityType, Triplet


def triplet(subject: str, relation: str, obj: str) -> Triplet:
    return Triplet(
        entity1=subject,
        entity1_type=EntityType.THREAT_ACTOR,
        relation=relation,
        entity2=obj,
        entity2_type=EntityType.TOOL,
    )


def test_triplet_exact_match_is_case_and_punctuation_insensitive():
    p = [triplet("APT-29", "USES", "Cobalt Strike")]
    g = [triplet("apt 29", "uses", "Cobalt Strike")]
    assert compute_triplet_f1(p, g) == (1.0, 1.0, 1.0)


def test_report_local_evaluation_prevents_cross_report_false_match():
    # The same relation appears in the gold set, but only for the other report.
    predicted = {"report-a": [triplet("APT29", "uses", "ToolX")], "report-b": []}
    gold = {"report-a": [], "report-b": [triplet("APT29", "uses", "ToolX")]}
    result = evaluate_pipeline_run("test", predicted, gold)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_duplicate_predictions_within_report_do_not_inflate_exact_match_scores():
    p = [triplet("APT29", "uses", "ToolX"), triplet("APT29", "uses", "ToolX")]
    g = [triplet("APT29", "uses", "ToolX")]
    assert compute_triplet_f1(p, g) == (1.0, 1.0, 1.0)


def test_empty_predictions_and_empty_gold_are_perfect_match():
    assert compute_triplet_f1([], []) == (1.0, 1.0, 1.0)
    assert compute_entity_f1([], []) == (1.0, 1.0, 1.0)


def test_rejection_rate_is_not_mislabeled_by_helper_semantics():
    rejected = [triplet("APT29", "uses", "BadTool")]
    assert compute_rejection_rate(rejected, 4) == 0.25
    # Kept temporarily for backward compatibility with existing callers.
    assert compute_hallucination_rate(rejected, 4) == 0.25


def test_rejection_rate_handles_zero_extractions():
    assert compute_rejection_rate([], 0) == 0.0
