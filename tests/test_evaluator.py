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


def test_rejection_rate_is_not_applicable_when_verifier_did_not_run():
    predicted = {"report-a": [triplet("APT29", "uses", "ToolX")]}
    gold = {"report-a": [triplet("APT29", "uses", "ToolX")]}

    result = evaluate_pipeline_run("baseline", predicted, gold)

    assert result.rejection_rate is None
    assert result.hallucination_rate is None


def test_rejection_rate_is_zero_when_verifier_ran_and_rejected_nothing():
    predicted = {"report-a": [triplet("APT29", "uses", "ToolX")]}
    gold = {"report-a": [triplet("APT29", "uses", "ToolX")]}

    result = evaluate_pipeline_run(
        "full_pipeline",
        predicted,
        gold,
        rejected_by_report={"report-a": []},
        extracted_counts_by_report={"report-a": 1},
    )

    assert result.rejection_rate == 0.0
    assert result.hallucination_rate == 0.0
def test_predictions_with_empty_gold_yields_zero_metrics():
    predicted = [triplet("APT29", "uses", "ToolX")]
    gold = []
    p, r, f = compute_triplet_f1(predicted, gold)
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_gold_with_empty_predictions_yields_zero_metrics():
    predicted = []
    gold = [triplet("APT29", "uses", "ToolX")]
    p, r, f = compute_triplet_f1(predicted, gold)
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_macro_versus_micro_metric_weighting_discrepancy():
    # Report A: 1 prediction, 1 gold, matches -> P=1.0, R=1.0, F1=1.0
    # Report B: 9 predictions, 9 gold, 0 match -> P=0.0, R=0.0, F1=0.0
    predicted = {
        "report-a": [triplet("APT29", "uses", "ToolA")],
        "report-b": [triplet("APT29", "uses", f"WrongTool_{i}") for i in range(9)],
    }
    gold = {
        "report-a": [triplet("APT29", "uses", "ToolA")],
        "report-b": [triplet("APT29", "uses", f"GoldTool_{i}") for i in range(9)],
    }
    result = evaluate_pipeline_run("test", predicted, gold)
    # Micro: 1 TP out of 10 predictions -> 0.1
    assert result.precision == 0.1
    assert result.recall == 0.1
    assert result.f1 == 0.1
    # Macro: (1.0 + 0.0) / 2 = 0.5
    assert result.macro_precision == 0.5
    assert result.macro_recall == 0.5
    assert result.macro_f1 == 0.5


def test_pipeline_result_rejection_rate_formula():
    from src.ingestion.models import PipelineResult

    v1 = triplet("APT29", "uses", "ToolA")
    v2 = triplet("APT29", "uses", "ToolB")
    rej = triplet("APT29", "uses", "BadTool")

    result = PipelineResult(
        report_id="test-rep",
        final_triplets=[v1],  # 1 final triplet after deduplication
        verified_triplets=[v1, v2],  # 2 verified triplets
        rejected_triplets=[rej],  # 1 rejected triplet
        revised_triplets=[],
        verification_decisions=[],
        iterations=1,
    )
    # Correct formula: len(rejected_triplets) / (len(verified_triplets) + len(rejected_triplets)) = 1 / (2 + 1) = 1/3
    assert abs(result.rejection_rate - (1.0 / 3.0)) < 1e-6
    assert abs(result.hallucination_rate - (1.0 / 3.0)) < 1e-6


def test_pipeline_result_rejection_rate_handles_zero_triplets():
    from src.ingestion.models import PipelineResult

    result = PipelineResult(
        report_id="test-rep",
        final_triplets=[],
        verified_triplets=[],
        rejected_triplets=[],
        revised_triplets=[],
        verification_decisions=[],
        iterations=1,
    )
    assert result.rejection_rate == 0.0
    assert result.hallucination_rate == 0.0


def test_resolve_evaluation_model_defaults_to_primary_model_without_override(monkeypatch):
    from src.evaluation.run_evaluation import resolve_evaluation_model
    from src.agents.llm_client import PRIMARY_MODEL

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    resolved = resolve_evaluation_model(None)
    assert resolved == PRIMARY_MODEL
    assert "llama-3.3-70b-versatile" not in resolved


def test_resolve_evaluation_model_respects_explicit_override_and_env(monkeypatch):
    from src.evaluation.run_evaluation import resolve_evaluation_model

    monkeypatch.setenv("LLM_MODEL", "env-llm-model")
    assert resolve_evaluation_model("cli-model-override") == "cli-model-override"

    monkeypatch.setenv("PRIMARY_MODEL", "env-primary-model")
    assert resolve_evaluation_model(None) == "env-llm-model"

    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert resolve_evaluation_model(None) == "env-primary-model"


def test_resolve_configured_fallback_model_defaults_without_override(monkeypatch):
    from src.evaluation.audit_log import resolve_configured_fallback_model
    from src.agents.llm_client import FALLBACK_MODEL

    monkeypatch.delenv("FALLBACK_MODEL", raising=False)
    resolved = resolve_configured_fallback_model()
    assert resolved == FALLBACK_MODEL
    assert "llama-3.1-8b-instant" not in resolved

    monkeypatch.setenv("FALLBACK_MODEL", "custom-fallback")
    assert resolve_configured_fallback_model() == "custom-fallback"
