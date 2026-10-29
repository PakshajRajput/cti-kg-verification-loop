"""Evaluation metrics for CTI triplet extraction.

Triplet and entity metrics are computed per report to prevent cross-report
matches. Micro metrics pool report-local counts; macro metrics average
per-report metrics. The verifier rejection fraction is named rejection_rate,
not hallucination_rate, because rejection is not itself ground-truth proof
that a triplet is hallucinated.
"""
from __future__ import annotations

import re
from typing import Optional

from src.ingestion.models import EvalMetrics, Triplet


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _triplet_key(t: Triplet) -> tuple[str, str, str]:
    return (_normalize(t.entity1), _normalize(t.relation), _normalize(t.entity2))


def _entity_keys(triplets: list[Triplet]) -> set[str]:
    return {_normalize(t.entity1) for t in triplets} | {
        _normalize(t.entity2) for t in triplets
    }


def _prf(tp: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = tp / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = tp / gold if gold else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_triplet_f1(
    predicted: list[Triplet], ground_truth: list[Triplet]
) -> tuple[float, float, float]:
    pred_keys = {_triplet_key(t) for t in predicted}
    gold_keys = {_triplet_key(t) for t in ground_truth}
    return _prf(len(pred_keys & gold_keys), len(pred_keys), len(gold_keys))


def compute_entity_f1(
    predicted: list[Triplet], ground_truth: list[Triplet]
) -> tuple[float, float, float]:
    pred_entities = _entity_keys(predicted)
    gold_entities = _entity_keys(ground_truth)
    return _prf(
        len(pred_entities & gold_entities), len(pred_entities), len(gold_entities)
    )


def compute_rejection_rate(rejected_triplets: list[Triplet] | int, total_extracted: int) -> float:
    """Share of extracted predictions rejected by the verifier (not a ground-truth hallucination rate).

    Accept an integer count internally as well as a list of rejected triplets for
    backwards compatibility with existing callers.
    """
    if total_extracted <= 0:
        return 0.0
    rejected_count = rejected_triplets if isinstance(rejected_triplets, int) else len(rejected_triplets)
    return rejected_count / total_extracted


def compute_hallucination_rate(
    rejected_triplets: list[Triplet], total_extracted: int
) -> float:
    """Backward-compatible alias; deprecated because rejection != confirmed hallucination."""
    return compute_rejection_rate(rejected_triplets, total_extracted)


def evaluate_pipeline_run(
    condition: str,
    predicted_by_report: dict[str, list[Triplet]],
    ground_truth_by_report: dict[str, list[Triplet]],
    rejected_by_report: Optional[dict[str, list[Triplet]]] = None,
    extracted_counts_by_report: Optional[dict[str, int]] = None,
) -> EvalMetrics:
    """Evaluate each report independently, then compute micro and macro metrics."""
    report_ids = sorted(set(predicted_by_report) | set(ground_truth_by_report))
    report_results: list[dict[str, float | int]] = []
    micro_tp = micro_pred = micro_gold = 0
    entity_tp = entity_pred = entity_gold = 0
    all_rejected = 0
    total_extracted = 0
    total_verified = 0
    total_ground_truth = 0

    for rid in report_ids:
        pred = predicted_by_report.get(rid, [])
        gold = ground_truth_by_report.get(rid, [])
        pred_keys, gold_keys = {_triplet_key(t) for t in pred}, {_triplet_key(t) for t in gold}
        tp = len(pred_keys & gold_keys)
        p, r, f = _prf(tp, len(pred_keys), len(gold_keys))
        pred_entities, gold_entities = _entity_keys(pred), _entity_keys(gold)
        etp = len(pred_entities & gold_entities)
        ep, er, ef = _prf(etp, len(pred_entities), len(gold_entities))

        report_results.append({
            "precision": p, "recall": r, "f1": f,
            "entity_precision": ep, "entity_recall": er, "entity_f1": ef,
        })
        micro_tp += tp
        micro_pred += len(pred_keys)
        micro_gold += len(gold_keys)
        entity_tp += etp
        entity_pred += len(pred_entities)
        entity_gold += len(gold_entities)
        total_verified += len(pred)
        total_ground_truth += len(gold)
        total_extracted += (
            extracted_counts_by_report.get(rid, len(pred) + len((rejected_by_report or {}).get(rid, [])))
            if extracted_counts_by_report is not None
            else len(pred) + len((rejected_by_report or {}).get(rid, []))
        )
        all_rejected += len((rejected_by_report or {}).get(rid, []))

    micro_p, micro_r, micro_f = _prf(micro_tp, micro_pred, micro_gold)
    emicro_p, emicro_r, emicro_f = _prf(entity_tp, entity_pred, entity_gold)

    def macro(metric: str) -> float:
        return sum(float(row[metric]) for row in report_results) / len(report_results) if report_results else 0.0

    # Macro metrics average report-level scores, complementing pooled micro scores.
    # Baseline has no verifier, so rejection rate is not applicable rather than zero.
    rejection_rate = (
        compute_rejection_rate(all_rejected, total_extracted)
        if rejected_by_report is not None
        else None
    )
    return EvalMetrics(
        condition=condition,
        precision=round(micro_p, 4),
        recall=round(micro_r, 4),
        f1=round(micro_f, 4),
        entity_precision=round(emicro_p, 4),
        entity_recall=round(emicro_r, 4),
        entity_f1=round(emicro_f, 4),
        # Backward-compatible field; use rejection_rate for new analyses.
        hallucination_rate=round(rejection_rate, 4) if rejection_rate is not None else None,
        rejection_rate=round(rejection_rate, 4) if rejection_rate is not None else None,
        macro_precision=round(macro("precision"), 4),
        macro_recall=round(macro("recall"), 4),
        macro_f1=round(macro("f1"), 4),
        macro_entity_precision=round(macro("entity_precision"), 4),
        macro_entity_recall=round(macro("entity_recall"), 4),
        macro_entity_f1=round(macro("entity_f1"), 4),
        total_extracted=total_extracted,
        total_verified=total_verified,
        total_rejected=all_rejected,
        total_ground_truth=total_ground_truth,
        num_reports=len(report_ids),
    )
