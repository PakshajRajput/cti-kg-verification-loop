"""
Evaluation metrics for CTI triplet extraction.

Implements Precision, Recall, F1 at:
  - Triplet level (exact match: entity1 + relation + entity2)
  - Entity level (any entity match regardless of relation)
  - Hallucination rate (fraction of extracted triplets rejected by verifier)

Matching strategy:
  - Exact: all three fields match (case-insensitive)
  - Partial: entity1 AND entity2 match (relation is ignored)
  
The hallucination_rate metric is the key novel contribution metric:
  hallucination_rate = rejected_triplets / total_extracted_triplets
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.ingestion.models import EvalMetrics, Triplet


import re
from rich.console import Console

console = Console()

def _normalize(s: str) -> str:
    """Case-fold, remove punctuation, and collapse spaces for comparison."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _triplet_key(t: Triplet) -> tuple[str, str, str]:
    return (_normalize(t.entity1), _normalize(t.relation), _normalize(t.entity2))


def _entity_pair_key(t: Triplet) -> tuple[str, str]:
    """Entity-pair key (ignores relation direction)."""
    e1, e2 = _normalize(t.entity1), _normalize(t.entity2)
    return (min(e1, e2), max(e1, e2))


def compute_triplet_f1(
    predicted: list[Triplet],
    ground_truth: list[Triplet],
) -> tuple[float, float, float]:
    """
    Exact triplet-level Precision, Recall, F1.

    A predicted triplet is a true positive if there exists a matching
    ground-truth triplet with the same entity1, relation, and entity2
    (case-insensitive).

    Returns: (precision, recall, f1)
    """
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not ground_truth:
        return 0.0, 0.0, 0.0

    gt_keys = {_triplet_key(t) for t in ground_truth}
    pred_keys = {_triplet_key(t) for t in predicted}

    true_positives = len(pred_keys & gt_keys)
    precision = true_positives / len(pred_keys) if pred_keys else 0.0
    recall = true_positives / len(gt_keys) if gt_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_entity_f1(
    predicted: list[Triplet],
    ground_truth: list[Triplet],
) -> tuple[float, float, float]:
    """
    Entity-level Precision, Recall, F1.

    A predicted entity is a true positive if it appears in ground truth
    (as entity1 or entity2 in any triplet).
    """
    pred_entities = {_normalize(t.entity1) for t in predicted} | {_normalize(t.entity2) for t in predicted}
    gt_entities = {_normalize(t.entity1) for t in ground_truth} | {_normalize(t.entity2) for t in ground_truth}

    if not pred_entities and not gt_entities:
        return 1.0, 1.0, 1.0
    if not pred_entities:
        return 0.0, 0.0, 0.0
    if not gt_entities:
        return 0.0, 0.0, 0.0

    tp = len(pred_entities & gt_entities)
    precision = tp / len(pred_entities)
    recall = tp / len(gt_entities)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_hallucination_rate(
    rejected_triplets: list[Triplet],
    total_extracted: int,
) -> float:
    """
    Fraction of extracted triplets that were rejected by the verifier.

    hallucination_rate = |rejected| / |total_extracted|

    This is 0.0 for the baseline (no verifier) by definition.
    For the full pipeline, lower is better — but recall that the verifier
    may also reject correct triplets (false negatives), so this metric
    must be interpreted alongside F1.
    """
    if total_extracted == 0:
        return 0.0
    return len(rejected_triplets) / total_extracted


def evaluate_pipeline_run(
    condition: str,
    predicted_by_report: dict[str, list[Triplet]],
    ground_truth_by_report: dict[str, list[Triplet]],
    rejected_by_report: Optional[dict[str, list[Triplet]]] = None,
    extracted_counts_by_report: Optional[dict[str, int]] = None,
) -> EvalMetrics:
    """
    Compute aggregate evaluation metrics across all reports.

    Args:
        condition: "baseline" or "full_pipeline"
        predicted_by_report: Report ID → list of final predicted triplets
        ground_truth_by_report: Report ID → list of ground-truth triplets
        rejected_by_report: Report ID → list of rejected triplets (for hallucination rate)
        extracted_counts_by_report: Report ID → total extracted before verification

    Returns: EvalMetrics with aggregated P/R/F1
    """
    all_predicted: list[Triplet] = []
    all_gt: list[Triplet] = []
    all_rejected: list[Triplet] = []
    total_extracted = 0

    report_ids = set(predicted_by_report.keys()) | set(ground_truth_by_report.keys())

    for rid in report_ids:
        pred = predicted_by_report.get(rid, [])
        gt = ground_truth_by_report.get(rid, [])
        all_predicted.extend(pred)
        all_gt.extend(gt)
        if rejected_by_report:
            all_rejected.extend(rejected_by_report.get(rid, []))
        if extracted_counts_by_report:
            total_extracted += extracted_counts_by_report.get(rid, len(pred))
        else:
            total_extracted += len(pred)
            
        # --- Debugging Output ---
        gt_keys = {_triplet_key(t): t for t in gt}
        pred_keys = {_triplet_key(t): t for t in pred}
        
        matched_keys = set(pred_keys.keys()) & set(gt_keys.keys())
        unmatched_pred_keys = set(pred_keys.keys()) - set(gt_keys.keys())
        missed_gt_keys = set(gt_keys.keys()) - set(pred_keys.keys())
        
        if len(gt) == 0:
            console.print(f"[bold yellow]Warning: Report {rid} has 0 gold triplets! It will negatively impact precision.[/bold yellow]")

        console.print(f"\nReport: {rid}")
        console.print(f"Gold: {len(gt)}")
        console.print(f"Predicted: {len(pred)}")
        console.print(f"Matched: {len(matched_keys)}")

    precision, recall, f1 = compute_triplet_f1(all_predicted, all_gt)
    e_prec, e_rec, e_f1 = compute_entity_f1(all_predicted, all_gt)
    hall_rate = compute_hallucination_rate(all_rejected, total_extracted) if rejected_by_report else 0.0

    return EvalMetrics(
        condition=condition,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        entity_precision=round(e_prec, 4),
        entity_recall=round(e_rec, 4),
        entity_f1=round(e_f1, 4),
        hallucination_rate=round(hall_rate, 4),
        total_extracted=total_extracted,
        total_verified=len(all_predicted),
        total_rejected=len(all_rejected),
        total_ground_truth=len(all_gt),
        num_reports=len(report_ids),
    )
