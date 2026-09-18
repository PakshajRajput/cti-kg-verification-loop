import os
import sys

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
from pathlib import Path
from src.ingestion.dataset_loader import load_ctinexus_dataset
from src.ingestion.models import Triplet
from src.evaluation.evaluator import evaluate_pipeline_run
from rich.console import Console

console = Console(force_terminal=True, legacy_windows=False)

def main():
    data_path = Path("data")
    results_path = Path("results")
    
    console.print("[cyan]Loading Dataset (3 Reports)...[/cyan]")
    train_reports, test_reports = load_ctinexus_dataset(data_path)
    test_reports = test_reports[:3]
    
    # Load predictions
    with open(results_path / "verified_triplets.json", "r", encoding="utf-8") as f:
        verified_data = json.load(f)
    with open(results_path / "extracted_triplets.json", "r", encoding="utf-8") as f:
        extracted_data = json.load(f)
        
    all_predicted = [Triplet(**t) for t in verified_data]
    
    # Reconstruct dictionary mapping
    # We evaluate everything globally because the flat JSON lists lost individual report mappings.
    predicted_by_report = {"global_aggregate": all_predicted}
    
    all_gt = []
    for r in test_reports:
        all_gt.extend(r.ground_truth_triplets)
    ground_truth_by_report = {"global_aggregate": all_gt}
    
    extracted_counts_by_report = {"global_aggregate": len(extracted_data)}
    
    # Mock rejected list just for hallucination rate calculation
    mock_rejected = [Triplet(entity1="dummy", relation="dummy", entity2="dummy")] * (len(extracted_data) - len(verified_data))
    rejected_by_report = {"global_aggregate": mock_rejected}
    
    console.print("\n[cyan]Evaluating Results (Canonicalized)...[/cyan]")
    metrics = evaluate_pipeline_run(
        "full_pipeline", 
        predicted_by_report, 
        ground_truth_by_report,
        rejected_by_report=rejected_by_report,
        extracted_counts_by_report=extracted_counts_by_report
    )
    
    with open(results_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)
        
    console.print(f"\n[bold green]Re-Evaluation Complete![/bold green]")
    console.print(f"Precision: {metrics.precision:.4f}")
    console.print(f"Recall: {metrics.recall:.4f}")
    console.print(f"F1 Score: {metrics.f1:.4f}")
    console.print(f"Hallucination Rate: {metrics.hallucination_rate:.4f}")

if __name__ == "__main__":
    main()
