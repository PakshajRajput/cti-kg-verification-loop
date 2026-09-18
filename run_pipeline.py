import os
import sys

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console

from src.ingestion.dataset_loader import load_ctinexus_dataset
from src.pipeline.runner import PipelineConfig, PipelineRunner
from src.rag.hybrid_retriever import HybridRetriever
from src.evaluation.evaluator import evaluate_pipeline_run
from src.kg.networkx_writer import NetworkXWriter

console = Console(force_terminal=True, legacy_windows=False)

def validate_environment(data_path: Path):
    console.print("[cyan]Validating Environment...[/cyan]")
    errors = False
    
    # Check API Key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "gsk_your_key_here":
        console.print("[red]✗ GROQ_API_KEY is missing in .env[/red]")
        errors = True
    else:
        console.print("[green]✓ GROQ_API_KEY found[/green]")
        
    # Check data directory
    if not data_path.exists():
        console.print(f"[red]✗ Data directory missing: {data_path}[/red]")
        errors = True
    else:
        console.print(f"[green]✓ Data directory exists[/green]")
        
    # Check Chroma
    chroma_path = data_path / "chroma"
    if not chroma_path.exists():
        console.print(f"[red]✗ Chroma index missing: {chroma_path}[/red]")
        errors = True
    else:
        console.print("[green]✓ Chroma index exists[/green]")
        
    # Check BM25
    bm25_path = data_path / "mitre" / "bm25_index.pkl"
    if not bm25_path.exists():
        console.print(f"[red]✗ BM25 index missing: {bm25_path}[/red]")
        errors = True
    else:
        console.print("[green]✓ BM25 index exists[/green]")
        
    if errors:
        console.print("\n[bold red]Environment validation failed. Please fix the above errors and rerun.[/bold red]\n")
        sys.exit(1)


def main():
    console.rule("[bold cyan]CTI KG Pipeline Execution")

    data_path = Path("data")
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)

    validate_environment(data_path)

    # 1. Setup Models
    llm_model = os.environ.get("PRIMARY_MODEL", "llama-3.3-70b-versatile")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 2. Load Dataset (Take 3 test reports)
    console.print("[cyan]Loading Dataset (3 Reports)...[/cyan]")
    train_reports, test_reports = load_ctinexus_dataset(data_path)
    
    # Filter to only reports that have ground truth triplets
    test_reports = [r for r in test_reports if len(r.ground_truth_triplets) > 0]
    if len(test_reports) == 0:
        console.print("[red]✗ No test reports with ground truth found! Exiting.[/red]")
        sys.exit(1)
        
    test_reports = test_reports[:3]

    # 3. Initialize Pipeline
    console.print("[cyan]Initializing RAG Retriever & Pipeline...[/cyan]")
    retriever = HybridRetriever.from_disk(data_path, embedding_model=embedding_model, reranker_model=reranker_model)
    config = PipelineConfig(
        model=llm_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        skip_verifier=False,
        data_dir=data_path,
    )
    runner = PipelineRunner.build(config, train_reports, retriever=retriever)

    # 4. Run Pipeline with Timeline Tracking
    all_final = []
    all_rejected = []
    predicted_by_report = {}
    ground_truth_by_report = {}
    rejected_by_report = {}
    extracted_counts_by_report = {}
    
    kg = NetworkXWriter(str(results_path / "kg.json"))
    kg.connect()

    markdown_report = "# CTI Pipeline Summary Report\n\n## Timeline\n"

    for i, report in enumerate(test_reports):
        console.print(f"\n[cyan]Processing Report {i+1}/3: {report.id}[/cyan]")
        markdown_report += f"\n### Report {i+1}: {report.id}\n"
        
        t0 = time.time()
        
        # In a real LangGraph setup we'd extract state timestamps, 
        # but we can simulate the timeline steps based on runner internal behavior here,
        # or measure the total run. We'll measure the full run for the summary.
        result = runner.run(report)
        t_total = time.time() - t0
        
        console.print(f"  [green]✓ {len(result.final_triplets)} verified, {len(result.rejected_triplets)} rejected ({t_total:.1f}s)[/green]")
        markdown_report += f"- **Total Time**: {t_total:.1f} s\n"
        markdown_report += f"- **Verified Triplets**: {len(result.final_triplets)}\n"
        markdown_report += f"- **Rejected (Hallucinations)**: {len(result.rejected_triplets)}\n"
        
        predicted_by_report[report.id] = result.final_triplets
        ground_truth_by_report[report.id] = report.ground_truth_triplets
        rejected_by_report[report.id] = result.rejected_triplets
        extracted_counts_by_report[report.id] = len(result.final_triplets) + len(result.rejected_triplets)
        
        kg.write_triplets(result.final_triplets)
        
        for t in result.final_triplets:
            all_final.append(t.model_dump())
            
        for t in result.rejected_triplets:
            all_rejected.append(t.model_dump())

    # Save the knowledge graph to JSON
    kg.save()

    # 5. Export JSONs
    with open(results_path / "extracted_triplets.json", "w") as f:
        json.dump(all_final + all_rejected, f, indent=2)
    with open(results_path / "verified_triplets.json", "w") as f:
        json.dump(all_final, f, indent=2)

    # 6. Evaluation Metrics
    console.print("\n[cyan]Evaluating Results...[/cyan]")
    metrics = evaluate_pipeline_run(
        "full_pipeline", 
        predicted_by_report, 
        ground_truth_by_report,
        rejected_by_report=rejected_by_report,
        extracted_counts_by_report=extracted_counts_by_report
    )
    
    with open(results_path / "evaluation_results.json", "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    markdown_report += f"\n## Evaluation Metrics\n"
    markdown_report += f"- **Precision**: {metrics.precision:.4f}\n"
    markdown_report += f"- **Recall**: {metrics.recall:.4f}\n"
    markdown_report += f"- **F1 Score**: {metrics.f1:.4f}\n"
    markdown_report += f"- **Hallucination Rate**: {metrics.hallucination_rate:.4f}\n"

    # 7. Visualization
    console.print("[cyan]Generating Graph Visualization...[/cyan]")
    kg_data = kg.get_stats() # Just to verify
    G = nx.node_link_graph(json.loads(Path(results_path / "kg.json").read_text()), edges="links")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    if len(G.nodes) < 30:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)
    
    node_colors = []
    node_sizes = []
    for node, d in G.nodes(data=True):
        lbl = d.get("label", "").lower()
        if lbl in ["threatactor", "threat_actor"]: node_colors.append("#E53935")
        elif lbl == "malware": node_colors.append("#8E24AA")
        elif lbl == "tool": node_colors.append("#1E88E5")
        elif lbl == "vulnerability": node_colors.append("#FB8C00")
        elif lbl in ["technique", "attack_pattern", "attackpattern"]: node_colors.append("#43A047")
        elif lbl == "target": node_colors.append("#00897B")
        else: node_colors.append("#607D8B")
        degree = G.degree(node) if G.degree(node) is not None else 1
        node_sizes.append(500 + degree * 250)
        
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, edge_color="gray", arrows=True, connectionstyle="arc3,rad=0.1", min_source_margin=15, min_target_margin=15)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(results_path / "kg_visualization.png", dpi=300)
    
    # Update markdown with graph stats
    markdown_report += f"\n## Knowledge Graph Stats\n"
    markdown_report += f"- **Total Nodes**: {G.number_of_nodes()}\n"
    markdown_report += f"- **Total Edges**: {G.number_of_edges()}\n"
    markdown_report += "\n### Entity Type Breakdown\n"
    
    entity_counts = {}
    for _, d in G.nodes(data=True):
        lbl = d.get("label", "Unknown")
        entity_counts[lbl] = entity_counts.get(lbl, 0) + 1
        
    for lbl, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        markdown_report += f"- **{lbl}**: {count}\n"
        
    markdown_report += f"\n## Graph Visualization\n"
    markdown_report += f"![Knowledge Graph](kg_visualization.png)\n"
    
    # 8. Markdown Summary
    with open(results_path / "summary_report.md", "w") as f:
        f.write(markdown_report)

    console.rule("[bold green]Pipeline Execution Complete!")
    console.print(f"Artifacts saved to [bold]{results_path.resolve()}[/bold]")

if __name__ == "__main__":
    main()
