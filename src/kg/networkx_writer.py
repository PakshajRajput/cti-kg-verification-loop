"""
NetworkX Graph storage for the CTI Pipeline.
Replaces Neo4j for a lightweight, self-contained graph representation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from rich.console import Console

from src.ingestion.models import Triplet

console = Console()


class NetworkXWriter:
    """
    Manages knowledge graph storage using NetworkX.
    Nodes are entities, edges are relations.
    Provides methods to save and load to/from JSON (node-link format).
    """

    def __init__(self, graph_path: str | Path = "data/kg.json"):
        self.graph_path = Path(graph_path)
        self.graph = nx.DiGraph()

    def connect(self):
        """Simulates connection / loads existing graph if available."""
        if self.graph_path.exists():
            self.load()
            console.print(f"[green]Loaded existing graph from {self.graph_path}[/green]")
        else:
            console.print("[yellow]Starting with an empty graph.[/yellow]")

    def close(self):
        """Saves the graph."""
        self.save()
        console.print(f"[green]Saved graph to {self.graph_path}[/green]")

    def clear(self):
        """Clear the graph database."""
        self.graph.clear()
        
    def write_triplets(self, triplets: list[Triplet]) -> None:
        """Add triplets to the NetworkX graph."""
        for t in triplets:
            e1 = t.entity1.lower()
            e2 = t.entity2.lower()

            # Add nodes with types
            if not self.graph.has_node(e1):
                self.graph.add_node(e1, label=t.entity1_type.value, name=t.entity1)
            
            if not self.graph.has_node(e2):
                self.graph.add_node(e2, label=t.entity2_type.value, name=t.entity2)
                
            # Add edge
            self.graph.add_edge(
                e1,
                e2,
                relation=t.relation,
                confidence=t.confidence,
            )

    def get_graph_data(self) -> dict[str, Any]:
        """Return the graph in NetworkX node-link JSON format."""
        return nx.node_link_data(self.graph, edges="links")

    def get_stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }

    def save(self):
        """Save the graph to JSON."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph, edges="links")
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load the graph from JSON."""
        if self.graph_path.exists():
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, edges="links")
