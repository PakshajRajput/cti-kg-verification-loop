"""
Demo script for querying the Neo4j knowledge graph.

Runs a set of pre-built Cypher queries and prints results in formatted tables.

Usage:
    python scripts/demo_query.py
    python scripts/demo_query.py --query actor_techniques
    python scripts/demo_query.py --list-queries
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console
from rich.table import Table

from src.kg.cypher_queries import QUERIES, run_named_query
from src.kg.neo4j_writer import Neo4jWriter

console = Console()
app = typer.Typer()


@app.command()
def main(
    query: str = typer.Option(None, help="Named query to run (default: runs all demo queries)"),
    list_queries: bool = typer.Option(False, "--list-queries", help="List all available queries"),
):
    """Demo: Query the Neo4j CTI knowledge graph."""
    if list_queries:
        console.print("[bold cyan]Available Named Queries:[/bold cyan]")
        for name in QUERIES:
            console.print(f"  • {name}")
        return

    # Connect to Neo4j
    console.rule("[cyan]Connecting to Neo4j")
    neo4j = Neo4jWriter.from_env()
    try:
        neo4j.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect to Neo4j: {e}[/red]")
        console.print(
            "[yellow]Make sure Neo4j is running:\n"
            "  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j[/yellow]"
        )
        sys.exit(1)

    # Print graph stats
    stats = neo4j.get_stats()
    console.print(
        f"[green]Connected. Graph: {stats['entity_nodes']} entities, "
        f"{stats['total_edges']} edges, {stats['report_nodes']} reports.[/green]"
    )

    # Run selected queries
    demo_queries = [query] if query else [
        "graph_stats",
        "top_threat_actors",
        "top_malware",
        "actor_techniques",
        "vulnerability_exploiters",
    ]

    with neo4j._driver.session() as session:
        for q_name in demo_queries:
            if q_name not in QUERIES:
                console.print(f"[yellow]Unknown query: {q_name}[/yellow]")
                continue

            console.rule(f"[cyan]{q_name}")
            try:
                results = run_named_query(session, q_name)
            except Exception as e:
                console.print(f"[red]Query error: {e}[/red]")
                continue

            if not results:
                console.print("[dim](No results)[/dim]")
                continue

            # Build table from result dicts
            table = Table(title=f"Query: {q_name} ({len(results)} rows)")
            headers = list(results[0].keys())
            for h in headers:
                table.add_column(h, style="cyan" if "actor" in h or "entity" in h else "white")

            for row in results[:20]:  # limit display to 20 rows
                table.add_row(*[str(row.get(h, "")) for h in headers])

            console.print(table)

    neo4j.close()


if __name__ == "__main__":
    app()
