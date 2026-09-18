"""
CTINexus Dataset Loader.

Downloads the CTINexus dataset from GitHub (peng-gao-lab/CTINexus)
and parses annotated JSON files into CTIReport objects.

The CTINexus dataset annotation format:
  Each report JSON contains:
    - "report_text": str
    - "entities": [{"text": str, "type": str, "start": int, "end": int}, ...]
    - "relations": [{"entity1": str, "relation": str, "entity2": str}, ...]
"""
from __future__ import annotations

import json
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.progress import track

from src.ingestion.models import CTIReport, EntityType, Triplet

console = Console()

CTINEXUS_REPO = "https://github.com/peng-gao-lab/CTINexus"
CTINEXUS_ZIP = "https://github.com/peng-gao-lab/CTINexus/archive/refs/heads/main.zip"

# Known alternative: the dataset might be in a separate data release
DATASET_DIRS = [
    "data/annotation",
    "dataset",
    "data/dataset",
    "annotation",
    "CTINexus-main/data/annotation",
    "CTINexus-main/dataset",
]


def _map_entity_type(raw: str) -> EntityType:
    """Map raw annotation type strings to our EntityType enum."""
    mapping = {
        "malware": EntityType.MALWARE,
        "threat-actor": EntityType.THREAT_ACTOR,
        "threat actor": EntityType.THREAT_ACTOR,
        "threatactor": EntityType.THREAT_ACTOR,
        "tool": EntityType.TOOL,
        "vulnerability": EntityType.VULNERABILITY,
        "cve": EntityType.VULNERABILITY,
        "target": EntityType.TARGET,
        "victim": EntityType.TARGET,
        "technique": EntityType.TECHNIQUE,
        "attack-pattern": EntityType.TECHNIQUE,
        "tactic": EntityType.TACTIC,
        "campaign": EntityType.CAMPAIGN,
        "ioc": EntityType.IOC,
        "indicator": EntityType.IOC,
        "ip": EntityType.IOC,
        "domain": EntityType.IOC,
        "hash": EntityType.IOC,
        "url": EntityType.IOC,
    }
    return mapping.get(raw.lower().strip(), EntityType.UNKNOWN)


def _parse_report_json(data: dict, report_id: str, source: str = "") -> CTIReport:
    """Parse a single CTINexus annotated report dict into a CTIReport."""
    # Handle varying field names across CTINexus versions
    text = (
        data.get("report_text")
        or data.get("text")
        or data.get("content")
        or data.get("body")
        or ""
    )
    title = data.get("title", data.get("report_title", ""))

    # Parse entities
    raw_entities = data.get("entities", data.get("entity_list", []))
    entities = []
    for e in raw_entities:
        entities.append({
            "text": e.get("text", e.get("mention", e.get("name", e.get("entity_name", "")))),
            "type": e.get("type", e.get("entity_type", "Unknown")),
        })

    # Parse relations / triplets
    raw_rels = data.get("relations", data.get("relation_list", data.get("triplets", data.get("explicit_triplets", []))))
    triplets = []
    for r in raw_rels:
        e1 = r.get("entity1", r.get("head", r.get("subject", "")))
        e2 = r.get("entity2", r.get("tail", r.get("object", "")))
        rel = r.get("relation", r.get("relation_type", r.get("predicate", "")))
        e1_type = r.get("entity1_type", r.get("head_type", "Unknown"))
        e2_type = r.get("entity2_type", r.get("tail_type", "Unknown"))
        if e1 and e2 and rel:
            triplets.append(Triplet(
                entity1=e1,
                entity1_type=_map_entity_type(e1_type),
                relation=rel,
                entity2=e2,
                entity2_type=_map_entity_type(e2_type),
            ))

    return CTIReport(
        id=report_id,
        source=source or data.get("source", ""),
        title=title,
        text=text,
        ground_truth_entities=entities,
        ground_truth_relations=raw_rels,
        ground_truth_triplets=triplets,
    )


def _find_json_files(base_dir: Path) -> list[Path]:
    """Recursively find all JSON files that look like CTI report annotations."""
    json_files = []
    for path in base_dir.rglob("*.json"):
        # Skip index/config files
        if path.name in ("package.json", "index.json", "config.json"):
            continue
        json_files.append(path)
    return sorted(json_files)


def download_ctinexus_dataset(data_dir: Path) -> Path:
    """
    Download the CTINexus dataset from GitHub.
    Returns the path to the directory containing JSON annotation files.
    """
    ctinexus_dir = data_dir / "ctinexus"
    ctinexus_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ctinexus.zip"
    extract_path = data_dir / "ctinexus_raw"

    if extract_path.exists() and any(extract_path.rglob("*.json")):
        console.print("[green]CTINexus already downloaded.[/green]")
        return extract_path

    console.print(f"[cyan]Downloading CTINexus from {CTINEXUS_ZIP}[/cyan]")
    try:
        resp = requests.get(CTINEXUS_ZIP, timeout=120, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        console.print("[cyan]Extracting...[/cyan]")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

        zip_path.unlink(missing_ok=True)
        console.print(f"[green]Extracted to {extract_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Download failed: {e}. Will use synthetic demo data.[/yellow]")
        return extract_path

    return extract_path


def _create_synthetic_reports(n: int = 30) -> list[CTIReport]:
    """
    Create synthetic CTI report annotations for testing when the real dataset
    is not available. These are simplified but structurally correct examples.
    """
    console.print(f"[yellow]Creating {n} synthetic demo reports for testing.[/yellow]")
    templates = [
        {
            "text": "APT29, also known as Cozy Bear, has been observed using Cobalt Strike "
                    "beacons for command and control. The threat actor exploited CVE-2021-40444 "
                    "to gain initial access to financial institutions.",
            "triplets": [
                Triplet(entity1="APT29", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="Cobalt Strike", entity2_type=EntityType.TOOL),
                Triplet(entity1="APT29", entity1_type=EntityType.THREAT_ACTOR,
                        relation="exploits", entity2="CVE-2021-40444", entity2_type=EntityType.VULNERABILITY),
                Triplet(entity1="APT29", entity1_type=EntityType.THREAT_ACTOR,
                        relation="targets", entity2="financial institutions", entity2_type=EntityType.TARGET),
            ],
            "source": "Symantec",
        },
        {
            "text": "Lazarus Group deployed WannaCry ransomware targeting healthcare organizations. "
                    "The malware used EternalBlue exploit to spread laterally across networks.",
            "triplets": [
                Triplet(entity1="Lazarus Group", entity1_type=EntityType.THREAT_ACTOR,
                        relation="deploys", entity2="WannaCry", entity2_type=EntityType.MALWARE),
                Triplet(entity1="WannaCry", entity1_type=EntityType.MALWARE,
                        relation="uses", entity2="EternalBlue", entity2_type=EntityType.TOOL),
                Triplet(entity1="Lazarus Group", entity1_type=EntityType.THREAT_ACTOR,
                        relation="targets", entity2="healthcare organizations", entity2_type=EntityType.TARGET),
            ],
            "source": "Trend Micro",
        },
        {
            "text": "The FIN7 group used Carbanak malware to steal financial data. "
                    "The attackers performed spear-phishing attacks to deliver the payload via macro-enabled documents.",
            "triplets": [
                Triplet(entity1="FIN7", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="Carbanak", entity2_type=EntityType.MALWARE),
                Triplet(entity1="FIN7", entity1_type=EntityType.THREAT_ACTOR,
                        relation="performs", entity2="Spear Phishing", entity2_type=EntityType.TECHNIQUE),
                Triplet(entity1="Carbanak", entity1_type=EntityType.MALWARE,
                        relation="steals", entity2="financial data", entity2_type=EntityType.TARGET),
            ],
            "source": "The Hacker News",
        },
        {
            "text": "Sandworm Team exploited vulnerabilities in industrial control systems. "
                    "The group used BlackEnergy malware and Industroyer to attack Ukrainian energy infrastructure.",
            "triplets": [
                Triplet(entity1="Sandworm Team", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="BlackEnergy", entity2_type=EntityType.MALWARE),
                Triplet(entity1="Sandworm Team", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="Industroyer", entity2_type=EntityType.MALWARE),
                Triplet(entity1="Sandworm Team", entity1_type=EntityType.THREAT_ACTOR,
                        relation="targets", entity2="Ukrainian energy infrastructure", entity2_type=EntityType.TARGET),
            ],
            "source": "Symantec",
        },
        {
            "text": "A new campaign by Turla group leveraged PowerShell scripts and mimikatz "
                    "for credential dumping. The attackers established persistence via scheduled tasks.",
            "triplets": [
                Triplet(entity1="Turla", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="PowerShell", entity2_type=EntityType.TOOL),
                Triplet(entity1="Turla", entity1_type=EntityType.THREAT_ACTOR,
                        relation="uses", entity2="Mimikatz", entity2_type=EntityType.TOOL),
                Triplet(entity1="Turla", entity1_type=EntityType.THREAT_ACTOR,
                        relation="establishes", entity2="Scheduled Task Persistence", entity2_type=EntityType.TECHNIQUE),
            ],
            "source": "Trend Micro",
        },
    ]

    reports = []
    for i in range(n):
        template = templates[i % len(templates)]
        report_id = f"synthetic_{i:04d}"
        reports.append(CTIReport(
            id=report_id,
            source=template["source"],
            title=f"CTI Report #{i}",
            text=template["text"],
            ground_truth_triplets=template["triplets"],  # type: ignore[arg-type]
        ))
    return reports


def load_ctinexus_dataset(
    data_dir: Path,
    test_split: float = 0.2,
    seed: int = 42,
    force_synthetic: bool = False,
) -> tuple[list[CTIReport], list[CTIReport]]:
    """
    Load CTINexus dataset. Returns (train_reports, test_reports).

    Args:
        data_dir: Root data directory.
        test_split: Fraction of data held out for evaluation.
        seed: Random seed for reproducible splits.
        force_synthetic: Use synthetic data even if real data exists.
    """
    processed_path = data_dir / "processed" / "reports.jsonl"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Load from cache if available
    if processed_path.exists() and not force_synthetic:
        console.print(f"[green]Loading cached dataset from {processed_path}[/green]")
        reports = []
        with open(processed_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reports.append(CTIReport.model_validate_json(line))
        console.print(f"[green]Loaded {len(reports)} reports from cache.[/green]")
        return _split_reports(reports, test_split, seed)

    # Try to load from CTINexus directory
    raw_dir = data_dir / "ctinexus_raw"
    reports: list[CTIReport] = []

    if not force_synthetic and raw_dir.exists():
        json_files = _find_json_files(raw_dir)
        console.print(f"[cyan]Found {len(json_files)} JSON files in CTINexus directory.[/cyan]")

        for fp in track(json_files, description="Parsing reports..."):
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        rid = f"{fp.stem}_{i}"
                        r = _parse_report_json(item, rid, source=fp.parent.name)
                        if r.text.strip():
                            reports.append(r)
                elif isinstance(data, dict):
                    r = _parse_report_json(data, fp.stem, source=fp.parent.name)
                    if r.text.strip():
                        reports.append(r)
            except Exception as e:
                console.print(f"[yellow]Skipping {fp.name}: {e}[/yellow]")

    if len(reports) < 10:
        console.print(f"[yellow]Only {len(reports)} real reports found. Supplementing with synthetic data.[/yellow]")
        synthetic = _create_synthetic_reports(max(30, 150 - len(reports)))
        reports = reports + synthetic

    console.print(f"[green]Total reports: {len(reports)}[/green]")

    # Save to cache
    with open(processed_path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(r.model_dump_json() + "\n")
    console.print(f"[green]Saved to {processed_path}[/green]")

    # Build demonstration examples (train split only)
    train, test = _split_reports(reports, test_split, seed)
    _save_demonstrations(train, data_dir / "processed" / "demonstrations.json")

    return train, test


def _split_reports(
    reports: list[CTIReport],
    test_split: float,
    seed: int,
) -> tuple[list[CTIReport], list[CTIReport]]:
    rng = random.Random(seed)
    shuffled = reports.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_split))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    for r in train:
        r.split = "train"
    for r in test:
        r.split = "test"
    console.print(f"[green]Split: {len(train)} train / {len(test)} test[/green]")
    return train, test


def _save_demonstrations(train: list[CTIReport], path: Path) -> None:
    """Save training reports as few-shot demonstration examples."""
    # Only keep reports that have ground-truth triplets (for demonstration retrieval)
    demos = [r for r in train if r.ground_truth_triplets][:100]
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in demos], f, indent=2, default=str)
    console.print(f"[green]Saved {len(demos)} demonstration examples to {path}[/green]")
