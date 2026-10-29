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
    """Find annotation JSON files only; never crawl demo or repository test fixtures."""
    candidates = [
        base_dir / "CTINexus-main" / "data" / "annotation",
        base_dir / "data" / "annotation",
        base_dir / "annotation",
    ]
    annotation_dir = next((path for path in candidates if path.is_dir()), None)
    if annotation_dir is None:
        raise RuntimeError(
            f"Could not locate CTINexus annotation directory under {base_dir}. "
            "Expected data/annotation; refusing to scan demo/test-fixture directories."
        )
    files = sorted(
        path for path in annotation_dir.rglob("*.json")
        if path.name not in ("package.json", "index.json", "config.json")
    )
    if not files:
        raise RuntimeError(f"No annotation JSON files found in {annotation_dir}")
    return files


def _normalized_report_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _annotation_signature(report: CTIReport) -> str:
    return json.dumps(
        [triplet.model_dump(mode="json") for triplet in report.ground_truth_triplets],
        sort_keys=True,
        separators=(",", ":"),
    )


def _deduplicate_reports(reports: list[CTIReport]) -> list[CTIReport]:
    """Drop identical copies; fail closed when the same text has conflicting labels."""
    by_id: dict[str, CTIReport] = {}
    by_text: dict[str, CTIReport] = {}
    unique: list[CTIReport] = []
    for report in reports:
        normalized_text = _normalized_report_text(report.text)
        if not normalized_text:
            continue
        previous_id = by_id.get(report.id)
        if previous_id is not None:
            if (
                _normalized_report_text(previous_id.text) != normalized_text
                or _annotation_signature(previous_id) != _annotation_signature(report)
            ):
                raise RuntimeError(
                    f"Conflicting duplicate report ID {report.id!r}; resolve dataset provenance before evaluation."
                )
            continue
        previous_text = by_text.get(normalized_text)
        if previous_text is not None:
            if _annotation_signature(previous_text) != _annotation_signature(report):
                raise RuntimeError(
                    "Duplicate source report text has conflicting ground-truth annotations; "
                    "resolve dataset provenance before evaluation."
                )
            continue
        by_id[report.id] = report
        by_text[normalized_text] = report
        unique.append(report)
    return unique


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
    allow_synthetic: bool = False,
) -> tuple[list[CTIReport], list[CTIReport]]:
    """
    Load CTINexus dataset. Returns (train_reports, test_reports).

    Args:
        data_dir: Root data directory.
        test_split: Fraction of data held out for evaluation.
        seed: Random seed for reproducible splits.
        force_synthetic: Explicitly use synthetic data even if real data exists.
        allow_synthetic: Allow synthetic supplementation if fewer than 10 real
            reports are available. Keep False for research evaluation.
    """
    processed_path = data_dir / "processed" / "reports.jsonl"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if force_synthetic and not allow_synthetic:
        raise RuntimeError(
            "force_synthetic=True requires allow_synthetic=True. "
            "Synthetic data is for demo/debug runs only, not research evaluation."
        )

    # Load from cache if available, but enforce the same provenance guard as raw data.
    if processed_path.exists() and not force_synthetic:
        console.print(f"[green]Loading cached dataset from {processed_path}[/green]")
        reports = []
        with open(processed_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reports.append(CTIReport.model_validate_json(line))
        original_count = len(reports)
        reports = _deduplicate_reports(reports)
        if len(reports) != original_count:
            console.print(f"[yellow]Removed {original_count - len(reports)} duplicate reports from cached data.[/yellow]")
            with processed_path.open("w", encoding="utf-8") as handle:
                for report in reports:
                    handle.write(report.model_dump_json() + "\n")
        synthetic_count = sum(report.id.startswith("synthetic_") for report in reports)
        real_count = len(reports) - synthetic_count
        if not allow_synthetic and (synthetic_count > 0 or real_count < 10):
            raise RuntimeError(
                f"Cached dataset contains {real_count} real reports and "
                f"{synthetic_count} synthetic reports. Research evaluation requires "
                "at least 10 real annotated reports and no synthetic examples. "
                "Use a clean dataset cache or explicitly enable synthetic data for demos."
            )
        console.print(
            f"[green]Loaded {len(reports)} reports from cache "
            f"({real_count} real, {synthetic_count} synthetic).[/green]"
        )
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

    if force_synthetic:
        reports = _create_synthetic_reports(30)
    else:
        reports = _deduplicate_reports(reports)
        if len(reports) < 10:
            if not allow_synthetic:
                raise RuntimeError(
                    f"Only {len(reports)} unique annotated reports were loaded from CTINexus. "
                    "Refusing to mix synthetic examples into the research dataset. "
                    "Check the dataset annotation path or explicitly pass allow_synthetic=True "
                    "for a demo-only run."
                )
            console.print(
                f"[yellow]Only {len(reports)} real reports found. "
                "Supplementing with synthetic demo data; do not use these results as research evidence.[/yellow]"
            )
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
