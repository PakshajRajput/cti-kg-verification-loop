"""
MITRE ATT&CK STIX2 Data Loader.

Downloads the enterprise-attack STIX2 bundle from mitre-attack/attack-stix-data
and extracts techniques, groups, software, mitigations, and tactics into
flat ATT_CK_Entry objects ready for embedding and BM25 indexing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.progress import track

from src.ingestion.models import ATT_CK_Entry

console = Console()

ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)

# Type mapping from STIX object type to our entry type
STIX_TYPE_MAP = {
    "attack-pattern": "technique",
    "intrusion-set": "group",
    "malware": "software",
    "tool": "software",
    "course-of-action": "mitigation",
    "x-mitre-tactic": "tactic",
    "campaign": "campaign",
}


def _extract_external_id(obj: dict[str, Any]) -> str:
    """Extract the ATT&CK ID (e.g. T1566, G0016) from external references."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return obj.get("id", "")


def _extract_tactics(obj: dict[str, Any]) -> list[str]:
    """Extract tactic short names from kill_chain_phases."""
    tactics = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase.get("phase_name", ""))
    return [t for t in tactics if t]


def _safe_description(obj: dict[str, Any]) -> str:
    """Get description, stripping STIX citation markers like (Citation: ...)."""
    desc = obj.get("description", "")
    # Remove citation markers
    import re
    desc = re.sub(r"\(Citation:[^)]+\)", "", desc)
    return desc.strip()


def load_mitre_attack(
    data_dir: Path,
    force_download: bool = False,
) -> list[ATT_CK_Entry]:
    """
    Load MITRE ATT&CK data. Downloads if not cached.

    Returns a flat list of ATT_CK_Entry objects (techniques, groups, software, etc.)
    """
    mitre_dir = data_dir / "mitre"
    mitre_dir.mkdir(parents=True, exist_ok=True)

    raw_path = mitre_dir / "enterprise-attack.json"
    entries_path = mitre_dir / "entries.jsonl"

    # Load from processed cache
    if entries_path.exists() and not force_download:
        console.print(f"[green]Loading cached MITRE ATT&CK entries from {entries_path}[/green]")
        entries = []
        with open(entries_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(ATT_CK_Entry.model_validate_json(line))
        console.print(f"[green]Loaded {len(entries)} ATT&CK entries.[/green]")
        return entries

    # Download raw STIX bundle if needed
    if not raw_path.exists() or force_download:
        console.print(f"[cyan]Downloading MITRE ATT&CK from {ATTACK_STIX_URL}[/cyan]")
        try:
            resp = requests.get(ATTACK_STIX_URL, timeout=120)
            resp.raise_for_status()
            with open(raw_path, "wb") as f:
                f.write(resp.content)
            console.print(f"[green]Saved to {raw_path} ({raw_path.stat().st_size // 1024} KB)[/green]")
        except Exception as e:
            console.print(f"[yellow]Download failed: {e}. Creating minimal fallback entries.[/yellow]")
            return _create_fallback_entries(entries_path)

    # Parse STIX bundle
    console.print("[cyan]Parsing STIX bundle...[/cyan]")
    with open(raw_path, encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    console.print(f"[cyan]Found {len(objects)} STIX objects.[/cyan]")

    entries: list[ATT_CK_Entry] = []
    for obj in track(objects, description="Parsing STIX objects..."):
        stix_type = obj.get("type", "")
        entry_type = STIX_TYPE_MAP.get(stix_type)
        if entry_type is None:
            continue

        # Skip deprecated / revoked objects
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        attack_id = _extract_external_id(obj)
        if not attack_id:
            continue

        name = obj.get("name", "")
        description = _safe_description(obj)
        if not description:
            continue

        aliases = obj.get("aliases", obj.get("x_mitre_aliases", []))
        platforms = obj.get("x_mitre_platforms", [])
        tactics = _extract_tactics(obj)

        entries.append(ATT_CK_Entry(
            id=attack_id,
            name=name,
            description=description,
            type=entry_type,
            external_references=[attack_id],
            aliases=[a for a in aliases if a != name],
            platforms=platforms,
            tactics=tactics,
        ))

    console.print(f"[green]Parsed {len(entries)} ATT&CK entries.[/green]")

    # Save processed entries
    with open(entries_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")
    console.print(f"[green]Saved to {entries_path}[/green]")

    return entries


def _create_fallback_entries(entries_path: Path) -> list[ATT_CK_Entry]:
    """Create a minimal set of hard-coded ATT&CK entries as fallback."""
    entries = [
        ATT_CK_Entry(
            id="T1566", name="Phishing",
            description="Adversaries may send phishing messages to gain access to victim systems. "
                        "All forms of phishing are electronically delivered social engineering.",
            type="technique", tactics=["initial-access"],
        ),
        ATT_CK_Entry(
            id="T1059", name="Command and Scripting Interpreter",
            description="Adversaries may abuse command and script interpreters to execute commands, "
                        "scripts, or binaries. Includes PowerShell, bash, Python.",
            type="technique", tactics=["execution"],
        ),
        ATT_CK_Entry(
            id="T1078", name="Valid Accounts",
            description="Adversaries may obtain and abuse credentials of existing accounts as a means "
                        "of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
            type="technique", tactics=["initial-access", "persistence"],
        ),
        ATT_CK_Entry(
            id="G0016", name="APT29",
            description="APT29 is a threat group attributed to Russia's Foreign Intelligence Service (SVR). "
                        "Also known as Cozy Bear, The Dukes, NOBELIUM.",
            type="group", aliases=["Cozy Bear", "The Dukes", "NOBELIUM"],
        ),
        ATT_CK_Entry(
            id="G0032", name="Lazarus Group",
            description="Lazarus Group is a threat group attributed to North Korea. "
                        "Also known as HIDDEN COBRA, Guardians of Peace.",
            type="group", aliases=["HIDDEN COBRA", "Guardians of Peace"],
        ),
        ATT_CK_Entry(
            id="S0154", name="Cobalt Strike",
            description="Cobalt Strike is a commercial, full-featured, remote access tool that describes "
                        "itself as adversary simulation software designed to execute targeted attacks.",
            type="software",
        ),
        ATT_CK_Entry(
            id="S0002", name="Mimikatz",
            description="Mimikatz is a credential dumper capable of obtaining plaintext Windows "
                        "account logins and passwords, along with many other features.",
            type="software",
        ),
        ATT_CK_Entry(
            id="T1055", name="Process Injection",
            description="Adversaries may inject code into processes in order to evade process-based "
                        "defenses as well as possibly elevate privileges.",
            type="technique", tactics=["defense-evasion", "privilege-escalation"],
        ),
        ATT_CK_Entry(
            id="T1486", name="Data Encrypted for Impact",
            description="Adversaries may encrypt data on target systems or on large numbers of systems "
                        "in a network to interrupt availability to system and network resources. Ransomware.",
            type="technique", tactics=["impact"],
        ),
        ATT_CK_Entry(
            id="T1190", name="Exploit Public-Facing Application",
            description="Adversaries may attempt to exploit a weakness in an Internet-facing host or system "
                        "to initially access a network. Includes web application vulnerabilities, CVEs.",
            type="technique", tactics=["initial-access"],
        ),
    ]

    with open(entries_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")

    console.print(f"[yellow]Saved {len(entries)} fallback ATT&CK entries to {entries_path}[/yellow]")
    return entries
