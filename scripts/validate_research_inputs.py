"""Validate real research inputs and reject synthetic/fallback data before benchmarking."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Required research input is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def validate_research_inputs(
    data_dir: Path = Path("data"),
    *,
    check_chroma_collection: bool = True,
) -> dict[str, int]:
    reports_path = data_dir / "processed" / "reports.jsonl"
    entries_path = data_dir / "mitre" / "entries.jsonl"
    corpus_path = data_dir / "mitre" / "bm25_corpus.jsonl"
    chroma_path = data_dir / "chroma" / "chroma.sqlite3"
    bm25_index_path = data_dir / "mitre" / "bm25_index.pkl"

    reports = _read_jsonl(reports_path)
    if len(reports) < 10:
        raise RuntimeError(f"Research evaluation requires at least 10 real reports; found {len(reports)}.")

    synthetic_ids = [
        str(row.get("id", "")) for row in reports
        if str(row.get("id", "")).startswith("synthetic_")
    ]
    if synthetic_ids:
        raise RuntimeError(
            f"Research input contains synthetic reports ({len(synthetic_ids)}). "
            "Remove the synthetic cache and rerun data setup; benchmark runs cannot opt in to synthetic data."
        )

    # Check for duplicate report IDs
    report_ids = [str(row.get("id", "")) for row in reports]
    if len(report_ids) != len(set(report_ids)):
        from collections import Counter
        duplicates = [rid for rid, count in Counter(report_ids).items() if count > 1]
        raise RuntimeError(
            f"Research input contains duplicate report IDs ({len(duplicates)} duplicates found: {duplicates[:5]}). "
            "Rebuild data to remove duplicate reports before evaluation."
        )

    # Check for duplicate document texts (prevent train/test contamination)
    seen_texts: dict[str, str] = {}
    for row in reports:
        raw_text = str(row.get("text", "")).strip().casefold()
        norm_text = " ".join(raw_text.split())
        rid = str(row.get("id", ""))
        if norm_text in seen_texts:
            raise RuntimeError(
                f"Research input contains duplicate report text between '{rid}' and '{seen_texts[norm_text]}'. "
                "Remove duplicate reports to prevent train/test data leakage."
            )
        seen_texts[norm_text] = rid

    annotated_count = sum(bool(row.get("ground_truth_triplets")) for row in reports)
    if annotated_count < 10:
        raise RuntimeError(
            f"Research evaluation requires at least 10 reports with ground-truth triplets; found {annotated_count}."
        )

    entries = _read_jsonl(entries_path)
    if len(entries) < 100:
        raise RuntimeError(
            f"Only {len(entries)} ATT&CK entries were loaded. This appears to be fallback/demo data; "
            "a research run requires at least 100 normalized ATT&CK entries."
        )

    corpus = _read_jsonl(corpus_path)
    if len(corpus) < 100:
        raise RuntimeError(f"BM25 corpus has only {len(corpus)} entries; rebuild the ATT&CK indexes.")
    if not bm25_index_path.is_file():
        raise RuntimeError(f"BM25 index is missing: {bm25_index_path}")
    if not chroma_path.is_file():
        raise RuntimeError(f"ChromaDB persistence file is missing: {chroma_path}")

    chroma_count = None
    if check_chroma_collection:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
            collection = client.get_collection(name="attack_entries")
            chroma_count = collection.count()
        except Exception as error:
            raise RuntimeError(f"Unable to validate the persisted ChromaDB collection: {error}") from error
        if chroma_count < 100 or chroma_count != len(entries):
            raise RuntimeError(
                f"ChromaDB index mismatch: vectors={chroma_count}, normalized ATT&CK entries={len(entries)}. "
                "Rebuild the dense index from the same ATT&CK snapshot."
            )

    if len(corpus) != len(entries):
        raise RuntimeError(
            f"ATT&CK index mismatch: normalized entries={len(entries)}, BM25 corpus={len(corpus)}. "
            "Rebuild both indexes from the same ATT&CK snapshot."
        )

    report_triplets = sum(len(row.get("ground_truth_triplets") or []) for row in reports)
    result = {
        "real_reports": len(reports),
        "annotated_reports": annotated_count,
        "ground_truth_triplets": report_triplets,
        "attack_entries": len(entries),
        "bm25_entries": len(corpus),
        "chroma_vectors": chroma_count if chroma_count is not None else -1,
    }
    print("Research input validation passed:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return result


if __name__ == "__main__":
    try:
        validate_research_inputs(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data"))
    except Exception as error:
        print(f"Research input validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
