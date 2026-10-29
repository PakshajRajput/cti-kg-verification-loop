"""Leakage-resistant selection of training demonstrations."""
from __future__ import annotations

import re
from collections.abc import Iterable

from src.ingestion.models import CTIReport


def normalize_report_text(text: str) -> str:
    """Normalize whitespace and case for exact source-text duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def filter_eligible_demonstrations(
    report: CTIReport,
    demonstrations: Iterable[CTIReport],
) -> list[CTIReport]:
    """Keep only other training reports with distinct normalized source text.

    This is deliberately independent of vector embeddings so the data-isolation
    rule is unit-testable without loading a sentence-transformer model.
    """
    query_text = normalize_report_text(report.text)
    return [
        demo for demo in demonstrations
        if demo.id != report.id
        and demo.split == "train"
        and normalize_report_text(demo.text) != query_text
    ]
