import json

import pytest

from src.ingestion.dataset_loader import load_ctinexus_dataset
from src.ingestion.models import CTIReport


def write_cached_reports(data_dir, reports):
    cache = data_dir / "processed" / "reports.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        "".join(report.model_dump_json() + "\n" for report in reports),
        encoding="utf-8",
    )


def make_report(report_id):
    return CTIReport(
        id=report_id,
        source="test",
        title=f"Report {report_id}",
        text="A threat actor used a tool.",
    )


def test_cached_synthetic_data_is_rejected_by_default(tmp_path):
    reports = [make_report(f"real_{i}") for i in range(10)]
    reports.append(make_report("synthetic_0001"))
    write_cached_reports(tmp_path, reports)

    with pytest.raises(RuntimeError, match="synthetic examples"):
        load_ctinexus_dataset(tmp_path)


def test_cached_dataset_with_fewer_than_ten_real_reports_is_rejected(tmp_path):
    write_cached_reports(tmp_path, [make_report(f"real_{i}") for i in range(9)])

    with pytest.raises(RuntimeError, match="at least 10 real annotated reports"):
        load_ctinexus_dataset(tmp_path)


def test_cached_synthetic_data_requires_and_accepts_explicit_demo_opt_in(tmp_path):
    reports = [make_report(f"real_{i}") for i in range(10)]
    reports.append(make_report("synthetic_0001"))
    write_cached_reports(tmp_path, reports)

    train, test = load_ctinexus_dataset(tmp_path, allow_synthetic=True, seed=7)

    assert len(train) + len(test) == 11
    assert any(report.id.startswith("synthetic_") for report in train + test)
