"""Regression tests for duplicate-safe CTINexus ingestion and cache safeguards."""
import pytest

from src.ingestion.dataset_loader import (
    _deduplicate_reports,
    _find_json_files,
    load_ctinexus_dataset,
)
from src.ingestion.models import CTIReport


def write_cached_reports(data_dir, reports):
    cache = data_dir / "processed" / "reports.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        "".join(report.model_dump_json() + "\n" for report in reports),
        encoding="utf-8",
    )


def make_report(report_id):
    # Text must be unique per ID for cache-policy tests; otherwise content
    # deduplication correctly treats these as duplicated source reports.
    return CTIReport(
        id=report_id,
        source="test",
        title=f"Report {report_id}",
        text=f"A threat actor used a tool in source report {report_id}.",
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


def test_annotation_discovery_ignores_demo_and_repo_tests(tmp_path):
    raw = tmp_path / "ctinexus_raw" / "CTINexus-main"
    annotation = raw / "data" / "annotation"
    annotation.mkdir(parents=True)
    (raw / "data" / "demo").mkdir(parents=True)
    (raw / "tests").mkdir(parents=True)
    (annotation / "report.json").write_text("{}", encoding="utf-8")
    (raw / "data" / "demo" / "demo.json").write_text("{}", encoding="utf-8")
    (raw / "tests" / "fixture.json").write_text("{}", encoding="utf-8")

    assert _find_json_files(tmp_path / "ctinexus_raw") == [annotation / "report.json"]


def test_annotation_discovery_fails_closed_if_annotation_directory_is_missing(tmp_path):
    (tmp_path / "ctinexus_raw" / "data" / "demo").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="refusing to scan"):
        _find_json_files(tmp_path / "ctinexus_raw")


def test_deduplicate_reports_removes_exact_text_clones():
    rows = [
        CTIReport(id="one", text=" Same  report text "),
        CTIReport(id="two", text="same report text"),
    ]

    assert [row.id for row in _deduplicate_reports(rows)] == ["one"]


def test_deduplicate_reports_rejects_conflicting_duplicate_ids():
    rows = [
        CTIReport(id="same", text="one"),
        CTIReport(id="same", text="two"),
    ]

    with pytest.raises(RuntimeError, match="Conflicting duplicate report ID"):
        _deduplicate_reports(rows)



def test_demo_selection_excludes_query_clone_and_non_train_reports():
    from src.ingestion.demo_selection import filter_eligible_demonstrations

    query = CTIReport(id="query", text="APT uses tool", split="test")
    demonstrations = [
        CTIReport(id="query", text="different text", split="train"),
        CTIReport(id="clone", text="  apt   uses tool ", split="train"),
        CTIReport(id="held-out", text="another unique report", split="test"),
        CTIReport(id="train-good", text="distinct training report", split="train"),
    ]

    selected = filter_eligible_demonstrations(query, demonstrations)

    assert [demo.id for demo in selected] == ["train-good"]
