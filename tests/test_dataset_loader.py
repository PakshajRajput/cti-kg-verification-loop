"""Regression tests for duplicate-safe CTINexus ingestion."""
import pytest

from src.ingestion.dataset_loader import _deduplicate_reports, _find_json_files
from src.ingestion.models import CTIReport


def _report(report_id, text):
    return CTIReport(id=report_id, text=text)


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
    rows = [_report("one", " Same  report text "), _report("two", "same report text")]
    assert [row.id for row in _deduplicate_reports(rows)] == ["one"]


def test_deduplicate_reports_rejects_conflicting_duplicate_ids():
    rows = [_report("same", "one"), _report("same", "two")]
    with pytest.raises(RuntimeError, match="Conflicting duplicate report ID"):
        _deduplicate_reports(rows)
