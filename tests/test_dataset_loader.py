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
        text=f"A threat actor used a tool in {report_id}.",
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


def test_cached_dataset_rejects_duplicate_report_ids(tmp_path):
    reports = [make_report(f"real_{i}") for i in range(10)]
    reports.append(make_report("real_0"))  # Duplicate ID
    write_cached_reports(tmp_path, reports)

    with pytest.raises(RuntimeError, match="duplicate report ID"):
        load_ctinexus_dataset(tmp_path)


def test_cached_dataset_rejects_duplicate_report_texts(tmp_path):
    reports = [make_report(f"real_{i}") for i in range(10)]
    dup_report = make_report("real_unique_id")
    dup_report.text = reports[0].text  # Duplicate text
    reports.append(dup_report)
    write_cached_reports(tmp_path, reports)

    with pytest.raises(RuntimeError, match="duplicate report text"):
        load_ctinexus_dataset(tmp_path)


def test_find_json_files_restricts_to_annotation_dir_ignoring_demo_and_test(tmp_path):
    from src.ingestion.dataset_loader import _find_json_files

    ann_dir = tmp_path / "ctinexus-main" / "ctinexus" / "data" / "annotation"
    demo_dir = tmp_path / "ctinexus-main" / "ctinexus" / "data" / "demo"
    test_dir = tmp_path / "ctinexus-main" / "ctinexus" / "data" / "test"
    ann_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    (ann_dir / "report1.json").write_text("{}", encoding="utf-8")
    (demo_dir / "report1.json").write_text("{}", encoding="utf-8")
    (test_dir / "report1.json").write_text("{}", encoding="utf-8")

    files = _find_json_files(tmp_path)
    assert len(files) == 1
    assert files[0] == ann_dir / "report1.json"


def test_split_reports_asserts_disjoint_ids_and_texts():
    from src.ingestion.dataset_loader import _split_reports

    reports = [make_report(f"real_{i}") for i in range(10)]
    train, test = _split_reports(reports, test_split=0.2, seed=42)
    assert len(train) == 8
    assert len(test) == 2
    assert not (set(r.id for r in train) & set(r.id for r in test))
    assert not ({r.text for r in train} & {r.text for r in test})
