import numpy as np
import pytest

from src.agents.extractor import _retrieve_demonstrations
from src.ingestion.models import CTIReport


class DummyEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return np.ones((len(texts), 4))


def test_retrieve_demonstrations_filters_out_query_report_by_id():
    query_report = CTIReport(id="rep-1", text="Report about APT29 attacking a financial institution.")
    demo1 = CTIReport(id="rep-1", text="Different text with identical ID.")
    demo2 = CTIReport(id="rep-2", text="Demonstration report about Lazarus group.")

    demos = _retrieve_demonstrations(query_report, [demo1, demo2], DummyEmbedder(), k=5)
    assert len(demos) == 1
    assert demos[0].id == "rep-2"


def test_retrieve_demonstrations_filters_out_duplicate_normalized_text():
    query_report = CTIReport(id="query-report", text="  Threat actor APT28 deployed malware   in 2024.  ")
    duplicate_demo = CTIReport(id="demo-dup", text="threat actor apt28 deployed malware in 2024.")
    clean_demo = CTIReport(id="demo-clean", text="Unrelated report about ransomware targeting hospitals.")

    demos = _retrieve_demonstrations(query_report, [duplicate_demo, clean_demo], DummyEmbedder(), k=5)
    assert len(demos) == 1
    assert demos[0].id == "demo-clean"


def test_retrieve_demonstrations_filters_out_test_split_reports():
    query_report = CTIReport(id="query", text="Some query text.")
    held_out_test = CTIReport(id="test-demo", text="Test split report text.", split="test")
    valid_train = CTIReport(id="train-demo", text="Train split report text.", split="train")

    demos = _retrieve_demonstrations(query_report, [held_out_test, valid_train], DummyEmbedder(), k=5)
    assert len(demos) == 1
    assert demos[0].id == "train-demo"
