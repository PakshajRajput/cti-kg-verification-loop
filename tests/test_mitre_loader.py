from unittest.mock import patch
import pytest

from src.ingestion.mitre_loader import load_mitre_attack


def test_load_mitre_attack_rejects_download_failure_in_research_mode(tmp_path):
    with patch("requests.get") as mock_get:
        mock_get.side_effect = ConnectionError("Network unreachable")

        with pytest.raises(RuntimeError, match="Refusing to generate fallback entries"):
            load_mitre_attack(tmp_path, force_download=True, allow_fallback=False)


def test_load_mitre_attack_allows_fallback_when_explicitly_requested(tmp_path):
    with patch("requests.get") as mock_get:
        mock_get.side_effect = ConnectionError("Network unreachable")

        entries = load_mitre_attack(tmp_path, force_download=True, allow_fallback=True)
        assert len(entries) > 0
        assert (tmp_path / "mitre" / "entries.jsonl").is_file()
