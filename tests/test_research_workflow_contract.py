"""Contract tests for the manually gated research benchmark workflow."""
from pathlib import Path


def test_research_workflow_is_manual_and_uses_real_data_safeguards():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/research-benchmark.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "python scripts/setup_data.py" in workflow
    assert "python scripts/validate_research_inputs.py data" in workflow
    assert "--condition both" in workflow
    assert "--test-split" in workflow
    assert "--seed" in workflow
    assert "secrets.GROQ_API_KEY" in workflow
    assert "secrets.OPENAI_API_KEY" in workflow
    assert "secrets.HF_TOKEN" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "subset={subset_value}" in workflow
    assert "subset={subset}," not in workflow
    assert "--allow-synthetic-data" not in workflow
    assert "openai/gpt-oss-120b" in workflow
    assert "openai/gpt-oss-20b" in workflow
