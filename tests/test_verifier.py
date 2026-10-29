from unittest.mock import MagicMock, patch
import pytest

from src.agents.verifier import VerifierAgent, make_verifier_node
from src.ingestion.models import CTIReport, EntityType, PipelineState, Triplet, VerificationVerdict


def make_test_triplet(e1="APT29", rel="uses", e2="Cobalt Strike"):
    return Triplet(
        entity1=e1,
        entity1_type=EntityType.THREAT_ACTOR,
        relation=rel,
        entity2=e2,
        entity2_type=EntityType.TOOL,
    )


def test_verifier_call_llm_incorporates_report_text():
    mock_retriever = MagicMock()
    verifier = VerifierAgent(retriever=mock_retriever)

    triplet = make_test_triplet()
    report_text = "Detailed threat report regarding Russian cyber activity."

    with patch("src.agents.verifier.call_llm") as mock_call:
        mock_call.return_value = '{"verdict":"valid","corrected_triplet":null,"reasoning":"Confirmed"}'
        verifier._call_llm(triplet, context="ATT&CK context snippet", report_text=report_text)

        assert mock_call.called
        messages = mock_call.call_args[1]["messages"]
        user_content = messages[1]["content"]
        assert "## Original CTI Report Excerpt" in user_content
        assert report_text in user_content


def test_verifier_node_passes_source_report_text_to_agent():
    mock_agent = MagicMock()
    mock_agent.verify_triplets.return_value = ([], [], [], False)

    node = make_verifier_node(mock_agent)
    state = PipelineState(
        report=CTIReport(id="rep-1", text="Source report text for verification."),
        canonical_triplets=[make_test_triplet()],
    )

    node(state.model_dump())
    assert mock_agent.verify_triplets.called
    call_kwargs = mock_agent.verify_triplets.call_args[1]
    assert call_kwargs.get("report_text") == "Source report text for verification."
