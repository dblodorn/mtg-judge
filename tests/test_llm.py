"""Tests for the LLM client."""

from unittest.mock import patch

from src.llm import JudgeLLM, JudgeRuling, verify_citations


def _llm(mock_response) -> JudgeLLM:
    """JudgeLLM with Ollama mocked out."""
    with patch("src.llm.ollama.Client") as client_cls:
        client_cls.return_value.chat.return_value = mock_response
        return JudgeLLM()


def test_answer_parses_valid_json():
    """Should parse structured JSON response from Ollama."""
    mock_response = {
        "message": {
            "content": '{"answer": "Protection prevents damage from sources with the stated quality (CR 702.16e).", '
            '"confidence": "high", "citations": ["CR 702.16e"]}'
        }
    }

    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        ruling = llm.answer("Does protection stop lightning bolt?", [{"source": "cr", "section": "702.16e", "text": "..."}])

    assert isinstance(ruling, JudgeRuling)
    assert "702.16e" in ruling.citations[0]
    assert ruling.confidence == "high"


def test_answer_handles_json_in_code_block():
    """Should extract JSON even if wrapped in ```json```."""
    mock_response = {
        "message": {
            "content": '```json\n{"answer": "Yes.", "confidence": "high", "citations": ["CR 702.16a"]}\n```'
        }
    }

    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        ruling = llm.answer("test query", [{"source": "cr", "section": "702.16a", "text": "..."}])

    assert ruling.answer == "Yes."
    assert ruling.citations == ["CR 702.16a"]


def test_answer_falls_back_on_invalid_json():
    """Should return low-confidence plain text if JSON parsing fails."""
    mock_response = {
        "message": {
            "content": "I don't know the answer to that."
        }
    }

    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        ruling = llm.answer("test query", [{"source": "cr", "section": "100.1", "text": "..."}])

    assert ruling.confidence == "low"
    assert ruling.citations == []
    assert "I don't know" in ruling.answer


def test_answer_includes_context_chunks_in_prompt():
    """Should include retrieved context in the user message."""
    mock_response = {"message": {"content": '{"answer": "test", "confidence": "high", "citations": []}'}}

    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        llm.answer("What is protection?", [
            {"source": "cr", "section": "702.16a", "text": "Protection is a static ability..."},
            {"source": "scryfall", "name": "Lightning Bolt", "text": "Lightning Bolt deals 3 damage..."},
        ])

    messages = llm.client.chat.call_args.kwargs["messages"]
    user_msg = messages[1]["content"]
    assert "702.16a" in user_msg
    assert "Lightning Bolt" in user_msg
    assert "Respond as JSON:" in user_msg


def test_answer_requests_schema_and_context_window():
    """Should constrain output to the JudgeRuling schema and set num_ctx."""
    mock_response = {"message": {"content": '{"answer": "x", "confidence": "high", "citations": []}'}}
    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        llm.answer("q", [{"source": "cr", "section": "100.1", "text": "..."}])

    kwargs = llm.client.chat.call_args.kwargs
    assert kwargs["format"] == JudgeRuling.model_json_schema()
    assert kwargs["options"]["num_ctx"] > 2048


def test_verify_citations_drops_unsupported_and_downgrades():
    chunks = [{"source": "cr", "section": "702.16", "text": "702.16e Any damage that would be dealt..."}]
    ruling = JudgeRuling(answer="a", confidence="high", citations=["CR 702.16e", "CR 999.1a"])

    checked = verify_citations(ruling, chunks)

    assert checked.citations == ["CR 702.16e"]
    assert checked.confidence == "low"


def test_verify_citations_keeps_supported_card_names():
    chunks = [{"source": "scryfall", "name": "Lightning Bolt", "section": "Lightning Bolt", "text": "Lightning Bolt deals 3 damage"}]
    ruling = JudgeRuling(answer="a", confidence="high", citations=["Lightning Bolt"])

    assert verify_citations(ruling, chunks) == ruling


def test_answer_survives_ollama_response_error():
    """A degenerate generation (e.g. repeat limit) returns a low-confidence fallback, not a crash."""
    import ollama

    with patch.object(JudgeLLM, "_preload"), patch("src.llm.ollama.Client") as client_cls:
        client_cls.return_value.chat.side_effect = ollama.ResponseError("prediction aborted, token repeat limit reached", 500)
        llm = JudgeLLM()
        ruling = llm.answer("q", [{"source": "cr", "section": "100.1", "text": "..."}])

    assert ruling.confidence == "low"
    assert ruling.citations == []


def test_answer_includes_format_when_given():
    mock_response = {"message": {"content": '{"answer": "x", "confidence": "high", "citations": []}'}}
    with patch.object(JudgeLLM, "_preload"):
        llm = _llm(mock_response)
        llm.answer("is it legal?", [{"source": "cr", "section": "100.1", "text": "..."}], game_format="commander")

    assert "Format being played: Commander" in llm.client.chat.call_args.kwargs["messages"][1]["content"]
