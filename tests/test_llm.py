"""Tests for the LLM client."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.llm import JudgeLLM, JudgeRuling


def test_answer_parses_valid_json():
    """Should parse structured JSON response from Ollama."""
    mock_response = {
        "message": {
            "content": '{"answer": "Protection prevents damage from sources with the stated quality (CR 702.16e).", "confidence": "high", "citations": ["CR 702.16e"]}'
        }
    }

    with patch.object(JudgeLLM, "_preload"), \
         patch("src.llm.ollama.chat", return_value=mock_response):
        llm = JudgeLLM()
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

    with patch.object(JudgeLLM, "_preload"), \
         patch("src.llm.ollama.chat", return_value=mock_response):
        llm = JudgeLLM()
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

    with patch.object(JudgeLLM, "_preload"), \
         patch("src.llm.ollama.chat", return_value=mock_response):
        llm = JudgeLLM()
        ruling = llm.answer("test query", [{"source": "cr", "section": "100.1", "text": "..."}])

    assert ruling.confidence == "low"
    assert ruling.citations == []
    assert "I don't know" in ruling.answer


def test_answer_includes_context_chunks_in_prompt():
    """Should include retrieved context in the user message."""
    mock_response = {"message": {"content": '{"answer": "test", "confidence": "high", "citations": []}'}}

    with patch.object(JudgeLLM, "_preload"), \
         patch("src.llm.ollama.chat", return_value=mock_response) as mock_chat:
        llm = JudgeLLM()
        llm.answer("What is protection?", [
            {"source": "cr", "section": "702.16a", "text": "Protection is a static ability..."},
            {"source": "scryfall", "name": "Lightning Bolt", "text": "Lightning Bolt deals 3 damage..."},
        ])

    messages = mock_chat.call_args.kwargs["messages"]
    user_msg = messages[1]["content"]
    assert "702.16a" in user_msg
    assert "Lightning Bolt" in user_msg
    assert "Respond as JSON:" in user_msg
