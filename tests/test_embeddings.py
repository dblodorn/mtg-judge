"""Tests for the embeddings client."""

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import Embeddings, EmbeddingsUnavailable


def _mock_response(n: int, value: float = 0.1):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[value] * 768 for _ in range(n)]}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def test_embed_returns_vector():
    """embed() should return a list of floats."""
    with patch("src.embeddings.requests.post", return_value=_mock_response(1)) as mock_post:
        vec = Embeddings().embed("protection from red")

        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)
        mock_post.assert_called_once()


def test_embed_uses_query_prefix():
    """Should POST to /api/embed with the nomic search_query prefix."""
    with patch("src.embeddings.requests.post", return_value=_mock_response(1, 0.0)) as mock_post:
        emb = Embeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        emb.embed("lightning bolt")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/api/embed")
        assert kwargs["json"]["model"] == "nomic-embed-text"
        assert kwargs["json"]["input"] == ["search_query: lightning bolt"]


def test_embed_raises_on_connection_error():
    """Should raise EmbeddingsUnavailable when Ollama is not running."""
    with patch("src.embeddings.requests.post") as mock_post:
        import requests

        mock_post.side_effect = requests.ConnectionError()

        with pytest.raises(EmbeddingsUnavailable):
            Embeddings().embed("test")


def test_embed_batch_single_request_with_doc_prefix():
    """Should embed multiple documents in one request."""
    with patch("src.embeddings.requests.post", return_value=_mock_response(3, 0.0)) as mock_post:
        results = Embeddings().embed_batch(["one", "two", "three"])

        assert len(results) == 3
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["json"]["input"][0] == "search_document: one"
