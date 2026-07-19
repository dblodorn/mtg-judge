"""Tests for the embeddings client."""

from unittest.mock import patch, MagicMock

import pytest

from src.embeddings import Embeddings, EmbeddingsUnavailable


def test_embed_returns_vector():
    """embed() should return a list of floats."""
    with patch("src.embeddings.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1] * 768}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        emb = Embeddings()
        vec = emb.embed("protection from red")

        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)
        mock_post.assert_called_once()


def test_embed_calls_correct_endpoint():
    """Should POST to /api/embeddings with model and prompt."""
    with patch("src.embeddings.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.0] * 768}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        emb = Embeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        emb.embed("lightning bolt")

        args, kwargs = mock_post.call_args
        assert "api/embeddings" in args[0]
        assert kwargs["json"]["model"] == "nomic-embed-text"
        assert kwargs["json"]["prompt"] == "lightning bolt"


def test_embed_raises_on_connection_error():
    """Should raise EmbeddingsUnavailable when Ollama is not running."""
    with patch("src.embeddings.requests.post") as mock_post:
        import requests

        mock_post.side_effect = requests.ConnectionError()

        emb = Embeddings()
        with pytest.raises(EmbeddingsUnavailable):
            emb.embed("test")


def test_embed_batch():
    """Should embed multiple texts."""
    with patch.object(Embeddings, "embed") as mock_embed:
        mock_embed.return_value = [0.0] * 768
        emb = Embeddings()
        results = emb.embed_batch(["one", "two", "three"])

        assert len(results) == 3
        assert mock_embed.call_count == 3
