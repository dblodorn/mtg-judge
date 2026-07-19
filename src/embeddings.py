"""Embeddings client — thin wrapper for Ollama nomic-embed-text."""

import requests


class EmbeddingsUnavailable(Exception):
    """Raised when the Ollama embeddings endpoint is unreachable."""


class Embeddings:
    """Thin wrapper around Ollama's /api/embeddings endpoint.

    Uses nomic-embed-text (768-dim) by default. Runs entirely locally.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Embed a single string, return a 768-dim float vector."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.ConnectionError:
            raise EmbeddingsUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings. Ollama processes sequentially."""
        return [self.embed(t) for t in texts]
