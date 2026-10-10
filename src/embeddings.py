"""Embeddings client — thin wrapper for Ollama nomic-embed-text."""

import requests

# nomic-embed-text is trained with task prefixes; omitting them degrades retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingsUnavailable(Exception):
    """Raised when the Ollama embeddings endpoint is unreachable."""


class Embeddings:
    """Thin wrapper around Ollama's /api/embed endpoint.

    Uses nomic-embed-text (768-dim) by default. Runs entirely locally.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _post(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs, "keep_alive": -1},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except requests.ConnectionError:
            raise EmbeddingsUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from None

    def embed(self, text: str) -> list[float]:
        """Embed a search query, return a 768-dim float vector."""
        return self._post([QUERY_PREFIX + text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing in one request."""
        return self._post([DOC_PREFIX + t for t in texts])
