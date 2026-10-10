"""Judge pipeline — text query → retrieve → LLM → structured response."""

import sys

sys.path.insert(0, ".")

from src.app import build_pipeline
from src.config import load_config


class JudgePipeline:
    """Wires retriever + router + LLM into a single ask() call."""

    _instance = None

    @classmethod
    def get(cls, index_path: str = "demo/index/"):
        """Singleton — avoids reloading the index on every request."""
        if cls._instance is None:
            cls._instance = cls(index_path)
        return cls._instance

    def __init__(self, index_path: str = "demo/index/"):
        config = {**load_config(), "index_path": index_path}
        self.retriever, self.router, self.llm = build_pipeline(config)

    def ask(self, question: str) -> dict:
        """Process a text question → return ruling + retrieved chunks."""
        route = self.router.route(question)
        chunks = self.retriever.retrieve(question, route=route)
        ruling = self.llm.answer(question, chunks)
        return {
            "answer": ruling.answer,
            "confidence": ruling.confidence,
            "citations": ruling.citations,
            "query_type": route["type"],
            "retrieved_chunks": [
                {
                    "id": c.get("id", "?"),
                    "source": c.get("source", "?"),
                    "section": c.get("section", c.get("parent_section", "")),
                    "text": c.get("text", "")[:300],
                }
                for c in chunks[:5]
            ],
        }
