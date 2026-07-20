"""Judge pipeline — text query → retrieve → LLM → structured response."""

import sys

sys.path.insert(0, ".")

from src.retriever import HybridRetriever
from src.query_router import QueryRouter
from src.llm import JudgeLLM


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
        self.retriever = HybridRetriever(index_path=index_path, k=8)
        self.router = QueryRouter()
        self.llm = JudgeLLM(model="llama3.1:8b")

    def ask(self, question: str) -> dict:
        """Process a text question → return ruling + retrieved chunks."""
        route = self.router.route(question)
        chunks = self.retriever.retrieve(question, sources=route.get("sources"))
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
