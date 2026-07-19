"""LLM client — Ollama chat with Pydantic structured output."""

import json
import sys
from pathlib import Path

import ollama
from pydantic import BaseModel, Field


class JudgeRuling(BaseModel):
    """Structured output from the judge LLM."""
    answer: str = Field(description="The ruling, 2-4 sentences with citation")
    confidence: str = Field(description="high, medium, or low")
    citations: list[str] = Field(description="Rule numbers or card names cited")


class JudgeLLM:
    """Wraps Ollama chat for judge-style Q&A with structured output."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        system_prompt_path: str = "config/prompts/judge_system.md",
    ):
        self.model = model
        self.system_prompt = Path(system_prompt_path).read_text()
        self._preload()

    def _preload(self):
        """Ensure the model is pulled and warm-loaded."""
        listed = ollama.list()
        model_names = [m.model for m in (listed.models or [])]
        if self.model not in model_names:
            print(f"Pulling Ollama model {self.model}...")
            ollama.pull(self.model)
        # Warm-load
        ollama.generate(model=self.model, prompt="", keep_alive="10m")

    def answer(self, query: str, context_chunks: list[dict]) -> JudgeRuling:
        """Generate a structured ruling from retrieved context."""
        ctx = "\n\n".join(
            f"[{c.get('source', '').upper()} {c.get('section', c.get('name', ''))}] {c['text']}"
            for c in context_chunks
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{ctx}\n\nQuestion: {query}\n\nRespond as JSON:",
            },
        ]
        response = ollama.chat(model=self.model, messages=messages)
        raw_text = response["message"]["content"]

        try:
            # Try to extract JSON from the response (LLMs sometimes wrap in ```json)
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                raw = json.loads(raw_text[json_start:json_end])
                return JudgeRuling(**raw)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat as plain text
        return JudgeRuling(answer=raw_text, confidence="low", citations=[])
