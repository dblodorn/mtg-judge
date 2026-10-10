"""LLM client — Ollama chat with Pydantic structured output."""

import json
import re
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

from src.formats import display_name

RULE_REF_RE = re.compile(r"\d{1,4}\.\d+[a-z]{0,2}")


class JudgeRuling(BaseModel):
    """Structured output from the judge LLM."""
    answer: str = Field(description="The ruling, 2-4 sentences with citation")
    confidence: Literal["high", "medium", "low"] = Field(description="high, medium, or low")
    citations: list[str] = Field(description="Rule numbers or card names cited")


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c.get('source', '').upper()} {c.get('section', c.get('name', ''))}] {c['text']}"
        for c in chunks
    )


def verify_citations(ruling: JudgeRuling, chunks: list[dict]) -> JudgeRuling:
    """Drop citations that don't appear in the retrieved context.

    A rule-number citation counts if that number appears anywhere in the context
    (parent text includes its sub-rules); a name citation counts if the name does.
    Any dropped citation downgrades confidence to "low".
    """
    haystack = "\n".join(
        f"{c.get('section', '')} {c.get('name', '')} {c.get('text', '')}" for c in chunks
    ).lower()

    kept = []
    for cite in ruling.citations:
        refs = RULE_REF_RE.findall(cite)
        if refs:
            ok = all(ref.lower() in haystack for ref in refs)
        else:
            ok = cite.strip().lower() in haystack
        if ok:
            kept.append(cite)

    if len(kept) == len(ruling.citations):
        return ruling
    return ruling.model_copy(update={"citations": kept, "confidence": "low"})


class JudgeLLM:
    """Wraps Ollama chat for judge-style Q&A with structured output."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str | None = None,
        system_prompt_path: str = "config/prompts/judge_system.md",
        num_ctx: int = 4096,
        temperature: float = 0.2,
        think: bool | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.think = think  # reasoning models (qwen3…): None = model default
        self.client = ollama.Client(host=host)
        self.system_prompt = Path(system_prompt_path).read_text()
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        if max_tokens:
            # Hard cap on generated tokens (reasoning included): a runaway generation
            # becomes a low-confidence fallback instead of minutes of silence.
            self.options["num_predict"] = max_tokens
        self._preload()

    def _preload(self):
        """Ensure the model is pulled and warm-loaded."""
        listed = self.client.list()
        model_names = [m.model for m in (listed.models or [])]
        if self.model not in model_names:
            print(f"Pulling Ollama model {self.model}...")
            self.client.pull(self.model)
        # Warm-load and keep resident between questions
        self.client.generate(model=self.model, prompt="", keep_alive=-1)

    def answer(self, query: str, context_chunks: list[dict], game_format: str | None = None) -> JudgeRuling:
        """Generate a structured ruling from retrieved context."""
        messages = [
            # System prompt first and unchanged so Ollama can reuse its KV cache.
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{format_context(context_chunks)}\n\n"
                + (f"Format being played: {display_name(game_format)}\n" if game_format else "")
                + f"Question: {query}\n\nRespond as JSON:",
            },
        ]
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=JudgeRuling.model_json_schema(),
                options=self.options,
                keep_alive=-1,
                **({"think": self.think} if self.think is not None else {}),
            )
        except ollama.ResponseError as e:
            # e.g. "token repeat limit reached" — a degenerate generation, not a dead server
            print(f"⚠ LLM error: {e}")
            self.last_stats = {"prompt_tokens": 0, "output_tokens": 0}
            return JudgeRuling(answer="Sorry, I couldn't work out a ruling for that.", confidence="low", citations=[])
        # Token counts — what Pi latency actually depends on (prompt eval dominates on CPU)
        self.last_stats = {
            "prompt_tokens": response.get("prompt_eval_count") or 0,
            "output_tokens": response.get("eval_count") or 0,
        }
        raw_text = response["message"]["content"]
        return verify_citations(self._parse(raw_text), context_chunks)

    @staticmethod
    def _parse(raw_text: str) -> JudgeRuling:
        try:
            # Structured output should be bare JSON, but tolerate ```json wrappers
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return JudgeRuling(**json.loads(raw_text[json_start:json_end]))
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

        # Fallback: treat as plain text
        return JudgeRuling(answer=raw_text, confidence="low", citations=[])
