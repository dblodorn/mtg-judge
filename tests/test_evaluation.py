"""Evaluation suite — measure retrieval quality on known MTG questions.

Integration tests (require Ollama + built index):
  pytest tests/test_evaluation.py -v -m integration

Run with: make eval
"""

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.query_router import QueryRouter
from src.retriever import HybridRetriever


@pytest.fixture
def eval_cases():
    return json.loads(Path("tests/fixtures/eval_cases.json").read_text())


@pytest.fixture(scope="module")
def retriever():
    config = load_config()
    return HybridRetriever(
        index_path=config["index_path"],
        k=config["k"],
        embed_model=config["embed_model"],
        ollama_host=config["ollama_host"],
        max_parent_chars=config["max_parent_chars"],
        max_context_chars=config["max_context_chars"],
    )


@pytest.fixture(scope="module")
def router():
    return QueryRouter(card_names_path=f"{load_config()['index_path'].rstrip('/')}/card_names.json")


def _parent_section(section: str) -> str:
    """Extract parent section number (e.g. '702.16a' → '702.16')."""
    import re
    m = re.match(r"(\d+\.\d+)", section)
    return m.group(1) if m else section


@pytest.mark.integration
def test_retrieval_returns_expected_sources(retriever, router, eval_cases):
    """For each eval case, verify retrieval returns expected sources."""
    for case in eval_cases:
        route = router.route(case["question"])
        results = retriever.retrieve(case["question"], route=route)

        retrieved_sources = {r.get("source") for r in results}
        for expected_source in case["expected_sources"]:
            assert expected_source in retrieved_sources, (
                f"Expected source '{expected_source}' for: {case['question']!r}\n"
                f"  Got sources: {retrieved_sources}"
            )


@pytest.mark.integration
def test_retrieval_returns_expected_sections(retriever, router, eval_cases):
    """For each eval case with expected_sections, verify coverage."""
    cases_with_sections = [c for c in eval_cases if "expected_sections_any" in c]
    if not cases_with_sections:
        pytest.skip("No cases with expected_sections_any")

    for case in cases_with_sections:
        route = router.route(case["question"])
        results = retriever.retrieve(case["question"], route=route)

        retrieved_sections = {
            _parent_section(r.get("section", r.get("parent_section", "")))
            for r in results
        }
        retrieved_sections.discard("")

        assert any(
            s == exp or s.startswith(exp + ".")
            for exp in case["expected_sections_any"]
            for s in retrieved_sections
        ), (
            f"None of {case['expected_sections_any']} found in {retrieved_sections}\n"
            f"  Question: {case['question']!r}"
        )


@pytest.mark.integration
def test_retrieval_returns_expected_keywords(retriever, router, eval_cases):
    """For each eval case, verify expected keywords appear in retrieved text."""
    for case in eval_cases:
        route = router.route(case["question"])
        results = retriever.retrieve(case["question"], route=route)

        all_text = " ".join(r.get("text", "") for r in results).lower()
        for kw in case.get("expected_keywords", []):
            assert kw.lower() in all_text, (
                f"Keyword '{kw}' not found in retrieved text for: {case['question']!r}"
            )


@pytest.mark.integration
def test_retrieval_returns_expected_cards(retriever, router, eval_cases):
    """For card-specific queries, verify the card appears in results."""
    card_cases = [c for c in eval_cases if "expected_cards" in c]
    if not card_cases:
        pytest.skip("No cases with expected_cards")

    for case in card_cases:
        route = router.route(case["question"])
        results = retriever.retrieve(case["question"], route=route)

        all_names = " ".join(
            r.get("name", "") + " " + r.get("text", "") for r in results
        )
        for card in case["expected_cards"]:
            assert card.lower() in all_names.lower(), (
                f"Card '{card}' not found in results for: {case['question']!r}"
            )


def test_eval_cases_loaded(eval_cases):
    """Sanity check — verify eval cases file is valid."""
    assert len(eval_cases) >= 10, f"Expected 10+ eval cases, got {len(eval_cases)}"
    for case in eval_cases:
        assert "question" in case
        assert "expected_sources" in case
