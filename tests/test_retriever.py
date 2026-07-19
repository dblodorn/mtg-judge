"""Tests for the hybrid retriever.

Most tests mock the embeddings + ChromaDB to avoid requiring a live index.
Integration tests (marked @pytest.mark.integration) require Ollama + built index.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.retriever import HybridRetriever


def test_rrf_fusion_prioritizes_docs_in_both_retrievers():
    """RRF should rank docs appearing in both dense + sparse higher."""
    retriever = _make_mock_retriever()

    dense = [
        {"id": "a", "text": "alpha", "score": 0.9, "source": "cr"},
        {"id": "b", "text": "bravo", "score": 0.7, "source": "cr"},
    ]
    sparse = [
        {"id": "b", "text": "bravo", "score": 2.1, "source": "cr"},
        {"id": "c", "text": "charlie", "score": 1.5, "source": "cr"},
    ]

    fused = retriever._rrf_fuse(dense, sparse)

    # "b" appears in both retrievers → highest RRF score
    assert fused[0]["id"] == "b"
    assert fused[0]["fused_score"] > fused[1]["fused_score"]


def test_rrf_fusion_single_retriever():
    """Docs from only one retriever should still appear."""
    retriever = _make_mock_retriever()

    dense = [{"id": "x", "text": "x-ray", "score": 0.9, "source": "cr"}]
    sparse = [{"id": "y", "text": "yankee", "score": 2.0, "source": "cr"}]

    fused = retriever._rrf_fuse(dense, sparse)
    ids = [f["id"] for f in fused]
    assert "x" in ids
    assert "y" in ids


def test_parent_expansion_replaces_child_text():
    """Child chunk text should be replaced by parent document text."""
    retriever = _make_mock_retriever()
    retriever.parents = {
        "cr:702.16": {
            "id": "cr:702.16",
            "source": "cr",
            "section": "702.16",
            "text": "Full Protection section with all sub-rules...",
            "children": ["702.16a", "702.16b"],
        }
    }

    results = [
        {
            "id": "cr:702.16a",
            "parent_id": "cr:702.16",
            "text": "Protection is a static ability...",
            "fused_score": 0.5,
            "source": "cr",
        }
    ]

    expanded = retriever._expand_parents(results)

    assert expanded[0]["text"] == "Full Protection section with all sub-rules..."
    assert expanded[0]["child_id"] == "cr:702.16a"
    assert expanded[0]["id"] == "cr:702.16"


def test_parent_expansion_deduplicates():
    """Multiple children with the same parent should deduplicate to one parent."""
    retriever = _make_mock_retriever()
    retriever.parents = {
        "cr:702.16": {
            "id": "cr:702.16",
            "source": "cr",
            "section": "702.16",
            "text": "Full Protection section...",
            "children": ["702.16a", "702.16b"],
        }
    }

    results = [
        {"id": "cr:702.16a", "parent_id": "cr:702.16", "text": "sub-rule a", "fused_score": 0.3, "source": "cr"},
        {"id": "cr:702.16b", "parent_id": "cr:702.16", "text": "sub-rule b", "fused_score": 0.5, "source": "cr"},
    ]

    expanded = retriever._expand_parents(results)

    assert len(expanded) == 1  # deduplicated
    assert expanded[0]["fused_score"] == 0.5  # kept the higher-scoring one


def test_parent_expansion_keeps_self_parented_chunks():
    """Chunks where parent_id == id (e.g. Scryfall cards) should still work."""
    retriever = _make_mock_retriever()
    retriever.parents = {
        "scf:lightning-bolt": {
            "id": "scf:lightning-bolt",
            "source": "scryfall",
            "text": "Lightning Bolt deals 3 damage...",
            "children": ["Lightning Bolt"],
        }
    }

    results = [
        {
            "id": "scf:lightning-bolt",
            "parent_id": "scf:lightning-bolt",
            "text": "Lightning Bolt deals 3 damage...",
            "fused_score": 0.8,
            "source": "scryfall",
        }
    ]

    expanded = retriever._expand_parents(results)

    assert len(expanded) == 1
    assert expanded[0]["text"] == "Lightning Bolt deals 3 damage..."


def test_mmr_returns_k_results():
    """MMR should return exactly k results from a larger candidate set."""
    retriever = _make_mock_retriever(k=3)

    candidates = [
        {"id": f"c{i}", "text": f"chunk {i}", "fused_score": 0.5} for i in range(10)
    ]

    with patch.object(retriever.emb, "embed", return_value=[0.1] * 768):
        result = retriever._mmr("query", candidates, k=3)

    assert len(result) == 3


def _make_mock_retriever(k: int = 8) -> HybridRetriever:
    """Create a HybridRetriever without loading the actual index."""
    with patch("src.retriever.chromadb.PersistentClient"), \
         patch("src.retriever.bm25s.BM25"):
        r = HybridRetriever.__new__(HybridRetriever)
        r.emb = MagicMock()
        r.emb.embed.return_value = [0.1] * 768
        r.k = k
        r.rrf_k = 60
        r.mmr_lambda = 0.5
        r.parents = {}
        return r
