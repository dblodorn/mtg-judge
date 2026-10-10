"""Tests for the hybrid retriever.

Most tests mock the embeddings + ChromaDB to avoid requiring a live index.
Integration tests (marked @pytest.mark.integration) require Ollama + built index.
"""

from unittest.mock import MagicMock, patch

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
        {"id": f"c{i}", "text": f"chunk {i}", "fused_score": 1.0 - i / 10} for i in range(10)
    ]

    retriever.col.get.return_value = {
        "ids": [c["id"] for c in candidates],
        "embeddings": [[float(i == j) for j in range(16)] for i in range(10)],
    }
    result = retriever._mmr("query", candidates, k=3, query_emb=[1.0] + [0.0] * 15)

    assert len(result) == 3
    assert result[0]["id"] == "c0"  # highest fused score first
    retriever.emb.embed.assert_not_called()  # uses stored embeddings, no re-embedding


def test_mmr_falls_back_to_fused_order_without_embeddings():
    retriever = _make_mock_retriever(k=2)
    retriever.emb.embed.side_effect = ConnectionError("ollama down")
    candidates = [{"id": f"c{i}", "text": "t", "fused_score": i / 10} for i in range(5)]

    result = retriever._mmr("query", candidates, k=2)

    assert [c["id"] for c in result] == ["c4", "c3"]


def test_trim_parent_keeps_heading_and_window():
    retriever = _make_mock_retriever()
    retriever.max_parent_chars = 200
    lines = ["702.19 Trample"] + [f"702.19{chr(97 + i)} " + "x" * 40 for i in range(10)]

    trimmed = retriever._trim_parent("\n".join(lines), "702.19f ")

    assert trimmed.startswith("702.19 Trample")
    assert "702.19f" in trimmed
    assert "702.19a" not in trimmed


def test_direct_lookups_pin_rule_parent_and_cards():
    retriever = _make_mock_retriever()
    retriever.parents = {
        "cr:702.16": {"id": "cr:702.16", "source": "cr", "section": "702.16", "text": "702.16 Protection\n702.16a ..."},
        "scryfall:abc": {"id": "scryfall:abc", "source": "scryfall", "name": "Lightning Bolt", "text": "Lightning Bolt — Instant"},
    }
    retriever.card_ids = {"lightning bolt": "scryfall:abc"}

    pinned = retriever._direct_lookups({"rule_numbers": ["702.16a"], "mentioned_cards": ["Lightning Bolt"]})

    assert [d["id"] for d in pinned] == ["cr:702.16", "scryfall:abc"]
    assert pinned[0]["child_id"] == "cr:702.16a"


def test_cap_context_respects_budget():
    retriever = _make_mock_retriever(k=8)
    retriever.max_context_chars = 100
    docs = [{"id": str(i), "text": "x" * 60} for i in range(5)]

    assert len(retriever._cap_context(docs)) == 1


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
        r.col = MagicMock()
        return r


def test_pinned_card_keeps_most_relevant_rulings():
    retriever = _make_mock_retriever()
    retriever.rulings_per_card = 1
    card = {"id": "scryfall:x", "text": "Card — Land\nRulings:\n- (2021) about mana\n- (2021) about blood moon\n- (2021) about tokens"}
    retriever.col.get.return_value = {
        "ids": ["scryfall:x:r0", "scryfall:x:r1", "scryfall:x:r2"],
        "embeddings": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
    }

    text = retriever._card_with_rulings(card, query_emb=[0.0, 1.0])

    assert text == "Card — Land\nRulings:\n- (2021) about blood moon"


def test_expand_card_names_swaps_name_for_card_description():
    retriever = _make_mock_retriever()
    retriever.parents = {"scryfall:u": {
        "id": "scryfall:u",
        "type": "Enchantment Land — Urza's Saga",
        "text": "Urza's Saga — Enchantment Land — Urza's Saga\nIII — Search your library.\nRulings:\n- (2021) keeps abilities",
    }}
    retriever.card_ids = {"urza's saga": "scryfall:u"}

    expanded = retriever._expand_card_names("What happens to my Urza's Saga?", {"mentioned_cards": ["Urza's Saga"]})

    assert expanded == "What happens to my (Enchantment Land — Urza's Saga: III — Search your library.)?"


def test_source_of_chunk_ids():
    assert HybridRetriever._source_of("cr:702.16a") == "cr"
    assert HybridRetriever._source_of("ipg:2.1:3") == "ipg"
    assert HybridRetriever._source_of("scryfall:abc-123") == "scryfall"
    assert HybridRetriever._source_of("scryfall:abc-123:r2") == "ruling"


def test_mmr_skips_near_duplicates():
    """A near-duplicate of the top doc loses to a less relevant but different one."""
    retriever = _make_mock_retriever()
    candidates = [{"id": c, "text": c, "fused_score": f} for c, f in [("a", 1.0), ("a2", 0.9), ("b", 0.5), ("z", 0.0)]]
    retriever.col.get.return_value = {
        "ids": ["a", "a2", "b", "z"],
        "embeddings": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.7, 0.7]],
    }

    result = retriever._mmr("q", candidates, k=2, query_emb=[1.0, 0.0])

    assert [c["id"] for c in result] == ["a", "b"]


def test_pinned_card_gets_legality_for_current_format():
    retriever = _make_mock_retriever()
    retriever.parents = {"scryfall:u": {"id": "scryfall:u", "source": "scryfall", "name": "Urza's Saga",
                                        "text": "Urza's Saga — Enchantment Land",
                                        "legalities": {"modern": "legal", "pauper": "not_legal"}}}
    retriever.card_ids = {"urza's saga": "scryfall:u"}

    modern = retriever._direct_lookups({"mentioned_cards": ["Urza's Saga"], "format": "modern"})
    pauper = retriever._direct_lookups({"mentioned_cards": ["Urza's Saga"], "format": "pauper"})
    none = retriever._direct_lookups({"mentioned_cards": ["Urza's Saga"]})

    assert modern[0]["text"].endswith("Modern legality: legal")
    assert pauper[0]["text"].endswith("Pauper legality: not legal")
    assert "legality" not in none[0]["text"]
