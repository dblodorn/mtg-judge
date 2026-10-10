"""Tests for the query router."""

import json

from src.query_router import QueryRouter


def test_rule_number_detection():
    """Queries with explicit rule numbers route to CR."""
    router = QueryRouter(card_names_path=None)
    result = router.route("What does rule 702.16a say?")

    assert result["type"] == "rule_lookup"
    assert result["sources"] == ["cr"]
    assert "702.16a" in result["rule_numbers"]


def test_multi_part_rule_number():
    """Multiple rule numbers in one query."""
    router = QueryRouter(card_names_path=None)
    result = router.route("How do rules 702.16a and 611.2b interact?")

    assert result["type"] == "rule_lookup"
    assert "702.16a" in result["rule_numbers"]
    assert "611.2b" in result["rule_numbers"]


def test_tournament_keyword_detection():
    """Tournament-related keywords route to MTR/IPG/JAR."""
    router = QueryRouter(card_names_path=None)
    result = router.route("What's the penalty for slow play?")

    assert result["type"] == "tournament"
    assert "ipg" in result["sources"]
    assert "mtr" in result["sources"]


def test_game_loss_penalty():
    router = QueryRouter(card_names_path=None)
    result = router.route("When is a game loss issued?")

    assert result["type"] == "tournament"


def test_card_name_detection(tmp_path):
    """Queries mentioning a card name route to card_interaction."""
    # Create a small card names file
    card_names_file = tmp_path / "card_names.json"
    card_names_file.write_text(json.dumps(["Lightning Bolt", "Counterspell", "Sol Ring"]))

    router = QueryRouter(card_names_path=str(card_names_file))
    result = router.route("Does Lightning Bolt target a player?")

    assert result["type"] == "card_interaction"
    assert "Lightning Bolt" in result["mentioned_cards"]
    assert "scryfall" not in result["sources"]  # card is pinned, not searched


def test_general_rules_question():
    """Non-specific questions route to general (search all sources)."""
    router = QueryRouter(card_names_path=None)
    result = router.route("How does priority work?")

    assert result["type"] == "general"
    assert "cr" in result["sources"] and "scryfall" not in result["sources"]


def test_card_name_case_insensitive(tmp_path):
    """Card matching should be case-insensitive."""
    card_names_file = tmp_path / "card_names.json"
    card_names_file.write_text(json.dumps(["Lightning Bolt"]))

    router = QueryRouter(card_names_path=str(card_names_file))
    result = router.route("does lightning bolt deal 3 damage?")

    assert result["type"] == "card_interaction"
    assert "Lightning Bolt" in result["mentioned_cards"]


def _router(tmp_path, names, ambiguous=()):
    path = tmp_path / "card_names.json"
    path.write_text(json.dumps({"names": names, "ambiguous": list(ambiguous)}))
    return QueryRouter(card_names_path=str(path))


def test_card_match_requires_whole_words(tmp_path):
    """'Opt' must not match inside 'option'."""
    router = _router(tmp_path, ["Opt"])

    assert router.route("Is there an option to skip combat?")["type"] == "general"


def test_card_match_prefers_longest_name(tmp_path):
    router = _router(tmp_path, ["Lightning", "Lightning Bolt"])

    assert router.find_cards("Can Lightning Bolt hit a planeswalker?") == ["Lightning Bolt"]


def test_ambiguous_single_word_needs_capitalization(tmp_path):
    router = _router(tmp_path, ["Fight", "Island"], ambiguous=["Fight", "Island"])

    assert router.find_cards("can my creature fight an island?") == []
    assert router.find_cards("Does Fight count as damage from a spell?") == ["Fight"]
    assert router.find_cards("Island walk: how does it work?") == []  # sentence-initial capital


def test_ambiguous_face_name_not_matched_at_sentence_start(tmp_path):
    router = _router(tmp_path, ["Who // What // When // Where // Why"], ambiguous=["What", "Who", "When", "Where", "Why"])

    assert router.find_cards("What is the stack?") == []
    assert router.find_cards("Blood Moon is out. What happens?") == []


def test_card_match_ignores_punctuation_and_faces(tmp_path):
    router = _router(tmp_path, ["Jace, the Mind Sculptor", "Delver of Secrets // Insectile Aberration"])

    found = router.find_cards("jace the mind sculptor and delver of secrets")

    assert found == ["Jace, the Mind Sculptor", "Delver of Secrets // Insectile Aberration"]


def test_full_card_name_beats_other_cards_face(tmp_path):
    router = _router(tmp_path, ["Grave Researcher // Reanimate", "Reanimate"])

    assert router.find_cards("if I Reanimate it") == ["Reanimate"]


def test_possessive_after_card_name(tmp_path):
    router = _router(tmp_path, ["Urza's Saga"])

    assert router.find_cards("Can Urza's Saga's chapter three find it?") == ["Urza's Saga"]
