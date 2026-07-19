"""Tests for the query router."""

import json
from pathlib import Path

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
    assert result["sources"] is None  # search all


def test_general_rules_question():
    """Non-specific questions route to general (search all sources)."""
    router = QueryRouter(card_names_path=None)
    result = router.route("How does priority work?")

    assert result["type"] == "general"
    assert result["sources"] is None


def test_card_name_case_insensitive(tmp_path):
    """Card matching should be case-insensitive."""
    card_names_file = tmp_path / "card_names.json"
    card_names_file.write_text(json.dumps(["Lightning Bolt"]))

    router = QueryRouter(card_names_path=str(card_names_file))
    result = router.route("does lightning bolt deal 3 damage?")

    assert result["type"] == "card_interaction"
    assert "Lightning Bolt" in result["mentioned_cards"]
