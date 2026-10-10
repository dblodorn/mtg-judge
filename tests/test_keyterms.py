"""Tests for speech-recognition key terms."""

import json

from src.keyterms import BASE_TERMS, MAX_TERMS, keyterms_for


def test_format_cards_follow_base_terms(tmp_path):
    (tmp_path / "modern").mkdir()
    (tmp_path / "modern" / "cards.json").write_text(json.dumps([
        {"name": "Island"}, {"name": "Urza's Saga"}, {"name": "Ragavan, Nimble Pilferer"},
        {"name": "Fire // Ice"}, {"name": "Urza's Saga"},
    ]))

    terms = keyterms_for("modern", metagame_root=str(tmp_path))

    assert terms[: len(BASE_TERMS)] == BASE_TERMS
    tail = terms[len(BASE_TERMS):]
    assert tail == ["Urza's Saga", "Ragavan Nimble Pilferer", "Fire", "Ice"]  # no basics, commas, dupes


def test_capped_and_comma_free(tmp_path):
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "cards.json").write_text(json.dumps([{"name": f"Card {i}, the Test"} for i in range(500)]))

    terms = keyterms_for("legacy", metagame_root=str(tmp_path))

    assert len(terms) == MAX_TERMS
    assert not any("," in t for t in terms)


def test_no_format_includes_format_names(tmp_path):
    terms = keyterms_for(None, metagame_root=str(tmp_path))
    assert {"Modern", "Commander", "EDH"} <= set(terms)
