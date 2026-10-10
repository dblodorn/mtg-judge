"""Tests for corpus parsing (CR structure, tournament docs, Scryfall cards)."""

import gzip
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("parse_corpus", Path("scripts/parse_corpus.py"))
parse_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse_corpus)

CR_SAMPLE = """Magic: The Gathering Comprehensive Rules

Contents
702. Keyword Abilities
Glossary
Credits

702. Keyword Abilities

702.1. Most abilities describe exactly what they do.

702.16. Protection

702.16a Protection is a static ability, written "Protection from [quality]."

702.16b A permanent with protection can't be targeted by spells with the stated quality.
Example: A creature with protection from red can't be the target of Lightning Bolt.

704.5a If a player has 0 or less life, that player loses the game.

704.5aa If a player has no speed, that player's speed becomes 1.

Glossary

Protection
A keyword ability that provides a range of benefits. See rule 702.16, "Protection."

Credits
"""


def _by_id(items):
    return {i["id"]: i for i in items}


def test_cr_headings_rules_examples_and_glossary():
    children, parents = parse_corpus.parse_cr(CR_SAMPLE)
    ch, pa = _by_id(children), _by_id(parents)

    # "702.16. Protection" is a title, not a rule
    assert pa["cr:702.16"]["title"] == "Protection"
    assert pa["cr:702.16"]["text"].startswith("702.16 Protection\n702.16a ")
    assert "cr:702.16" not in ch

    # Un-lettered rule with trailing period is a rule in its own group
    assert ch["cr:702.1"]["parent_id"] == "cr:702.1"

    # Examples fold into their rule; children carry the group title
    assert "Example: A creature with protection" in ch["cr:702.16b"]["text"]
    assert ch["cr:702.16b"]["text"].startswith("702.16b Protection: ")
    assert ch["cr:702.16b"]["anchor"] == "702.16b "

    # Double-letter sub-rules don't overwrite single-letter ones
    assert "0 or less life" in ch["cr:704.5a"]["text"]
    assert "speed becomes 1" in ch["cr:704.5aa"]["text"]

    # Glossary entries, but not the table of contents
    assert ch["cr:glossary:protection"]["text"].startswith("Protection: A keyword ability")
    assert len([c for c in children if c["id"].startswith("cr:702.")]) == 3


def test_sectioned_docs_parent_per_section(tmp_path):
    path = tmp_path / "ipg.jsonl"
    path.write_text(json.dumps({
        "section": "3.4", "title": "3.4 Tournament Error — Decklist Problem",
        "paragraphs": ["The decklist is illegal.", "Penalty: Game Loss"],
    }) + "\n")

    children, parents = parse_corpus.parse_sectioned(path, "ipg")

    assert parents[0]["id"] == "ipg:3.4"
    assert parents[0]["text"].split("\n")[0] == "IPG 3.4 Tournament Error — Decklist Problem"
    assert [c["id"] for c in children] == ["ipg:3.4:0", "ipg:3.4:1"]
    assert children[1]["text"] == "IPG 3.4 Tournament Error — Decklist Problem: Penalty: Game Loss"


def test_scryfall_faces_cmc_and_oracle_ids(tmp_path):
    paper = {"vintage": "legal", "commander": "legal"}
    cards = [
        {"oracle_id": "o1", "name": "Lightning Bolt", "layout": "normal", "cmc": 1.0, "mana_cost": "{R}",
         "type_line": "Instant", "oracle_text": "Lightning Bolt deals 3 damage to any target.", "legalities": paper},
        {"oracle_id": "o2", "name": "Fire // Ice", "layout": "split", "cmc": 4.0, "type_line": "Instant // Instant",
         "legalities": paper, "card_faces": [
             {"name": "Fire", "mana_cost": "{1}{R}", "type_line": "Instant", "oracle_text": "Fire deals 2 damage."},
             {"name": "Ice", "mana_cost": "{1}{U}", "type_line": "Instant", "oracle_text": "Tap target permanent."},
         ]},
        # Arena-only design: not legal in paper formats
        {"oracle_id": "o3", "name": "A-Rebalanced", "layout": "normal", "digital": True,
         "legalities": {"vintage": "not_legal", "commander": "not_legal"}},
        {"oracle_id": "o4", "name": "Goblin", "layout": "token", "legalities": paper},
        # Real paper card whose oracle printing happens to be MTGO-only (e.g. Humility → TPR)
        {"oracle_id": "o5", "name": "Humility", "layout": "normal", "digital": True, "cmc": 4.0,
         "type_line": "Enchantment", "oracle_text": "All creatures lose all abilities and have base power and toughness 1/1.",
         "legalities": {"vintage": "legal", "commander": "legal"}},
    ]
    path = tmp_path / "scryfall_oracle.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(c) + "\n" for c in cards)

    rulings = {"o1": [{"oracle_id": "o1", "published_at": "2020-01-01", "comment": "It can target a planeswalker."}]}
    children, parents, names = parse_corpus.parse_scryfall_streaming(path, rulings)
    ch, pa = _by_id(children), _by_id(parents)

    assert names == ["Lightning Bolt", "Fire // Ice", "Humility"]
    assert ch["scryfall:o1"]["mv"] == 1.0
    assert "Fire deals 2 damage." in ch["scryfall:o2"]["text"]
    assert "Tap target permanent." in ch["scryfall:o2"]["text"]

    # Rulings: own child (source "ruling") under the card, and appended to the card's parent text
    assert ch["scryfall:o1:r0"]["source"] == "ruling"
    assert ch["scryfall:o1:r0"]["parent_id"] == "scryfall:o1"
    assert "Rulings:\n- (2020-01-01) It can target a planeswalker." in pa["scryfall:o1"]["text"]


def test_ambiguous_card_names_uses_rules_vocab():
    names = ["Fight", "Counterspell", "Lightning Bolt"]
    assert "Fight" in parse_corpus.ambiguous_card_names(names, "701.14 Fight")
    assert "Lightning Bolt" not in parse_corpus.ambiguous_card_names(names, "")
