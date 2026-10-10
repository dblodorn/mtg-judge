"""Tests for spoken format parsing."""

import pytest

from src.formats import parse_format, parse_switch


@pytest.mark.parametrize("text,expected", [
    ("Modern.", "modern"),
    ("we are playing EDH", "commander"),
    ("pre modern please", "premodern"),  # longest alias wins over "modern"
    ("um, legacy I think", "legacy"),
    ("old school", "oldschool"),
    ("no idea", None),
])
def test_parse_format(text, expected):
    assert parse_format(text) == expected


def test_parse_switch_needs_a_switch_phrase():
    assert parse_switch("switch to Pioneer") == "pioneer"
    assert parse_switch("does it work in modern") is None
