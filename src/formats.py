"""Game formats — names, spoken aliases, and parsing a format out of speech."""

import re

# Scryfall legality keys → spoken aliases (lowercase, as speech-to-text writes them)
FORMATS: dict[str, list[str]] = {
    "standard": ["standard"],
    "pioneer": ["pioneer"],
    "modern": ["modern"],
    "legacy": ["legacy"],
    "vintage": ["vintage"],
    "pauper": ["pauper"],
    "commander": ["commander", "edh", "e d h", "elder dragon highlander"],
    "premodern": ["premodern", "pre modern", "pre-modern"],
    "oldschool": ["old school", "oldschool"],
}

DISPLAY = {"commander": "Commander", "oldschool": "Old School", "premodern": "Premodern"}

# "switch to legacy", "we're playing commander", "format is modern", "change format to pioneer"
SWITCH_RE = re.compile(
    r"\b(switch(?:ing)? to|change (?:the )?format to|(?:we're|we are|let's|lets) play(?:ing)?|"
    r"(?:the )?format is|set (?:the )?format to|playing)\b",
    re.IGNORECASE,
)


def display_name(fmt: str) -> str:
    return DISPLAY.get(fmt, fmt.title())


def parse_format(text: str) -> str | None:
    """The format named in `text`, or None. Longest alias wins ("pre modern" over "modern")."""
    lowered = " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "
    hits = [
        (len(alias), fmt)
        for fmt, aliases in FORMATS.items()
        for alias in aliases
        if f" {alias} " in lowered
    ]
    return max(hits)[1] if hits else None


def parse_switch(text: str) -> str | None:
    """A format-change command ("switch to Legacy"), or None if `text` isn't one."""
    if not SWITCH_RE.search(text):
        return None
    return parse_format(text)
