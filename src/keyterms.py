"""Speech-recognition key terms — bias Moonshine toward MTG vocabulary and card names.

Live testing showed the STT model hearing "lore counter" as "lower counter" and
"Consign to Memory" as "consigned to money". Moonshine can bias its decoder toward
a list of terms (streaming models only); a long list costs accuracy on everything
else, so the list stays around 200 terms: core game words + the most-played cards
in the current format (from the MTGO metagame analysis, when there is one).
"""

import json
from pathlib import Path

from src.formats import FORMATS, display_name

MAX_TERMS = 200

# Game words the STT model is likely to mishear or split. Plain English words it
# already gets right ("attack", "draw") are left out.
BASE_TERMS = [
    "Comprehensive Rules", "lore counter", "chapter", "Saga", "priority", "the stack",
    "mana value", "converted mana cost", "state-based action", "replacement effect",
    "triggered ability", "activated ability", "static ability", "mana ability",
    "layer", "timestamp", "dependency", "sideboard", "mulligan", "decklist",
    "planeswalker", "loyalty", "battlefield", "graveyard", "exile", "library",
    "trample", "deathtouch", "lifelink", "first strike", "double strike", "hexproof",
    "indestructible", "ward", "flash", "haste", "vigilance", "menace", "reach",
    "protection", "cascade", "ninjutsu", "flashback", "storm", "affinity", "convoke",
    "delve", "split second", "suspend", "evoke", "prowess", "landfall", "energy",
    "+1/+1 counter", "-1/-1 counter", "token", "Construct", "Eldrazi", "Equipment",
    "Aura", "commander", "command zone", "Infraction Procedure Guide", "Game Loss",
    "Warning", "Missed Trigger", "Head Judge", "Regular REL", "Competitive REL",
]
BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def _speakable(card: str) -> list[str]:
    """Card name(s) as they'd be transcribed: faces split, commas dropped."""
    return [face.replace(",", "").strip() for face in card.split("//") if face.strip()]


def keyterms_for(game_format: str | None, metagame_root: str = "corpus/metagame") -> list[str]:
    """Base vocabulary + (format names while choosing one) + the format's top cards."""
    terms = list(BASE_TERMS)
    if game_format is None:
        terms += [display_name(f) for f in FORMATS] + ["EDH", "format"]
    else:
        cards_path = Path(metagame_root, game_format, "cards.json")
        if cards_path.exists():
            for row in json.loads(cards_path.read_text()):  # most-played first
                if row["name"] in BASIC_LANDS:
                    continue
                terms += _speakable(row["name"])
                if len(terms) >= MAX_TERMS:
                    break
    # Stable order, no duplicates, no commas (Moonshine's delimiter)
    return list(dict.fromkeys(t.replace(",", "") for t in terms))[:MAX_TERMS]
