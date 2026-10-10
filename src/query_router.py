"""Query router — detect query type and route to appropriate retrieval strategy."""

import json
import re
from pathlib import Path

RULE_NUMBER_RE = re.compile(r"\b(\d{3}\.\d+[a-z]{0,2})\b")
WORD_RE = re.compile(r"[a-z0-9]+")
TOURNAMENT_KEYWORDS = {
    "tournament",
    "penalty",
    "infraction",
    "slow play",
    "warning",
    "game loss",
    "match loss",
    "disqualif",
    "tiebreaker",
    "decklist",
    "sideboard limit",
}
MAX_CARDS = 5
# Card text and the 77k card rulings swamp the CR in search results, so rules routes
# search rules documents only; mentioned cards (with their most relevant rulings) are
# pinned by the retriever instead.
RULES_SOURCES = ["cr", "mtr", "ipg", "jar"]


def _norm_words(text: str) -> list[str]:
    """'Jace, the Mind-Sculptor' → ['jace', 'the', 'mind', 'sculptor']"""
    return WORD_RE.findall(text.lower().replace("'", "").replace("’", ""))


class QueryRouter:
    """Analyzes a user query and returns routing hints for the retriever.

    Detection types:
    - rule_lookup: explicit rule number (e.g. "What does 702.16a say?")
    - tournament: tournament procedure / penalty questions
    - card_interaction: mentions a specific card name
    - general: general rules question (search all sources)

    Every route also carries `mentioned_cards` so the retriever can pin them.
    """

    def __init__(self, card_names_path: str | None = "index/card_names.json"):
        """Load the card name index for whole-word matching.

        The file is either a list of names, or {"names": [...], "ambiguous": [...]}
        where ambiguous names (single words that are also ordinary words, like
        "Opt" or "Island", including single-word faces like "What") only match
        when capitalized somewhere other than the start of the query.
        """
        self.card_names: list[str] = []
        self.ambiguous: set[str] = set()
        if card_names_path and Path(card_names_path).exists():
            data = json.loads(Path(card_names_path).read_text())
            if isinstance(data, dict):
                self.card_names = data.get("names", [])
                self.ambiguous = {n.lower() for n in data.get("ambiguous", [])}
            else:
                self.card_names = data

        # Normalized word tuple → canonical name. Split/DFC cards also match each face,
        # but a card's full name wins over another card's face ("Reanimate" vs
        # "Grave Researcher // Reanimate").
        self._lookup: dict[tuple[str, ...], str] = {}
        for name in self.card_names:
            if words := tuple(_norm_words(name)):
                self._lookup.setdefault(words, name)
        for name in self.card_names:
            for face in name.split(" // "):
                if words := tuple(_norm_words(face)):
                    self._lookup.setdefault(words, name)
        self._max_len = max((len(k) for k in self._lookup), default=0)

    def find_cards(self, query: str) -> list[str]:
        """Longest-first, whole-word, non-overlapping card name matches."""
        if not self._lookup:
            return []
        tokens = list(re.finditer(r"[A-Za-z0-9'’]+", query))
        words = ["".join(_norm_words(t.group())) for t in tokens]
        # "Urza's Saga's chapter" — a possessive may end a name ("saga's" → "saga")
        bare = [re.sub(r"['’]s$", "", t.group()).lower() if re.search(r"['’]s$", t.group()) else w
                for t, w in zip(tokens, words, strict=True)]

        found: list[str] = []
        i = 0
        while i < len(words):
            for n in range(min(self._max_len, len(words) - i), 0, -1):
                name = self._lookup.get(tuple(words[i : i + n])) or self._lookup.get(
                    (*words[i : i + n - 1], bare[i + n - 1])
                )
                before = query[: tokens[i].start()].rstrip()
                sentence_start = not before or before[-1] in ".?!:"
                if name and self._accept(words[i], tokens[i].group(), n, first=sentence_start):
                    if name not in found:
                        found.append(name)
                    i += n
                    break
            else:
                i += 1
        return found[:MAX_CARDS]

    def _accept(self, word: str, surface: str, n_words: int, first: bool) -> bool:
        """Multi-word names always match; ambiguous single words need a capital mid-sentence.

        `first` = the word starts a sentence, where a capital letter means nothing.
        """
        if n_words > 1 or word not in self.ambiguous:
            return True
        return surface[:1].isupper() and not first

    def route(self, query: str) -> dict:
        """Analyze query and return routing info."""
        query_lower = query.lower()
        mentioned_cards = self.find_cards(query)

        # 1. Explicit rule numbers → CR only
        rule_numbers = RULE_NUMBER_RE.findall(query)
        if rule_numbers:
            return {
                "type": "rule_lookup",
                "sources": ["cr"],
                "rule_numbers": rule_numbers,
                "mentioned_cards": mentioned_cards,
                "query": query,
            }

        # 2. Tournament keywords → MTR/IPG/JAR
        if any(kw in query_lower for kw in TOURNAMENT_KEYWORDS):
            return {
                "type": "tournament",
                "sources": ["mtr", "ipg", "jar"],
                "mentioned_cards": mentioned_cards,
                "query": query,
            }

        # 3. Card name mentions → rules docs; cards get pinned by the retriever
        if mentioned_cards:
            return {
                "type": "card_interaction",
                "sources": RULES_SOURCES,
                "mentioned_cards": mentioned_cards,
                "query": query,
            }

        # 4. Default: general rules question
        return {
            "type": "general",
            "sources": RULES_SOURCES,
            "mentioned_cards": [],
            "query": query,
        }
