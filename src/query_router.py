"""Query router — detect query type and route to appropriate retrieval strategy."""

import json
import re
from pathlib import Path

RULE_NUMBER_RE = re.compile(r"\b(\d{3,4}\.\d+[a-z]?)\b")
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


class QueryRouter:
    """Analyzes a user query and returns routing hints for the retriever.

    Detection types:
    - rule_lookup: explicit rule number (e.g. "What does 702.16a say?")
    - tournament: tournament procedure / penalty questions
    - card_interaction: mentions a specific card name
    - general: general rules question (search all sources)
    """

    def __init__(self, card_names_path: str | None = "index/card_names.json"):
        """Load card name index for substring matching."""
        self.card_names: list[str] = []
        if card_names_path and Path(card_names_path).exists():
            self.card_names = json.loads(Path(card_names_path).read_text())

    def route(self, query: str) -> dict:
        """Analyze query and return routing info."""
        query_lower = query.lower()

        # 1. Explicit rule numbers → CR only
        rule_numbers = RULE_NUMBER_RE.findall(query)
        if rule_numbers:
            return {
                "type": "rule_lookup",
                "sources": ["cr"],
                "rule_numbers": rule_numbers,
                "query": query,
            }

        # 2. Tournament keywords → MTR/IPG/JAR
        if any(kw in query_lower for kw in TOURNAMENT_KEYWORDS):
            return {
                "type": "tournament",
                "sources": ["mtr", "ipg", "jar"],
                "query": query,
            }

        # 3. Card name mentions → search all, card chunks will rank high
        mentioned_cards = [
            name for name in self.card_names if name.lower() in query_lower
        ]
        if mentioned_cards:
            # Limit to top 5 to avoid noise from very common words matching card names
            mentioned_cards = mentioned_cards[:5]
            return {
                "type": "card_interaction",
                "sources": None,
                "mentioned_cards": mentioned_cards,
                "query": query,
            }

        # 4. Default: general rules question
        return {
            "type": "general",
            "sources": None,
            "query": query,
        }
