#!/usr/bin/env python3
"""Fetch recent MTGO decklists for a format from mtgo.com (official Wizards results).

  python scripts/fetch_metagame.py --format modern --months 6

Walks the monthly archives (https://www.mtgo.com/decklists/YYYY/MM) and saves each
Challenge / Showcase / Qualifier event that has published decklists to
corpus/metagame/<format>/<event_id>.json:

  {"event_id", "name", "date", "player_count",
   "decks": [{"player", "rank", "main": {card: qty}, "side": {card: qty}}]}

Polite: one request per second, identifies itself, and never refetches a saved
event. Recent events often have no decklists yet (Wizards publishes them a few
days later); those aren't saved, so a later run picks them up.

mtgdecks.net is deliberately not used: its robots.txt disallows AI crawlers.
"""

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://www.mtgo.com"
HEADERS = {"User-Agent": "mtg-judge/0.1 (+https://github.com/dblodorn/mtg-judge)"}
DELAY_SECONDS = 1.0
# Competitive events with full top-N decklists. Leagues are excluded: they publish
# a curated sample of 5-0 lists, which would skew metagame share.
EVENT_KINDS = ("challenge", "showcase", "qualifier", "championship")
DATA_RE = re.compile(r"window\.MTGO\.decklists\.data\s*=\s*(\{.*?\});\s*\n", re.S)


def _get(session: requests.Session, url: str) -> str:
    time.sleep(DELAY_SECONDS)
    resp = session.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def _months(since: date, until: date) -> list[tuple[int, int]]:
    months, y, m = [], since.year, since.month
    while (y, m) <= (until.year, until.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def event_links(session: requests.Session, fmt: str, since: date, until: date) -> list[str]:
    """Event page paths for `fmt` competitive events, from the monthly archives."""
    link_re = re.compile(rf'href="(/decklist/{fmt}-[a-z0-9-]*?(?:{"|".join(EVENT_KINDS)})[a-z0-9-]*)"')
    links: list[str] = []
    for y, m in _months(since, until):
        html = _get(session, f"{BASE}/decklists/{y}/{m:02d}?filter={fmt.title()}")
        for path in link_re.findall(html):
            if path not in links:
                links.append(path)
    return links


def _cards(entries: list[dict]) -> dict[str, int]:
    cards: dict[str, int] = {}
    for e in entries:
        name = e["card_attributes"]["card_name"]
        cards[name] = cards.get(name, 0) + int(e["qty"])
    return cards


def parse_event(html: str) -> dict | None:
    """Event JSON from the page's embedded data, or None if decklists aren't published."""
    m = DATA_RE.search(html)
    if not m:
        return None
    data = json.loads(m.group(1))
    if not data.get("decklists"):
        return None
    ranks = {s["loginid"]: int(s["rank"]) for s in data.get("standings", []) if s.get("rank")}
    return {
        "event_id": data["event_id"],
        "name": data["description"],
        "date": data["starttime"][:10],
        "player_count": int((data.get("player_count") or {}).get("players") or 0),
        "decks": [
            {
                "player": d["player"],
                "rank": ranks.get(d["loginid"]),
                "main": _cards(d["main_deck"]),
                "side": _cards(d["sideboard_deck"]),
            }
            for d in data["decklists"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--format", default="modern")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--out", default=None, help="default: corpus/metagame/<format>/")
    args = parser.parse_args()

    fmt = args.format.lower()
    out = Path(args.out or f"corpus/metagame/{fmt}")
    out.mkdir(parents=True, exist_ok=True)
    until = date.today()
    since = until - timedelta(days=round(args.months * 30.44))

    session = requests.Session()
    print(f"Collecting {fmt} events {since} → {until} from {BASE} ...")
    links = event_links(session, fmt, since, until)
    print(f"  {len(links)} candidate events")

    saved = skipped = pending = 0
    for path in links:
        # Paths end in "<yyyy-mm-dd><event id>", e.g. ".../modern-challenge-64-2026-04-3012841326"
        event_id = re.search(r"\d{4}-\d{2}-\d{2}(\d+)$", path).group(1)
        if any(out.glob(f"*_{event_id}.json")):
            skipped += 1
            continue
        try:
            event = parse_event(_get(session, BASE + path))
        except requests.RequestException as e:
            print(f"  ⚠ {path}: {e}")
            continue
        if event is None:
            pending += 1  # decklists not published yet
            continue
        if not (since <= datetime.strptime(event["date"], "%Y-%m-%d").date() <= until):
            continue
        (out / f"{event['date']}_{event['event_id']}.json").write_text(json.dumps(event))
        saved += 1
        print(f"  ✓ {event['date']} {event['name']} ({len(event['decks'])} decks)")

    total = len(list(out.glob("*.json")))
    print(f"\n✅ {saved} new, {skipped} already saved, {pending} without published decklists yet")
    print(f"   {total} events in {out}/")
    if total == 0:
        sys.exit("✗ No events saved — check the archive page format.")


if __name__ == "__main__":
    main()
