#!/usr/bin/env python3
"""Refresh everything the judge knows, then report what changed.

  python scripts/update.py              # or: make update
  python scripts/update.py --skip-metagame

Steps: fetch rules + cards + rulings → parse → incremental index (only new or
changed chunks get embedded) → refresh the MTGO metagame for `metagame_formats`.
Needs network; the judge itself stays offline. Run it on the Mac and
`make sync-pi`, or on the Pi directly (deploy/mtg-judge-update.timer runs it weekly).
"""

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from src.config import load_config

CORPUS = Path("corpus")
PY = sys.executable


def _run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run([PY, *args], check=True)


def _card_names() -> set[str]:
    path = CORPUS / "card_names.json"
    return set(json.loads(path.read_text())["names"]) if path.exists() else set()


def _versions() -> dict:
    path = CORPUS / "versions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _sets_of(names: set[str]) -> Counter:
    """Set names for the given cards (from the Scryfall oracle file)."""
    sets: Counter = Counter()
    with gzip.open(CORPUS / "scryfall_oracle.jsonl.gz", "rt") as f:
        for line in f:
            card = json.loads(line)
            if card["name"] in names:
                sets[card.get("set_name", "?")] += 1
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config")
    parser.add_argument("--skip-metagame", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    before_versions, before_cards = _versions(), _card_names()

    _run("scripts/fetch_corpus.py")
    _run("scripts/parse_corpus.py")
    _run("scripts/build_index.py", *(["--config", args.config] if args.config else []))
    if not args.skip_metagame:
        for fmt in config.get("metagame_formats") or []:
            _run("scripts/fetch_metagame.py", "--format", fmt)
            _run("scripts/analyze_metagame.py", "--format", fmt)

    after_versions, after_cards = _versions(), _card_names()
    new_cards = after_cards - before_cards

    index_version = {**after_versions, "built_at": datetime.now().isoformat(timespec="seconds")}
    Path(config["index_path"], "VERSION.json").write_text(json.dumps(index_version, indent=1))

    print("\n" + "=" * 50 + "\nWhat changed")
    old_cr, new_cr = before_versions.get("cr"), after_versions.get("cr")
    print(f"  Comprehensive Rules: {new_cr}" + (f" (was {old_cr})" if old_cr and old_cr != new_cr else " (unchanged)"))
    print(f"  Cards: {len(after_cards)} total, {len(new_cards)} new"
          + (f", {len(before_cards - after_cards)} removed" if before_cards - after_cards else ""))
    if new_cards and before_cards:
        for set_name, n in _sets_of(new_cards).most_common(5):
            print(f"    {n:4} from {set_name}")
    print(f"  Scryfall data as of {after_versions.get('scryfall_cards', '?')[:10]}")
    for fmt in [] if args.skip_metagame else config.get("metagame_formats") or []:
        report = Path(f"corpus/metagame/{fmt}/report.json")
        if report.exists():
            r = json.loads(report.read_text())
            print(f"  {fmt.title()} metagame: {r['events']} events, {r['decks']} decks, {r['window'][0]} → {r['window'][1]}")
    print("\n  Restart the judge to load the new index (on the Pi: sudo systemctl restart mtg-judge).")


if __name__ == "__main__":
    main()
