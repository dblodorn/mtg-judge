#!/usr/bin/env python3
"""Fetch MTG rules corpus: CR, MTR, JAR, IPG, and Scryfall oracle cards."""

import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CORPUS_DIR = Path("corpus")
CORPUS_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "mtg-judge/0.1 (+https://github.com/dblodorn/mtg-judge)"}
RULE_RE = re.compile(r"^(\d{3,4}\.\d+[a-z]?)\s+(.+)$", re.MULTILINE)


def _fetch_html(url: str) -> str:
    """Fetch HTML and return plain text with rules preserved."""
    resp = requests.get(url, timeout=30, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    return soup.get_text()


def _extract_rules(text: str) -> str:
    """Extract rule-numbered lines from page text, preserving structure."""
    rules = RULE_RE.findall(text)
    lines = [f"{num} {body}" for num, body in rules]
    return "\n".join(lines)


def fetch_cr():
    """Fetch Comprehensive Rules from magicjudges.org."""
    print("Fetching Comprehensive= Comprehensive Rules...")
    text = _fetch_html("https://blogs.magicjudges.org/rules/cr/")
    rules_text = _extract_rules(text)
    (CORPUS_DIR / "cr.txt").write_text(rules_text)
    print(f"  → {len(rules_text.splitlines())} rules written to corpus/cr.txt")


def fetch_mtr():
    """Fetch Magic Tournament Rules."""
    print("Fetching Magic Tournament Rules...")
    try:
        text = _fetch_html("https://blogs.magicjudges.org/rules/mtr/")
        rules_text = _extract_rules(text)
        if not rules_text:
            # MTR may not have the same numbering; save raw text
            rules_text = text
        (CORPUS_DIR / "mtr.txt").write_text(rules_text)
        print(f"  → {len(rules_text.splitlines())} lines written to corpus/mtr.txt")
    except Exception as e:
        print(f"  ⚠ MTR fetch failed: {e}")
        (CORPUS_DIR / "mtr.txt").write_text("")


def fetch_jar():
    """Fetch Judging at Regular (JAR)."""
    print("Fetching JAR...")
    try:
        text = _fetch_html("https://blogs.magicjudges.org/rules/jar/")
        (CORPUS_DIR / "jar.txt").write_text(text)
        print(f"  → {len(text.splitlines())} lines written to corpus/jar.txt")
    except Exception as e:
        print(f"  ⚠ JAR fetch failed: {e}")
        (CORPUS_DIR / "jar.txt").write_text("")


def fetch_ipg():
    """Fetch Infraction Procedure Guide."""
    print("Fetching IPG...")
    try:
        text = _fetch_html("https://blogs.magicjudges.org/rules/ipg/")
        (CORPUS_DIR / "ipg.txt").write_text(text)
        print(f"  → {len(text.splitlines())} lines written to corpus/ipg.txt")
    except Exception as e:
        print(f"  ⚠ IPG fetch failed: {e}")
        (CORPUS_DIR / "ipg.txt").write_text("")


def fetch_scryfall():
    """Fetch Scryfall oracle cards bulk JSON (~180MB)."""
    print("Fetching Scryfall bulk-data index...")
    resp = requests.get("https://api.scryfall.com/bulk-data", timeout=15, headers=HEADERS)
    resp.raise_for_status()

    bulk_data = resp.json()["data"]
    oracle = next(b for b in bulk_data if b["type"] == "oracle_cards")
    size_mb = oracle["size"] // 1024 // 1024
    print(f"  Oracle cards: {oracle['name']} ({size_mb}MB, updated {oracle['updated_at'][:10]})")

    print(f"  Downloading...")
    resp = requests.get(oracle["download_uri"], stream=True, timeout=60, headers=HEADERS)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    outpath = CORPUS_DIR / "scryfall_oracle.json"
    with open(outpath, "wb") as f:
        for chunk in tqdm(
            resp.iter_content(chunk_size=65536),
            total=total // 65536 if total else None,
            desc="  Scryfall",
            unit="chunks",
        ):
            f.write(chunk)

    print(f"  → {size_mb}MB written to corpus/scryfall_oracle.json")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MTG rules corpus")
    parser.add_argument(
        "--only",
        choices=["cr", "mtr", "jar", "ipg", "scryfall"],
        help="Fetch only one source",
    )
    args = parser.parse_args()

    fetchers = {
        "cr": fetch_cr,
        "mtr": fetch_mtr,
        "jar": fetch_jar,
        "ipg": fetch_ipg,
        "scryfall": fetch_scryfall,
    }

    if args.only:
        fetchers[args.only]()
    else:
        for fetch in fetchers.values():
            fetch()

    print("\n✅ Corpus fetched.")


if __name__ == "__main__":
    main()
