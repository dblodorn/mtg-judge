#!/usr/bin/env python3
"""Fetch MTG rules corpus: CR, MTR, JAR, IPG, and Scryfall oracle cards.

Outputs (corpus/):
  cr.txt                — official Comprehensive Rules text (from magic.wizards.com)
  mtr.jsonl, ipg.jsonl  — one {section, title, paragraphs} record per section page
  jar.jsonl             — one record per JAR heading
  scryfall_oracle.jsonl.gz — Scryfall oracle_cards bulk file
  scryfall_rulings.jsonl.gz — official card rulings
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CORPUS_DIR = Path("corpus")

HEADERS = {"User-Agent": "mtg-judge/0.1 (+https://github.com/dblodorn/mtg-judge)"}
BLOG = "https://blogs.magicjudges.org/rules"
CR_INDEX = "https://magic.wizards.com/en/rules"
CR_TXT_RE = re.compile(r"https://media\.wizards\.com/[^\"'<>]+?\.txt")
# Minimum sections per source — below this the site layout probably changed.
MIN_SECTIONS = {"mtr": 40, "ipg": 20, "jar": 5}


def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=60, headers=HEADERS, **kwargs)
    resp.raise_for_status()
    return resp


def _soup(url: str) -> BeautifulSoup:
    return BeautifulSoup(_get(url).text, "lxml")


def _content(soup: BeautifulSoup):
    article = soup.find("article") or soup.find("main") or soup.body
    return article.find(class_="entry-content") or article


def _official_paragraphs(container) -> list[str]:
    """Text blocks of the official document, skipping blog annotations (.alert boxes)."""
    paragraphs = []
    for el in container.find_all(["p", "li", "div"], recursive=True):
        if el.find_parent(class_="alert") or "alert" in (el.get("class") or []):
            continue
        # Leaf-ish blocks only, so nested divs/lists aren't emitted twice
        if el.name == "div" and el.find(["p", "li", "div"]):
            continue
        if el.name == "p" and el.find_parent("li"):
            continue
        text = el.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return paragraphs


def record_version(key: str, value: str) -> None:
    """Remember what was fetched (corpus/versions.json) so `make update` can report changes."""
    path = CORPUS_DIR / "versions.json"
    versions = json.loads(path.read_text()) if path.exists() else {}
    versions[key] = value
    path.write_text(json.dumps(versions, indent=1))


def _check(source: str, records: list) -> None:
    if len(records) < MIN_SECTIONS[source]:
        sys.exit(
            f"✗ {source.upper()}: only {len(records)} sections parsed (expected ≥{MIN_SECTIONS[source]}). "
            "The page layout may have changed — check scripts/fetch_corpus.py."
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def fetch_cr():
    """Fetch the official Comprehensive Rules .txt linked from magic.wizards.com/en/rules."""
    print("Fetching Comprehensive Rules...")
    links = CR_TXT_RE.findall(_get(CR_INDEX).text)
    if not links:
        sys.exit(f"✗ No CR .txt link found on {CR_INDEX}")
    url = quote(links[0], safe=":/")
    print(f"  {links[0]}")
    resp = _get(url)
    text = resp.content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    (CORPUS_DIR / "cr.txt").write_text(text)
    print(f"  → {len(text.splitlines())} lines written to corpus/cr.txt")
    m = re.search(r"(\d{8})\.txt$", links[0])
    record_version("cr", m.group(1) if m else links[0].rsplit("/", 1)[-1])


def _fetch_sectioned(source: str, prefix: str):
    """MTR/IPG: hub page links to one page per section (e.g. /rules/mtr3-5/)."""
    print(f"Fetching {source.upper()}...")
    hub = _content(_soup(f"{BLOG}/{source}/"))
    link_re = re.compile(rf"/rules/{prefix}(\d+(?:-\d+)*)/?$")
    urls = []
    for a in hub.find_all("a", href=True):
        if link_re.search(a["href"]):
            url = a["href"].replace("http://", "https://")
            if url not in urls:
                urls.append(url)

    records = []
    for url in tqdm(urls, desc=f"  {source.upper()}", unit="page"):
        soup = _soup(url)
        # <title>: "MTR 3.5 Substitute Cards - Rules Resources"
        title = (soup.title.string if soup.title else "").split(" - Rules Resources")[0].strip()
        m = re.match(rf"{source.upper()}\s+([\d.]+)\s*(.*)", title, re.IGNORECASE)
        section = m.group(1).rstrip(".") if m else link_re.search(url).group(1).replace("-", ".")
        paragraphs = _official_paragraphs(_content(soup))
        if paragraphs:
            records.append({
                "section": section,
                "title": (m.group(2) if m else title).strip(),
                "url": url,
                "paragraphs": paragraphs,
            })
        time.sleep(0.2)  # be polite to the judges' blog

    _check(source, records)
    _write_jsonl(CORPUS_DIR / f"{source}.jsonl", records)
    print(f"  → {len(records)} sections written to corpus/{source}.jsonl")


def fetch_mtr():
    _fetch_sectioned("mtr", "mtr")


def fetch_ipg():
    _fetch_sectioned("ipg", "ipg")


def fetch_jar():
    """JAR is a single page; split it on h2/h3 headings."""
    print("Fetching JAR...")
    content = _content(_soup(f"{BLOG}/jar/"))
    records, current = [], None
    for el in content.find_all(["h2", "h3", "p", "li"]):
        if el.name in ("h2", "h3"):
            title = el.get_text(" ", strip=True)
            current = {"section": re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
                       "title": title, "paragraphs": []}
            records.append(current)
        elif current is not None and not el.find_parent(class_="alert"):
            if el.name == "p" and el.find_parent("li"):
                continue
            text = el.get_text(" ", strip=True)
            if text:
                current["paragraphs"].append(text)
    records = [r for r in records if r["paragraphs"]]

    _check("jar", records)
    _write_jsonl(CORPUS_DIR / "jar.jsonl", records)
    print(f"  → {len(records)} sections written to corpus/jar.jsonl")


def fetch_scryfall():
    """Fetch Scryfall oracle cards bulk data (~25MB gzipped JSONL)."""
    print("Fetching Scryfall bulk-data index...")
    bulk_data = _get("https://api.scryfall.com/bulk-data").json()["data"]
    oracle = next(b for b in bulk_data if b["type"] == "oracle_cards")
    print(f"  Oracle cards: {oracle['name']} (updated {oracle['updated_at'][:10]})")
    record_version("scryfall_cards", oracle["updated_at"])

    # Scryfall now publishes gzipped JSON Lines; older API responses had a JSON array
    if oracle.get("jsonl_download_uri"):
        url, outpath = oracle["jsonl_download_uri"], CORPUS_DIR / "scryfall_oracle.jsonl.gz"
    else:
        url, outpath = oracle["download_uri"], CORPUS_DIR / "scryfall_oracle.json"

    resp = _get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(outpath, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, desc="  Scryfall") as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  → {outpath.stat().st_size // 1024 // 1024}MB written to {outpath}")

    # Official rulings (Gatherer/WotC + Scryfall notes), keyed by oracle_id
    rulings = next((b for b in bulk_data if b["type"] == "rulings"), None)
    if rulings and rulings.get("jsonl_download_uri"):
        path = CORPUS_DIR / "scryfall_rulings.jsonl.gz"
        path.write_bytes(_get(rulings["jsonl_download_uri"]).content)
        record_version("scryfall_rulings", rulings["updated_at"])
        print(f"  → rulings written to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MTG rules corpus")
    parser.add_argument(
        "--only",
        choices=["cr", "mtr", "jar", "ipg", "scryfall"],
        help="Fetch only one source",
    )
    args = parser.parse_args()
    CORPUS_DIR.mkdir(exist_ok=True)

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
