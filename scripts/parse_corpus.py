#!/usr/bin/env python3
"""
Parse MTG corpus into hierarchical chunks:
  - corpus/chunks.jsonl     — child chunks (what gets embedded for precise matching)
  - corpus/parents.jsonl    — parent documents (what gets returned to the LLM for context)
  - corpus/card_names.json  — {"names", "ambiguous"} for query routing (copied to index/ by build_index)

Scryfall rulings (corpus/scryfall_rulings.jsonl.gz) attach to their card when present.

CR: each numbered rule (702.16a, or 100.1 when it has no letter) is a child, with
  its "Example:" lines folded in; its parent is the rule group (702.16), one line per rule.
  Glossary entries are self-parented chunks.
MTR/IPG/JAR: each section page is a parent; each paragraph is a child.
Scryfall: each card is both a child and its own parent (cards are short enough).

Every child carries an `anchor` — the line of the parent text it came from — so the
retriever can trim a long parent down to the part around the match.
"""

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import ijson

sys.path.insert(0, ".")
from src.formats import FORMATS

CORPUS_DIR = Path("corpus")

# "702.16a Text", "702.16. Protection", "100.1. Text"
CR_RULE_RE = re.compile(r"^(\d{3})\.(\d+)([a-z]{0,2})\.?\s+(.+)$")
CR_CHAPTER_RE = re.compile(r"^\d{3}\.\s+\S")
SKIP_LAYOUTS = {"token", "double_faced_token", "art_series", "planar", "scheme", "vanguard", "emblem", "augment", "host"}
MIN_COUNTS = {"cr": 2500, "cr_glossary": 300, "mtr": 40, "ipg": 20, "jar": 3, "scryfall": 20000}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ----------------------------------------------------------------------
# Comprehensive Rules
# ----------------------------------------------------------------------

def _cr_body_and_glossary(text: str) -> tuple[list[str], list[str]]:
    """Split cr.txt into (rules lines, glossary lines), skipping the table of contents."""
    lines = [line.strip() for line in text.splitlines()]
    credits = [i for i, line in enumerate(lines) if line == "Credits"]
    glossary = [i for i, line in enumerate(lines) if line == "Glossary"]
    if len(credits) < 2 or len(glossary) < 2:
        raise ValueError("cr.txt doesn't look like the official CR (missing Glossary/Credits markers)")
    body_start = credits[0] + 1  # the table of contents ends with "Credits"
    return lines[body_start : glossary[1]], lines[glossary[1] + 1 : credits[1]]


def parse_cr(text: str) -> tuple[list[dict], list[dict]]:
    body, glossary = _cr_body_and_glossary(text)

    # rule number → {"title"/"text"}; examples fold into the preceding rule
    rules: dict[str, str] = {}
    titles: dict[str, str] = {}  # "702.16" → "Protection"
    last = None
    for line in body:
        if not line:
            continue
        m = CR_RULE_RE.match(line)
        if m:
            group = f"{m.group(1)}.{m.group(2)}"
            num = group + m.group(3)
            body_text = m.group(4)
            # "702.16. Protection" — a short heading for a group with lettered sub-rules
            if not m.group(3) and line.startswith(group + ".") and len(body_text) < 60 and not body_text.endswith("."):
                titles[group] = body_text
            else:
                rules[num] = body_text
            last = num
        elif line.startswith("Example:") and last in rules:
            rules[last] += " " + line
        elif CR_CHAPTER_RE.match(line):
            last = None

    groups: dict[str, list[str]] = {}
    for num in rules:
        groups.setdefault(re.match(r"\d{3}\.\d+", num).group(), []).append(num)

    children, parents = [], []
    for group, nums in groups.items():
        title = titles.get(group, "")
        parent_id = f"cr:{group}"
        heading = f"{group} {title}".strip()
        lines = ([heading] if title else []) + [f"{n} {rules[n]}" for n in nums]
        parents.append({
            "id": parent_id,
            "source": "cr",
            "section": group,
            "title": title,
            "text": "\n".join(lines),
        })
        for n in nums:
            children.append({
                "id": f"cr:{n}",
                "source": "cr",
                "section": n,
                "parent_id": parent_id,
                "anchor": f"{n} ",
                "text": f"{n} {title + ': ' if title else ''}{rules[n]}",
            })

    # Glossary: "Term" line, then definition line(s), blank-separated
    entries, current = [], []
    for line in glossary + [""]:
        if line:
            current.append(line)
        elif current:
            entries.append(current)
            current = []
    for term, *definition in entries:
        if not definition:
            continue
        gid = f"cr:glossary:{_slug(term)}"
        entry = {
            "id": gid,
            "source": "cr",
            "section": f"Glossary: {term}",
            "title": term,
            "text": f"{term}: {' '.join(definition)}",
        }
        parents.append(entry)
        children.append({**entry, "parent_id": gid, "anchor": f"{term}:"})

    return children, parents


# ----------------------------------------------------------------------
# MTR / IPG / JAR
# ----------------------------------------------------------------------

def parse_sectioned(path: Path, source: str) -> tuple[list[dict], list[dict]]:
    children, parents = [], []
    for line in open(path):
        rec = json.loads(line)
        if source == "jar" and rec["section"] == "resources":
            continue
        section = rec["section"]
        title = re.sub(rf"^{re.escape(section)}\s+", "", rec["title"])  # "3.4 Tournament Error…" → "Tournament Error…"
        parent_id = f"{source}:{section}"
        label = f"{source.upper()} {section}" if source != "jar" else "JAR"
        parents.append({
            "id": parent_id,
            "source": source,
            "section": section if source != "jar" else title,
            "title": title,
            "text": "\n".join([f"{label} {title}", *rec["paragraphs"]]),
        })
        for i, para in enumerate(rec["paragraphs"]):
            children.append({
                "id": f"{parent_id}:{i}",
                "source": source,
                "section": section if source != "jar" else title,
                "parent_id": parent_id,
                "anchor": para[:80],
                "text": f"{label} {title}: {para}",
            })
    return children, parents


# ----------------------------------------------------------------------
# Scryfall
# ----------------------------------------------------------------------

def _card_text(card: dict) -> str:
    """Name, cost, type, rules text and stats — every face for multi-face cards."""
    faces = card.get("card_faces") or [card]
    parts = []
    for face in faces:
        bits = [face.get("name", card["name"])]
        if face.get("mana_cost"):
            bits.append(face["mana_cost"])
        if face.get("type_line") or card.get("type_line"):
            bits.append(face.get("type_line") or card["type_line"])
        line = " — ".join(bits)
        if face.get("power") is not None:
            line += f" ({face['power']}/{face['toughness']})"
        if face.get("loyalty"):
            line += f" (loyalty {face['loyalty']})"
        if face.get("oracle_text"):
            line += f"\n{face['oracle_text']}"
        parts.append(line)
    return "\n//\n".join(parts)


def _iter_cards(path: Path):
    """Stream cards from the bulk file: gzipped JSON Lines, or a legacy JSON array."""
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "rb") as f:
            yield from ijson.items(f, "item", use_float=True)


def load_rulings(path: Path | None) -> dict[str, list[dict]]:
    """oracle_id → rulings (oldest first)."""
    rulings: dict[str, list[dict]] = {}
    if path and path.exists():
        for r in _iter_cards(path):
            rulings.setdefault(r["oracle_id"], []).append(r)
    return rulings


TOURNAMENT_LEGAL = {"legal", "restricted", "banned"}


def _is_real_card(card: dict) -> bool:
    """Black-bordered paper Magic: legal (or banned/restricted) in Vintage or Commander.

    Excludes Arena-only/Alchemy designs and Un-/playtest cards. Not `digital`: oracle_cards
    sometimes represents a real paper card by an MTGO-only printing (Humility → Tempest
    Remastered), and filtering on it dropped ~1,000 real cards.
    """
    legal = card.get("legalities", {})
    return legal.get("vintage") in TOURNAMENT_LEGAL or legal.get("commander") in TOURNAMENT_LEGAL


def parse_scryfall_streaming(
    path: Path, rulings: dict[str, list[dict]] | None = None
) -> tuple[list[dict], list[dict], list[str]]:
    """Stream-parse Scryfall oracle cards to avoid OOM on large files.

    Each card is a child chunk; its parent is the card text plus its official
    rulings, and each ruling is a child of its own (so "does Grave-Troll count
    itself?" can match the ruling directly). IDs use the stable Scryfall
    oracle_id (names can collide once punctuation is stripped).
    """
    rulings = rulings or {}
    children, parents, card_names = [], [], []
    seen: set[str] = set()

    for card in _iter_cards(path):
        if card.get("layout", "normal") in SKIP_LAYOUTS:
            continue
        if not _is_real_card(card):
            continue
        name, oracle_id = card.get("name"), card.get("oracle_id")
        if not name or not oracle_id or oracle_id in seen:
            continue
        seen.add(oracle_id)

        chunk_id = f"scryfall:{oracle_id}"
        chunk = {
            "id": chunk_id,
            "source": "scryfall",
            "name": name,
            "section": name,
            "parent_id": chunk_id,
            "anchor": name,
            "text": _card_text(card),
            "type": card.get("type_line", ""),
            "mv": card.get("cmc", 0.0),
            # Dict → kept in parents.jsonl, dropped from Chroma metadata (flat scalars only)
            "legalities": {f: s for f, s in card.get("legalities", {}).items() if f in FORMATS},
        }
        children.append(chunk)
        card_rulings = rulings.get(oracle_id, [])
        parent_lines = [chunk["text"]]
        if card_rulings:
            parent_lines.append("Rulings:")
        for i, r in enumerate(card_rulings):
            line = f"- ({r['published_at']}) {r['comment']}"
            parent_lines.append(line)
            children.append({
                **{k: v for k, v in chunk.items() if k not in ("id", "text", "anchor")},
                "id": f"{chunk_id}:r{i}",
                "source": "ruling",
                "anchor": line,
                "text": f"{name} ruling ({r['published_at']}): {r['comment']}",
            })
        parents.append({**chunk, "text": "\n".join(parent_lines)})
        card_names.append(name)

    return children, parents, card_names


SYSTEM_DICT = Path("/usr/share/dict/words")


def ambiguous_card_names(card_names: list[str], cr_text: str) -> list[str]:
    """Single-word card names that are also ordinary words ("Fight", "Opt", "Island").

    Vocabulary = rules text + the system word list when present (macOS ships one).
    Distinctive names like "Counterspell" or "Thoughtseize" stay unambiguous.
    """
    vocab = set(re.findall(r"[a-z]+", cr_text.lower()))
    if SYSTEM_DICT.exists():
        vocab |= {w.strip().lower() for w in SYSTEM_DICT.read_text().splitlines()}
    # Faces count too: "What" in "Who // What // When // Where // Why"
    words = {face for n in card_names for face in [n, *n.split(" // ")] if " " not in face}
    return sorted(w for w in words if w.lower() in vocab)


# ----------------------------------------------------------------------

def main():
    children: list[dict] = []
    parents: list[dict] = []
    counts: Counter = Counter()

    cr_path = CORPUS_DIR / "cr.txt"
    cr_text = cr_path.read_text() if cr_path.exists() else ""
    if cr_text:
        print("Parsing CR...")
        ch, pa = parse_cr(cr_text)
        children += ch
        parents += pa
        counts["cr"] = sum(1 for c in ch if ":glossary:" not in c["id"])
        counts["cr_glossary"] = len(ch) - counts["cr"]
        print(f"  → {counts['cr']} rules in {sum(1 for p in pa if ':glossary:' not in p['id'])} groups, {counts['cr_glossary']} glossary entries")

    for source in ("mtr", "ipg", "jar"):
        path = CORPUS_DIR / f"{source}.jsonl"
        if path.exists():
            print(f"Parsing {source.upper()}...")
            ch, pa = parse_sectioned(path, source)
            children += ch
            parents += pa
            counts[source] = len(pa)
            print(f"  → {len(pa)} sections, {len(ch)} paragraphs")

    scf_path = next(
        (p for p in (CORPUS_DIR / "scryfall_oracle.jsonl.gz", CORPUS_DIR / "scryfall_oracle.json") if p.exists()),
        None,
    )
    if scf_path:
        print("Parsing Scryfall (streaming)...")
        rulings = load_rulings(CORPUS_DIR / "scryfall_rulings.jsonl.gz")
        ch, pa, card_names = parse_scryfall_streaming(scf_path, rulings)
        children += ch
        parents += pa
        counts["scryfall"] = len(pa)
        print(f"  → {sum(1 for c in ch if c['source'] == 'ruling')} rulings attached")
        ambiguous = ambiguous_card_names(card_names, cr_text)
        (CORPUS_DIR / "card_names.json").write_text(json.dumps({"names": card_names, "ambiguous": ambiguous}))
        print(f"  → {len(pa)} cards ({len(ambiguous)} single-word names marked ambiguous)")

    # Fail loudly on a parse that silently produced too little
    low = {s: (counts[s], n) for s, n in MIN_COUNTS.items() if s in counts and counts[s] < n}
    if low:
        sys.exit(f"✗ Suspiciously few chunks parsed: {low} (got, expected ≥). Check the corpus files.")

    dupes = [i for i, n in Counter(c["id"] for c in children).items() if n > 1]
    if dupes:
        sys.exit(f"✗ Duplicate chunk IDs: {dupes[:5]}")

    with open(CORPUS_DIR / "chunks.jsonl", "w") as f:
        for chunk in children:
            f.write(json.dumps(chunk) + "\n")
    with open(CORPUS_DIR / "parents.jsonl", "w") as f:
        for parent in parents:
            f.write(json.dumps(parent) + "\n")

    source_counts = Counter(c["source"] for c in children)
    print(f"\n✅ Wrote {len(children)} child chunks, {len(parents)} parent documents")
    print(f"   By source: {dict(source_counts)}")

    parent_ids = {p["id"] for p in parents}
    orphans = [c for c in children if c["parent_id"] not in parent_ids]
    if orphans:
        sys.exit(f"✗ {len(orphans)} orphan chunks (parent_id not found), e.g. {orphans[0]['id']}")
    print("   ✓ All child chunks have valid parent IDs")


if __name__ == "__main__":
    main()
