#!/usr/bin/env python3
"""
Parse MTG corpus into hierarchical chunks:
  - corpus/chunks.jsonl  — child chunks (what gets embedded for precise matching)
  - corpus/parents.jsonl — parent documents (what gets returned to the LLM for context)
  - index/card_names.json — card name list for query routing

CR/MTR/JAR/IPG: each numbered sub-rule is a child; its parent is the top-level rule (e.g. 702.16).
  Parent text = the full section including all sub-rules.
Scryfall: each card is both a child and its own parent (cards are short enough).
"""

import json
import re
import sys
from pathlib import Path

import ijson

CORPUS_DIR = Path("corpus")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

RULE_RE = re.compile(r"^(\d{3,4}\.\d+[a-z]?)\s+(.+)$")
PARENT_NUM_RE = re.compile(r"^(\d{3,4}\.\d+)")


def parse_rules_hierarchical(text_path: str, source: str):
    """Parse CR/MTR/JAR/IPG into child chunks + parent documents.

    Returns (children, parents) where each is a list of dicts.
    """
    text = Path(text_path).read_text()
    if not text:
        return [], []

    # Parse all rule lines
    rules = {}  # section -> body
    for line in text.splitlines():
        m = RULE_RE.match(line)
        if m:
            section = m.group(1)
            body = m.group(2)
            rules[section] = body

    if not rules:
        # MTR/JAR/IPG may not have CR-style numbering — treat each line as its own chunk
        children = []
        parents = {}
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            chunk_id = f"{source}:{i}"
            children.append({
                "id": chunk_id,
                "source": source,
                "section": str(i),
                "parent_id": chunk_id,
                "title": line[:60],
                "text": line,
            })
            parents[chunk_id] = {
                "id": chunk_id,
                "source": source,
                "section": str(i),
                "text": line,
                "children": [str(i)],
            }
        return children, list(parents.values())

    # Group rules by parent (top-level number, e.g. "702.16" from "702.16a")
    parent_groups = {}  # parent_num -> {sections: [], bodies: []}
    for section, body in rules.items():
        m = PARENT_NUM_RE.match(section)
        if not m:
            continue
        parent_num = m.group(1)
        if parent_num not in parent_groups:
            parent_groups[parent_num] = {"sections": [], "bodies": []}
        parent_groups[parent_num]["sections"].append(section)
        parent_groups[parent_num]["bodies"].append(f"{section} {body}")

    # Build child chunks
    children = []
    for section, body in rules.items():
        m = PARENT_NUM_RE.match(section)
        parent_num = m.group(1) if m else section
        parent_id = f"{source}:{parent_num}"
        children.append({
            "id": f"{source}:{section}",
            "source": source,
            "section": section,
            "parent_id": parent_id,
            "text": body,
        })

    # Build parent documents
    parents = []
    for parent_num, data in parent_groups.items():
        # Sort sections for deterministic ordering
        sorted_pairs = sorted(zip(data["sections"], data["bodies"]))
        full_text = "\n".join(body for _, body in sorted_pairs)
        parents.append({
            "id": f"{source}:{parent_num}",
            "source": source,
            "section": parent_num,
            "text": full_text,
            "children": [s for s, _ in sorted_pairs],
        })

    return children, parents


def parse_scryfall_streaming(json_path: str):
    """Stream-parse Scryfall oracle JSON to avoid OOM on large files.

    Each card is both a child chunk and its own parent.
    """
    children = []
    parents = []
    card_names = []

    SKIP_LAYOUTS = {"token", "art_series", "planar", "scheme", "vanguard", "emblem", "augment"}

    with open(json_path, "rb") as f:
        for card in ijson.items(f, "item"):
            layout = card.get("layout", "normal")
            if layout in SKIP_LAYOUTS:
                continue

            name = card.get("name", "")
            if not name:
                continue

            oracle_text = card.get("oracle_text", "")
            type_line = card.get("type_line", "")
            mv = card.get("mana_value", 0)

            # Slug for ID
            slug = name.lower().replace(" ", "-").replace(",", "").replace("'", "").replace("/", "")
            chunk_id = f"scf:{slug}"

            # Combine into searchable text
            full_text = f"{name} — {type_line} — MV:{mv}"
            if oracle_text:
                full_text += f" — {oracle_text}"

            chunk = {
                "id": chunk_id,
                "source": "scryfall",
                "name": name,
                "section": name,
                "parent_id": chunk_id,
                "text": full_text,
                "type": type_line,
                "mv": mv,
            }
            children.append(chunk)
            parents.append({**chunk, "children": [name]})
            card_names.append(name)

    return children, parents, card_names


def main():
    children = []
    parents = []

    # Parse rules documents
    for source, filename in [("cr", "cr.txt"), ("mtr", "mtr.txt"), ("jar", "jar.txt"), ("ipg", "ipg.txt")]:
        path = CORPUS_DIR / filename
        if path.exists():
            print(f"Parsing {source.upper()}...")
            ch, pa = parse_rules_hierarchical(str(path), source)
            children.extend(ch)
            parents.extend(pa)
            print(f"  → {len(ch)} child chunks, {len(pa)} parent docs")

    # Parse Scryfall (streaming)
    scf_path = CORPUS_DIR / "scryfall_oracle.json"
    if scf_path.exists():
        print("Parsing Scryfall (streaming)...")
        scf_children, scf_parents, card_names = parse_scryfall_streaming(str(scf_path))
        children.extend(scf_children)
        parents.extend(scf_parents)
        print(f"  → {len(scf_children)} card chunks")

        # Save card names for query routing
        (INDEX_DIR / "card_names.json").write_text(json.dumps(card_names))
        print(f"  → {len(card_names)} card names saved to index/card_names.json")

    # Write outputs
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    parents_path = CORPUS_DIR / "parents.jsonl"

    with open(chunks_path, "w") as f:
        for chunk in children:
            f.write(json.dumps(chunk) + "\n")

    with open(parents_path, "w") as f:
        for parent in parents:
            f.write(json.dumps(parent) + "\n")

    # Summary by source
    from collections import Counter
    source_counts = Counter(c["source"] for c in children)
    print(f"\n✅ Wrote {len(children)} child chunks, {len(parents)} parent documents")
    print(f"   By source: {dict(source_counts)}")

    # Verify parent-child linkage
    parent_ids = {p["id"] for p in parents}
    orphans = [c for c in children if c["parent_id"] not in parent_ids]
    if orphans:
        print(f"   ⚠ {len(orphans)} orphan chunks (parent_id not found)")
    else:
        print(f"   ✓ All child chunks have valid parent IDs")


if __name__ == "__main__":
    main()
