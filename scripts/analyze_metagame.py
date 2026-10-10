#!/usr/bin/env python3
"""Analyze a format's metagame from decklists saved by fetch_metagame.py.

  python scripts/analyze_metagame.py --format modern

MTGO results don't name archetypes, so decks are clustered by their nonland
mainboards (cosine similarity of card counts) and each cluster is named by its
colors and most distinctive cards. Tiers come from share of published decklists
(default: Tier 1 ≥ 5%, Tier 2 ≥ 2%).

Writes to corpus/metagame/<format>/:
  report.json — archetypes (share, top-8 share, avg finish, core/flex cards, sample list)
  report.md   — the same as a readable table
  cards.json  — every card in Tier 1–2 decks (main + side), by how much it's played;
                input for the card/ruling cache
"""

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SIMILARITY = 0.55   # cosine threshold to join an archetype
MIN_SHARE = 0.005   # clusters below 0.5% of decks are reported as "Other"
COLOR_ORDER = "WUBRG"


def load_card_info(path: Path = Path("corpus/scryfall_oracle.jsonl.gz")) -> dict[str, dict]:
    """Card name (full and front face, lowercased) → {is_land, colors}."""
    info: dict[str, dict] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            c = json.loads(line)
            entry = {
                "is_land": "Land" in c.get("type_line", "").split("//")[0],
                "colors": c.get("color_identity", []),
            }
            for name in {c["name"], c["name"].split(" // ")[0]}:
                info.setdefault(name.lower(), entry)
    return info


def load_decks(folder: Path, since: date) -> list[dict]:
    decks = []
    for path in sorted(folder.glob("*_*.json")):
        event = json.loads(path.read_text())
        if date.fromisoformat(event["date"]) < since:
            continue
        for d in event["decks"]:
            decks.append({**d, "event": event["name"], "date": event["date"]})
    return decks


def _normalize(name: str) -> str:
    # MTGO writes split/adventure cards as "Fire/Ice"; Scryfall as "Fire // Ice"
    return re.sub(r"\s*/+\s*", " // ", name).lower()


def cluster(vectors: np.ndarray, threshold: float = SIMILARITY, passes: int = 3) -> np.ndarray:
    """Greedy leader clustering on L2-normalized rows, then k-means style reassignment."""
    labels = np.full(len(vectors), -1)
    centroids: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        if centroids:
            sims = np.array(centroids) @ v
            best = int(sims.argmax())
            if sims[best] >= threshold:
                labels[i] = best
                continue
        centroids.append(v.copy())
        labels[i] = len(centroids) - 1
    for _ in range(passes):
        cents = []
        for k in range(len(centroids)):
            members = vectors[labels == k]
            if len(members):
                c = members.mean(axis=0)
                cents.append(c / (np.linalg.norm(c) + 1e-9))
        centroids = cents
        labels = (vectors @ np.array(centroids).T).argmax(axis=1)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--format", default="modern")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--tier1", type=float, default=0.05, help="min share for Tier 1")
    parser.add_argument("--tier2", type=float, default=0.02, help="min share for Tier 2")
    args = parser.parse_args()

    folder = Path(f"corpus/metagame/{args.format.lower()}")
    since = date.today() - timedelta(days=round(args.months * 30.44))
    decks = load_decks(folder, since)
    if not decks:
        raise SystemExit(f"✗ No decks in {folder}/ — run scripts/fetch_metagame.py first.")
    card_info = load_card_info()

    # Nonland mainboard vectors
    def nonland(main: dict[str, int]) -> dict[str, int]:
        return {n: q for n, q in main.items() if not card_info.get(_normalize(n), {}).get("is_land", False)}

    vocab = sorted({n for d in decks for n in nonland(d["main"])})
    index = {n: i for i, n in enumerate(vocab)}
    vectors = np.zeros((len(decks), len(vocab)), dtype=np.float32)
    for row, d in enumerate(decks):
        for n, q in nonland(d["main"]).items():
            vectors[row, index[n]] = q
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    labels = cluster(vectors)

    # Distinctiveness: how much more often a card shows up in this cluster than overall
    overall = Counter(n for d in decks for n in nonland(d["main"]))
    total = len(decks)
    groups: dict[int, list[dict]] = defaultdict(list)
    for d, k in zip(decks, labels, strict=True):
        groups[int(k)].append(d)

    archetypes = []
    other: list[dict] = []
    for members in groups.values():
        if len(members) / total < MIN_SHARE:
            other.extend(members)
            continue
        n = len(members)
        freq = Counter(c for d in members for c in nonland(d["main"]))
        signature = sorted(freq, key=lambda c: -(freq[c] / n) * math.log(total / overall[c]))[:2]
        core = sorted(c for c, k in freq.items() if k / n >= 0.8)
        colors = {col for c in core for col in card_info.get(_normalize(c), {}).get("colors", [])}
        color_code = "".join(c for c in COLOR_ORDER if c in colors) or "C"
        ranks = [d["rank"] for d in members if d.get("rank")]
        sample = min(members, key=lambda d: (d.get("rank") or 999, d["date"]))
        archetypes.append({
            "name": f"{color_code} — {' / '.join(signature)}",
            "colors": color_code,
            "decks": n,
            "share": n / total,
            "top8_share": sum(1 for r in ranks if r <= 8) / max(sum(1 for d in decks if (d.get("rank") or 99) <= 8), 1),
            "avg_finish": round(sum(ranks) / len(ranks), 1) if ranks else None,
            "core_cards": core,
            "flex_cards": sorted((c for c, k in freq.items() if 0.3 <= k / n < 0.8), key=lambda c: -freq[c]),
            "sample": {"player": sample["player"], "event": sample["event"], "date": sample["date"],
                       "rank": sample.get("rank"), "main": sample["main"], "side": sample["side"]},
            "_members": members,
        })
    archetypes.sort(key=lambda a: -a["share"])
    for a in archetypes:
        a["tier"] = 1 if a["share"] >= args.tier1 else 2 if a["share"] >= args.tier2 else 3

    # Card table for Tier 1–2: copies per deck averaged over all Tier 1–2 decks
    tiered = [a for a in archetypes if a["tier"] <= 2]
    tiered_decks = [d for a in tiered for d in a["_members"]]
    cards: Counter = Counter()
    seen_in: dict[str, set] = defaultdict(set)
    for a in tiered:
        for d in a["_members"]:
            for zone in ("main", "side"):
                for c, q in d[zone].items():
                    cards[c] += q
                    seen_in[c].add(a["name"])
    card_rows = [
        {"name": c, "copies_per_deck": round(q / len(tiered_decks), 3), "archetypes": sorted(seen_in[c])}
        for c, q in cards.most_common()
    ]

    for a in archetypes:
        del a["_members"]
    report = {
        "format": args.format.lower(),
        "window": [since.isoformat(), date.today().isoformat()],
        "events": len({(d["event"], d["date"]) for d in decks}),
        "decks": total,
        "other_decks": len(other),
        "thresholds": {"tier1": args.tier1, "tier2": args.tier2},
        "archetypes": archetypes,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=1))
    (folder / "cards.json").write_text(json.dumps(card_rows, indent=1))

    lines = [
        f"# {args.format.title()} metagame {since} → {date.today()}",
        "",
        f"{report['events']} MTGO events · {total} published decklists · "
        f"{len(other)} decks in archetypes under {MIN_SHARE:.1%} (\"Other\").",
        "",
        "| Tier | Archetype | Share | Top-8 share | Avg finish | Decks |",
        "|---|---|---|---|---|---|",
    ]
    for a in archetypes:
        if a["tier"] <= 2:
            lines.append(f"| {a['tier']} | {a['name']} | {a['share']:.1%} | {a['top8_share']:.1%} | "
                         f"{a['avg_finish']} | {a['decks']} |")
    lines += ["", f"{len(card_rows)} distinct cards across Tier 1–2 decks (main + side) → cards.json"]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
