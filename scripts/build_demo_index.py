#!/usr/bin/env python3
"""
Build a lightweight demo index: CR + MTR + JAR + IPG only (no Scryfall).
~5k chunks, builds in a few minutes.

Output: demo/index/ — same layout as index/, built with scripts/build_index.py's functions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from build_index import CORPUS_DIR, build_dense_index, build_sparse_index

from src.config import load_config
from src.embeddings import Embeddings

DEMO_INDEX_DIR = Path("demo/index")
SKIP_SOURCES = {"scryfall"}


def build_demo_index():
    """Build a Scryfall-free index for the demo."""
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit("✗ corpus/chunks.jsonl not found. Run `make corpus` first.")

    chunks = [c for c in map(json.loads, open(chunks_path)) if c.get("source") not in SKIP_SOURCES]
    print(f"Building demo index for {len(chunks)} chunks (excluded Scryfall)...\n")
    DEMO_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    print("1. Building dense index...")
    dense = build_dense_index(chunks, Embeddings(config["embed_model"], config["ollama_host"]), DEMO_INDEX_DIR)

    print("\n2. Building sparse index (BM25)...")
    build_sparse_index(chunks, DEMO_INDEX_DIR)

    print("\n3. Building parent store...")
    parents = [p for p in map(json.loads, open(CORPUS_DIR / "parents.jsonl")) if p.get("source") not in SKIP_SOURCES]
    with open(DEMO_INDEX_DIR / "parents.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in parents)
    print(f"  ✓ Parent store: {len(parents)} documents")

    print(f"\n✅ Demo index built at {DEMO_INDEX_DIR}/ (dense {dense}, sparse {len(chunks)}, parents {len(parents)})")


if __name__ == "__main__":
    build_demo_index()
