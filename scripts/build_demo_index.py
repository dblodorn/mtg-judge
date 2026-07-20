#!/usr/bin/env python3
"""
Build a lightweight demo index: CR + MTR + JAR + IPG only (no Scryfall).
~2400 chunks, builds in ~5 min on CPU.

Output: demo/index/
"""

import json
import shutil
import sys
from pathlib import Path

import bm25s
import chromadb
from tqdm import tqdm

sys.path.insert(0, ".")
from src.embeddings import Embeddings, EmbeddingsUnavailable

CORPUS_DIR = Path("corpus")
DEMO_INDEX_DIR = Path("demo/index")
DEMO_CHROMA_DIR = DEMO_INDEX_DIR / "chroma"
DEMO_BM25_DIR = DEMO_INDEX_DIR / "bm25"

SKIP_SOURCES = {"scryfall"}


def build_demo_index():
    """Build a Scryfall-free index for the demo."""
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print("✗ corpus/chunks.jsonl not found. Run `make corpus` first.")
        sys.exit(1)

    all_chunks = [json.loads(line) for line in open(chunks_path)]
    chunks = [c for c in all_chunks if c.get("source") not in SKIP_SOURCES]
    print(f"Building demo index for {len(chunks)} chunks (excluded Scryfall)...\n")

    # 1. Dense (ChromaDB)
    print("1. Building dense index (ChromaDB + nomic-embed-text)...")
    DEMO_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DEMO_CHROMA_DIR))
    col = client.get_or_create_collection("mtg_judge", metadata={"hnsw:space": "cosine"})

    existing = set()
    if col.count() > 0:
        existing = set(col.get()["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing]
    print(f"  ChromaDB: {len(existing)} existing, {len(new_chunks)} to index")

    if new_chunks:
        emb = Embeddings()
        batch_size = 50
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="  Dense", unit="batch"):
            batch = new_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            try:
                embeddings = emb.embed_batch(texts)
            except EmbeddingsUnavailable:
                print("\n  ⚠ Ollama not available. Run `ollama serve` first.")
                sys.exit(1)
            col.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{k: v for k, v in c.items() if k != "text"} for c in batch],
            )

    print(f"  ✓ Dense index: {col.count()} chunks")

    # 2. Sparse (BM25)
    print("\n2. Building sparse index (BM25)...")
    DEMO_BM25_DIR.mkdir(parents=True, exist_ok=True)
    corpus_tokens = bm25s.tokenize([c["text"] for c in chunks], show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(DEMO_BM25_DIR), corpus=corpus_tokens)
    (DEMO_INDEX_DIR / "chunk_ids.json").write_text(json.dumps([c["id"] for c in chunks]))
    print(f"  ✓ BM25 index: {len(chunks)} chunks")

    # 3. Parent store (filtered, no Scryfall parents)
    print("\n3. Building parent store...")
    parents_path = CORPUS_DIR / "parents.jsonl"
    demo_parents = [
        json.loads(line)
        for line in open(parents_path)
        if json.loads(line).get("source") not in SKIP_SOURCES
    ]
    with open(DEMO_INDEX_DIR / "parents.jsonl", "w") as f:
        for p in demo_parents:
            f.write(json.dumps(p) + "\n")
    print(f"  ✓ Parent store: {len(demo_parents)} documents")

    print(f"\n{'='*50}")
    print(f"✅ Demo index built at {DEMO_INDEX_DIR}/")
    print(f"   Dense: {col.count()} | Sparse: {len(chunks)} | Parents: {len(demo_parents)}")


if __name__ == "__main__":
    build_demo_index()
