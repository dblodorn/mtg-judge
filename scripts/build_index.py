#!/usr/bin/env python3
"""
Build the dual index for hybrid retrieval:
  1. ChromaDB — dense vector store (nomic-embed-text, 768-dim)
  2. BM25 — sparse keyword index (bm25s)
  3. Parent store — JSON file for parent document expansion

Resumable: skips chunks already in ChromaDB.
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
INDEX_DIR = Path("index")
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_DIR = INDEX_DIR / "bm25"


def build_dense_index(chunks: list[dict], emb: Embeddings, batch_size: int = 50) -> int:
    """Build ChromaDB dense index. Resumable — skips already-indexed IDs."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection("mtg_judge", metadata={"hnsw:space": "cosine"})

    # Get already-indexed IDs
    existing = set()
    if col.count() > 0:
        existing = set(col.get()["ids"])

    new_chunks = [c for c in chunks if c["id"] not in existing]
    print(f"  ChromaDB: {len(existing)} existing, {len(new_chunks)} to index")

    if not new_chunks:
        print("  ✓ Dense index already complete")
        return len(existing)

    indexed = 0
    for i in tqdm(range(0, len(new_chunks), batch_size), desc="  Dense", unit="batch"):
        batch = new_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Embed batch
        try:
            embeddings = emb.embed_batch(texts)
        except EmbeddingsUnavailable:
            print("\n  ⚠ Ollama not available. Run `ollama serve` first.")
            sys.exit(1)

        # Add to ChromaDB
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {k: v for k, v in c.items() if k != "text"} for c in batch
            ],
        )
        indexed += len(batch)

    total = len(existing) + indexed
    print(f"  ✓ Dense index: {total} chunks total")
    return total


def build_sparse_index(chunks: list[dict]) -> int:
    """Build BM25 sparse index."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  BM25: tokenizing {len(chunks)} chunks...")
    corpus_tokens = bm25s.tokenize([c["text"] for c in chunks], show_progress=False)

    print("  BM25: indexing...")
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(BM25_DIR), corpus=corpus_tokens)

    # Save chunk ID list (for mapping BM25 results back to chunk metadata)
    (INDEX_DIR / "chunk_ids.json").write_text(json.dumps([c["id"] for c in chunks]))
    print(f"  ✓ BM25 index: {len(chunks)} chunks")
    return len(chunks)


def build_parent_store() -> int:
    """Copy parents.jsonl to index/ for runtime access."""
    src = CORPUS_DIR / "parents.jsonl"
    dst = INDEX_DIR / "parents.jsonl"
    shutil.copy(src, dst)

    count = sum(1 for _ in open(dst))
    print(f"  ✓ Parent store: {count} parent documents")
    return count


def main():
    # Load chunks
    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print("✗ corpus/chunks.jsonl not found. Run `make corpus` first.")
        sys.exit(1)

    chunks = [json.loads(line) for line in open(chunks_path)]
    print(f"Building index for {len(chunks)} chunks...\n")

    # 1. Dense (ChromaDB)
    print("1. Building dense index (ChromaDB + nomic-embed-text)...")
    emb = Embeddings()
    dense_count = build_dense_index(chunks, emb, batch_size=50)

    # 2. Sparse (BM25)
    print("\n2. Building sparse index (BM25)...")
    sparse_count = build_sparse_index(chunks)

    # 3. Parent store
    print("\n3. Building parent store...")
    parent_count = build_parent_store()

    # Summary
    print(f"\n{'='*50}")
    print(f"✅ Index built successfully.")
    print(f"   Dense (ChromaDB): {dense_count} chunks")
    print(f"   Sparse (BM25): {sparse_count} chunks")
    print(f"   Parents: {parent_count} documents")
    print(f"   Location: {INDEX_DIR}/")
    print(f"\n   Next: `make run` to start the judge.")


if __name__ == "__main__":
    main()
