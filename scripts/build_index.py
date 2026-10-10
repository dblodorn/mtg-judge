#!/usr/bin/env python3
"""
Build the dual index for hybrid retrieval:
  1. ChromaDB — dense vector store (nomic-embed-text, 768-dim)
  2. BM25 — sparse keyword index (bm25s)
  3. Parent store + card names — JSON files for parent expansion and query routing

Resumable: skips chunks already in ChromaDB, drops ones no longer in the corpus,
and starts over if the embedding model (or prefix scheme) changed.

Uses the same config as the app (MTG_JUDGE_CONFIG / EMBED_MODEL / OLLAMA_HOST), so
an index built on the Mac can be copied to the Pi as long as embed_model matches.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import bm25s
import chromadb
from tqdm import tqdm

sys.path.insert(0, ".")
from src.config import load_config
from src.embeddings import DOC_PREFIX, Embeddings, EmbeddingsUnavailable
from src.retriever import COLLECTION

CORPUS_DIR = Path("corpus")


def _metadata(chunk: dict) -> dict:
    """Chroma metadata must be flat str/int/float/bool."""
    return {
        k: v for k, v in chunk.items()
        if k != "text" and isinstance(v, (str, int, float, bool))
    }


def build_dense_index(chunks: list[dict], emb: Embeddings, index_dir: Path, batch_size: int = 64) -> int:
    """Build ChromaDB dense index. Resumable — skips already-indexed IDs."""
    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    signature = {"embed_model": emb.model, "doc_prefix": DOC_PREFIX}

    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine", **signature})
    meta = col.metadata or {}
    if any(meta.get(k) != v for k, v in signature.items()):
        print(f"  Embedding config changed ({meta.get('embed_model')} → {emb.model}); rebuilding dense index")
        client.delete_collection(COLLECTION)
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", **signature})

    wanted = {c["id"] for c in chunks}
    existing = set(col.get(include=[])["ids"]) if col.count() else set()
    stale = list(existing - wanted)
    for i in range(0, len(stale), 5000):
        col.delete(ids=stale[i : i + 5000])
    existing -= set(stale)

    new_chunks = [c for c in chunks if c["id"] not in existing]
    print(f"  ChromaDB: {len(existing)} existing, {len(stale)} stale removed, {len(new_chunks)} to index")
    if not new_chunks:
        print("  ✓ Dense index already complete")
        return len(existing)

    for i in tqdm(range(0, len(new_chunks), batch_size), desc="  Dense", unit="batch"):
        batch = new_chunks[i : i + batch_size]
        try:
            embeddings = emb.embed_batch([c["text"] for c in batch])
        except EmbeddingsUnavailable as e:
            sys.exit(f"\n  ⚠ {e}")
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )

    total = col.count()
    print(f"  ✓ Dense index: {total} chunks total")
    return total


def build_sparse_index(chunks: list[dict], index_dir: Path) -> int:
    """Build BM25 sparse index. Row i ↔ chunk_ids.json[i]."""
    bm25_dir = index_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)

    print(f"  BM25: tokenizing {len(chunks)} chunks...")
    corpus_tokens = bm25s.tokenize([c["text"] for c in chunks], show_progress=False)

    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens, show_progress=False)
    bm25.save(str(bm25_dir))

    (index_dir / "chunk_ids.json").write_text(json.dumps([c["id"] for c in chunks]))
    print(f"  ✓ BM25 index: {len(chunks)} chunks")
    return len(chunks)


def copy_lookup_files(index_dir: Path) -> int:
    """Copy parents.jsonl (+ card_names.json if present) into the index."""
    shutil.copy(CORPUS_DIR / "parents.jsonl", index_dir / "parents.jsonl")
    if (CORPUS_DIR / "card_names.json").exists():
        shutil.copy(CORPUS_DIR / "card_names.json", index_dir / "card_names.json")
    count = sum(1 for _ in open(index_dir / "parents.jsonl"))
    print(f"  ✓ Parent store: {count} parent documents")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build the hybrid retrieval index")
    parser.add_argument("--config", help="Config profile (default: $MTG_JUDGE_CONFIG or config/default.yaml)")
    args = parser.parse_args()
    config = load_config(args.config)
    index_dir = Path(config["index_path"])
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = CORPUS_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit("✗ corpus/chunks.jsonl not found. Run `make corpus` first.")

    chunks = [json.loads(line) for line in open(chunks_path)]
    print(f"Building index for {len(chunks)} chunks → {index_dir}/\n")

    print(f"1. Building dense index (ChromaDB + {config['embed_model']})...")
    emb = Embeddings(model=config["embed_model"], base_url=config["ollama_host"])
    dense_count = build_dense_index(chunks, emb, index_dir)

    print("\n2. Building sparse index (BM25)...")
    sparse_count = build_sparse_index(chunks, index_dir)

    print("\n3. Copying parent store + card names...")
    parent_count = copy_lookup_files(index_dir)

    print(f"\n{'='*50}")
    print("✅ Index built successfully.")
    print(f"   Dense (ChromaDB): {dense_count} chunks")
    print(f"   Sparse (BM25): {sparse_count} chunks")
    print(f"   Parents: {parent_count} documents")
    print(f"   Location: {index_dir}/")
    print("\n   Next: `make run` to start the judge.")


if __name__ == "__main__":
    main()
