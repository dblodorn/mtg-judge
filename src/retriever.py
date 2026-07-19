"""Hybrid retriever — BM25 + Dense → RRF → parent expansion → MMR."""

import json
import sys
from pathlib import Path

import bm25s
import chromadb
import numpy as np

sys.path.insert(0, ".")
from src.embeddings import Embeddings


class HybridRetriever:
    """Combines dense (ChromaDB) and sparse (BM25) retrieval with:
    - Reciprocal Rank Fusion (RRF) to merge results
    - Parent document expansion (child match → return full section)
    - Maximal Marginal Relevance (MMR) for diversity
    """

    def __init__(
        self,
        index_path: str = "index/",
        k: int = 8,
        rrf_k: int = 60,
        mmr_lambda: float = 0.5,
    ):
        self.emb = Embeddings()
        self.k = k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda

        # Dense store
        chroma_path = Path(index_path) / "chroma"
        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        self.col = self.chroma.get_or_create_collection("mtg_judge")

        # Sparse store
        self.bm25 = bm25s.BM25()
        bm25_path = Path(index_path) / "bm25"
        self.bm25.load(str(bm25_path), load_corpus=True)
        self.chunk_ids: list[str] = json.loads(
            (Path(index_path) / "chunk_ids.json").read_text()
        )

        # Parent store
        self.parents: dict[str, dict] = {}
        parents_path = Path(index_path) / "parents.jsonl"
        for line in open(parents_path):
            p = json.loads(line)
            self.parents[p["id"]] = p

    def retrieve(
        self, query: str, sources: list[str] | None = None
    ) -> list[dict]:
        """Full hybrid pipeline: dense + sparse → RRF → parent expansion → MMR."""
        # 1. Dense retrieval (2x candidates for fusion)
        dense_results = self._dense_search(query, n=self.k * 3, sources=sources)

        # 2. Sparse retrieval (BM25)
        sparse_results = self._bm25_search(query, n=self.k * 3, sources=sources)

        # 3. RRF fusion
        fused = self._rrf_fuse(dense_results, sparse_results)

        # 4. Parent expansion + dedup
        expanded = self._expand_parents(fused[: self.k * 2])

        # 5. MMR diversification (skip if we have <= k results)
        if len(expanded) <= self.k:
            return expanded
        final = self._mmr(query, expanded, k=self.k)
        return final

    def _dense_search(self, query: str, n: int, sources: list[str] | None) -> list[dict]:
        """Query ChromaDB for top-n dense matches."""
        try:
            embedding = self.emb.embed(query)
        except Exception:
            return []

        where = {"source": {"$in": sources}} if sources else None
        try:
            res = self.col.query(
                query_embeddings=[embedding], n_results=n, where=where
            )
        except Exception:
            return []

        results = []
        for i in range(len(res["ids"][0])):
            results.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "score": 1.0 - (i / max(n, 1)),  # rank-based score
                **res["metadatas"][0][i],
            })
        return results

    def _bm25_search(self, query: str, n: int, sources: list[str] | None) -> list[dict]:
        """Query BM25 for top-n sparse matches."""
        tokens = bm25s.tokenize(query, show_progress=False)
        scores, indices = self.bm25.retrieve(tokens, k=n)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]

            # Get chunk metadata from ChromaDB
            try:
                chunk_data = self.col.get(ids=[chunk_id])
                if not chunk_data["ids"]:
                    continue
                meta = chunk_data["metadatas"][0]
            except Exception:
                continue

            if sources and meta.get("source") not in sources:
                continue

            results.append({
                "id": chunk_id,
                "text": chunk_data["documents"][0],
                "score": float(score),
                **meta,
            })
        return results

    def _rrf_fuse(self, dense: list[dict], sparse: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(rank + rrf_k)."""
        scores: dict[str, float] = {}

        for rank, doc in enumerate(dense):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (rank + self.rrf_k)
        for rank, doc in enumerate(sparse):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (rank + self.rrf_k)

        # Map back to documents (prefer dense for text/metadata)
        doc_map = {d["id"]: d for d in dense}
        doc_map.update({d["id"]: d for d in sparse})

        fused = []
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            doc = doc_map[doc_id].copy()
            doc["fused_score"] = score
            fused.append(doc)
        return fused

    def _expand_parents(self, results: list[dict]) -> list[dict]:
        """Replace child text with parent document text. Deduplicate by parent_id."""
        seen_parents: dict[str, dict] = {}
        for doc in results:
            parent_id = doc.get("parent_id", doc["id"])

            if parent_id in self.parents:
                doc["child_id"] = doc["id"]
                doc["id"] = parent_id
                doc["text"] = self.parents[parent_id]["text"]
                doc["parent_section"] = self.parents[parent_id].get("section", "")

            if parent_id not in seen_parents or doc["fused_score"] > seen_parents[parent_id]["fused_score"]:
                seen_parents[parent_id] = doc

        return list(seen_parents.values())

    def _mmr(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        """Maximal Marginal Relevance: balance relevance with diversity.

        λ=1.0 → pure relevance. λ=0.0 → max diversity. Default 0.5.
        """
        if len(candidates) <= k:
            return candidates

        # Embed query and candidate texts
        query_emb = np.array(self.emb.embed(query))
        doc_embs = np.array([self.emb.embed(c["text"]) for c in candidates])

        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

        # Relevance scores (cosine sim with query)
        relevance = doc_norms @ query_norm

        selected = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            best_idx, best_score = None, -float("inf")
            for i in remaining:
                if selected:
                    max_sim = max(doc_norms[i] @ doc_norms[j] for j in selected)
                else:
                    max_sim = 0
                mmr_score = self.mmr_lambda * relevance[i] - (1 - self.mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]
