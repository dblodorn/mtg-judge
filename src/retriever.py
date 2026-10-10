"""Hybrid retriever — BM25 + Dense → RRF → parent expansion → MMR."""

import json
import re
import sys
from pathlib import Path

import bm25s
import chromadb
import numpy as np

sys.path.insert(0, ".")
from src.embeddings import Embeddings
from src.formats import display_name

COLLECTION = "mtg_judge"


class IndexMissing(Exception):
    """Raised when the on-disk index hasn't been built."""


class HybridRetriever:
    """Combines dense (ChromaDB) and sparse (BM25) retrieval with:
    - Direct lookups for exact rule numbers and named cards (pinned first)
    - Reciprocal Rank Fusion (RRF) to merge results
    - Parent document expansion (child match → surrounding section, trimmed)
    - Maximal Marginal Relevance (MMR) for diversity
    """

    # Class defaults so tests can build a bare instance via __new__.
    max_parent_chars = 1500
    max_context_chars = 6000
    rulings_per_card = 2
    card_ids: dict[str, str] = {}
    _warned_dense = False

    def __init__(
        self,
        index_path: str = "index/",
        k: int = 8,
        rrf_k: int = 60,
        mmr_lambda: float = 0.5,
        embed_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        max_parent_chars: int = 1500,
        max_context_chars: int = 6000,
    ):
        index = Path(index_path)
        for required in ("chroma", "bm25", "chunk_ids.json", "parents.jsonl"):
            if not (index / required).exists():
                raise IndexMissing(
                    f"{index / required} not found. Build the index with `make corpus && make index`."
                )

        self.emb = Embeddings(model=embed_model, base_url=ollama_host)
        self.k = k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.max_parent_chars = max_parent_chars
        self.max_context_chars = max_context_chars

        # Dense store
        self.chroma = chromadb.PersistentClient(path=str(index / "chroma"))
        self.col = self.chroma.get_collection(COLLECTION)

        # Sparse store
        self.bm25 = bm25s.BM25.load(str(index / "bm25"))
        self.chunk_ids: list[str] = json.loads((index / "chunk_ids.json").read_text())
        self.chunk_sources = np.array([self._source_of(cid) for cid in self.chunk_ids])

        # Parent store (+ card name → chunk id for direct card lookups)
        self.parents: dict[str, dict] = {}
        self.card_ids = {}
        with open(index / "parents.jsonl") as f:
            for line in f:
                p = json.loads(line)
                self.parents[p["id"]] = p
                if p.get("source") == "scryfall" and p.get("name"):
                    self.card_ids[p["name"].lower()] = p["id"]

    def retrieve(
        self, query: str, sources: list[str] | None = None, route: dict | None = None
    ) -> list[dict]:
        """Full hybrid pipeline. `route` (from QueryRouter) enables direct lookups."""
        if route is not None:
            sources = route.get("sources")
        query_emb = self._embed_query(query)
        pinned = self._direct_lookups(route or {}, query_emb)

        # 1–2. Dense + sparse search (3x candidates for fusion). When cards are named,
        # also search with each name swapped for what the card *is* ("Urza's Saga" →
        # "Enchantment Land — Urza's Saga: … lore counter … Sacrifice after III"), which
        # finds Saga/land-type rules instead of rules that happen to mention "Urza".
        queries = [(query, query_emb)]
        search_query = self._expand_card_names(query, route or {})
        if search_query != query:
            queries.append((search_query, self._embed_query(search_query)))
        rankings = []
        for q, emb in queries:
            rankings.append(self._dense_search(emb, n=self.k * 3, sources=sources))
            rankings.append(self._bm25_search(q, n=self.k * 3, sources=sources))

        # 3. RRF fusion across every ranking
        fused = self._rrf_fuse(*rankings)

        # 4. Parent expansion + dedup (pinned docs win)
        pinned_ids = {d["id"] for d in pinned}
        expanded = [
            d for d in self._expand_parents(fused[: self.k * 2]) if d["id"] not in pinned_ids
        ]

        # 5. MMR diversification over the remaining slots
        slots = max(self.k - len(pinned), 0)
        if len(expanded) > slots:
            expanded = self._mmr(query, expanded, k=slots, query_emb=query_emb)

        return self._cap_context(pinned + expanded)

    # ------------------------------------------------------------------
    # Direct lookups
    # ------------------------------------------------------------------

    def _direct_lookups(self, route: dict, query_emb=None) -> list[dict]:
        """Exact rule numbers and mentioned cards, fetched straight from the parent store."""
        docs = []
        for num in route.get("rule_numbers", []):
            child_id = f"cr:{num}"
            parent_id = self._parent_of(num)
            parent = self.parents.get(parent_id)
            if parent:
                docs.append({
                    **{k: v for k, v in parent.items() if k != "children"},
                    "child_id": child_id,
                    "text": self._trim_parent(parent["text"], f"{num} "),
                    "parent_section": parent.get("section", ""),
                    "fused_score": 1.0,
                })
        for name in route.get("mentioned_cards", []):
            card_id = self.card_ids.get(name.lower())
            if card_id and card_id in self.parents:
                card = self.parents[card_id]
                docs.append({
                    **{k: v for k, v in card.items() if k != "children"},
                    "text": self._card_with_rulings(card, query_emb) + self._legality_line(card, route.get("format")),
                    "fused_score": 1.0,
                })

        seen, unique = set(), []
        for d in docs:
            if d["id"] not in seen:
                seen.add(d["id"])
                unique.append(d)
        return unique

    def _expand_card_names(self, query: str, route: dict, max_chars: int = 200) -> str:
        """Replace mentioned card names with their type line + rules text (no rulings)."""
        for name in route.get("mentioned_cards", []):
            card = self.parents.get(self.card_ids.get(name.lower(), ""))
            if not card:
                continue
            # Rules text = card text minus its "Name — cost — type" header line and rulings
            card_text = card["text"].partition("\nRulings:\n")[0]
            rules = card_text.partition("\n")[2]
            summary = f"({card.get('type', '')}: {rules})".replace("\n", " ")[:max_chars]
            query = re.sub(rf"\b{re.escape(name)}(['’]s)?", summary, query, flags=re.IGNORECASE)
        return query

    @staticmethod
    def _legality_line(card: dict, game_format: str | None) -> str:
        """'\nModern legality: banned' for the format being played (from Scryfall)."""
        status = (card.get("legalities") or {}).get(game_format or "")
        if not status:
            return ""
        return f"\n{display_name(game_format)} legality: {status.replace('_', ' ')}"

    def _card_with_rulings(self, card: dict, query_emb) -> str:
        """Card text plus its `rulings_per_card` rulings most similar to the question.

        Card parents carry every official ruling (Urza's Saga has 5, some cards 20+);
        sending them all would eat the whole context budget on a Pi.
        """
        card_text, _, rulings_block = card["text"].partition("\nRulings:\n")
        rulings = rulings_block.split("\n") if rulings_block else []
        if len(rulings) > self.rulings_per_card:
            ids = [f"{card['id']}:r{i}" for i in range(len(rulings))]
            embs = self._stored_embeddings([{"id": i} for i in ids]) if query_emb is not None else None
            if embs is not None:
                q = np.array(query_emb, dtype=float)
                scores = embs @ q / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
                keep = sorted(np.argsort(-scores)[: self.rulings_per_card])
            else:
                keep = range(self.rulings_per_card)
            rulings = [rulings[i] for i in keep]
        text = card_text + ("\nRulings:\n" + "\n".join(rulings) if rulings else "")
        return text[: self.max_parent_chars]

    @staticmethod
    def _parent_of(rule_num: str) -> str:
        """702.16a → cr:702.16"""
        base = rule_num.rstrip("abcdefghijklmnopqrstuvwxyz")
        return f"cr:{base}"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float] | None:
        try:
            return self.emb.embed(query)
        except Exception as e:
            if not self._warned_dense:
                print(f"⚠ Dense retrieval unavailable ({e}); falling back to BM25 only.")
                self._warned_dense = True
            return None

    def _dense_search(self, query_emb, n: int, sources: list[str] | None) -> list[dict]:
        """Query ChromaDB for top-n dense matches."""
        if query_emb is None:
            return []
        where = {"source": {"$in": sources}} if sources else None
        res = self.col.query(query_embeddings=[query_emb], n_results=n, where=where)

        results = []
        for i in range(len(res["ids"][0])):
            results.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "score": 1.0 - (i / max(n, 1)),  # rank-based score
                **res["metadatas"][0][i],
            })
        return results

    @staticmethod
    def _source_of(chunk_id: str) -> str:
        """Source from the ID scheme: "cr:702.16a", "ipg:2.1:3", "scryfall:<oid>", "scryfall:<oid>:r2"."""
        prefix = chunk_id.split(":", 1)[0]
        if prefix == "scryfall" and chunk_id.count(":") == 2:
            return "ruling"
        return prefix

    def _bm25_search(self, query: str, n: int, sources: list[str] | None) -> list[dict]:
        """Query BM25 for top-n sparse matches within `sources`.

        Filtering happens inside the search (weight_mask). Post-filtering a global
        top-n doesn't work here: ~110k card/ruling chunks crowd every rules chunk out.
        """
        tokens = bm25s.tokenize(query, show_progress=False)
        mask = np.isin(self.chunk_sources, sources).astype(np.float32) if sources else None
        indices, scores = self.bm25.retrieve(
            tokens, k=min(n, len(self.chunk_ids)), show_progress=False, weight_mask=mask
        )

        ranked = [
            (self.chunk_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if 0 <= idx < len(self.chunk_ids) and score > 0
        ]
        if not ranked:
            return []

        # One batched metadata fetch instead of one round trip per hit
        got = self.col.get(ids=[cid for cid, _ in ranked], include=["documents", "metadatas"])
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=True)}

        return [
            {"id": chunk_id, "text": by_id[chunk_id][0], "score": score, **by_id[chunk_id][1]}
            for chunk_id, score in ranked
            if chunk_id in by_id
        ]

    def _rrf_fuse(self, *rankings: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(rank + rrf_k) over every ranking."""
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                scores[doc["id"]] = scores.get(doc["id"], 0) + 1.0 / (rank + self.rrf_k)
                doc_map[doc["id"]] = doc

        fused = []
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            doc = doc_map[doc_id].copy()
            doc["fused_score"] = score
            fused.append(doc)
        return fused

    # ------------------------------------------------------------------
    # Expansion + trimming
    # ------------------------------------------------------------------

    def _expand_parents(self, results: list[dict]) -> list[dict]:
        """Replace child text with (trimmed) parent text. Deduplicate by parent_id."""
        seen_parents: dict[str, dict] = {}
        for doc in results:
            parent_id = doc.get("parent_id", doc["id"])

            if parent_id in self.parents:
                parent = self.parents[parent_id]
                doc["child_id"] = doc["id"]
                doc["id"] = parent_id
                doc["text"] = self._trim_parent(parent["text"], doc.get("anchor", ""))
                doc["parent_section"] = parent.get("section", "")

            if parent_id not in seen_parents or doc["fused_score"] > seen_parents[parent_id]["fused_score"]:
                seen_parents[parent_id] = doc

        return list(seen_parents.values())

    def _trim_parent(self, text: str, anchor: str, window: int = 2) -> str:
        """Keep a long section small: heading + the matched line ± `window` lines.

        `anchor` is the child's line prefix ("702.19f ") or opening text (MTR/IPG
        paragraphs). Sections like CR 702.19 (trample) run to thousands of
        characters; sending all of them dominates prompt-processing time on a Pi.
        """
        if len(text) <= self.max_parent_chars:
            return text
        lines = text.split("\n")
        hit = 0
        if anchor:
            hit = next(
                (i for i, line in enumerate(lines) if line.startswith(anchor)),
                next((i for i, line in enumerate(lines) if anchor in line), 0),
            )
        keep = sorted({0, *range(max(hit - window, 0), min(hit + window + 1, len(lines)))})
        trimmed = "\n".join(lines[i] for i in keep)
        return trimmed[: self.max_parent_chars]

    def _cap_context(self, docs: list[dict]) -> list[dict]:
        """Stop adding docs once the total context budget is spent (always keep one)."""
        out, total = [], 0
        for d in docs[: max(self.k, 1)]:
            if out and total + len(d["text"]) > self.max_context_chars:
                break
            out.append(d)
            total += len(d["text"])
        return out

    # ------------------------------------------------------------------
    # MMR
    # ------------------------------------------------------------------

    def _stored_embeddings(self, candidates: list[dict]) -> np.ndarray | None:
        """Fetch the child-chunk embeddings already in Chroma (no re-embedding)."""
        ids = [c.get("child_id", c["id"]) for c in candidates]
        got = self.col.get(ids=ids, include=["embeddings"])
        by_id = dict(zip(got["ids"], got["embeddings"], strict=True))
        if any(i not in by_id for i in ids):
            return None
        return np.array([by_id[i] for i in ids], dtype=float)

    def _mmr(self, query: str, candidates: list[dict], k: int, query_emb=None) -> list[dict]:
        """Maximal Marginal Relevance: balance relevance with diversity.

        Relevance is the RRF fused score (so BM25's exact-term evidence counts), min-max
        normalized; stored embeddings only measure redundancy between candidates.
        λ=1.0 → pure relevance. λ=0.0 → max diversity. Default 0.5.
        Without embeddings (Ollama down) falls back to fused-score order.
        """
        if len(candidates) <= k:
            return candidates
        if query_emb is None:
            query_emb = self._embed_query(query)
        doc_embs = self._stored_embeddings(candidates) if query_emb is not None else None
        if doc_embs is None:
            return sorted(candidates, key=lambda c: -c.get("fused_score", 0))[:k]

        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        sim = doc_norms @ doc_norms.T

        fused = np.array([c.get("fused_score", 0.0) for c in candidates], dtype=float)
        spread = fused.max() - fused.min()
        relevance = (fused - fused.min()) / spread if spread > 0 else np.ones_like(fused)

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        for _ in range(k):
            best_idx, best_score = None, -float("inf")
            for i in remaining:
                max_sim = max((sim[i, j] for j in selected), default=0.0)
                mmr_score = self.mmr_lambda * relevance[i] - (1 - self.mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score, best_idx = mmr_score, i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]
