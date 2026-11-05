"""Dense, BM25 and hybrid (reciprocal-rank fusion) retrieval, with an optional local cross-encoder rerank."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import Chunk

_TOK = re.compile(r"[a-z0-9]+")


def _toks(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # rank per retriever (1-based), for diagnosis


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([_toks(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            return []
        from .store import _match

        scores = self.bm25.get_scores(_toks(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if _match(self.chunks[i], filters):
                out.append((self.chunks[i], float(scores[i])))
                if len(out) == k:
                    break
        return out


class CrossEncoderReranker:
    """Local cross-encoder, default BAAI/bge-reranker-v2-m3 (sentence-transformers CrossEncoder)."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", max_length: int = 512,
                 batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=self.batch_size)]


def rrf(rankings: dict[str, list[Chunk]], k: int = 60) -> list[Hit]:
    fused: dict[str, Hit] = {}
    for name, ranked in rankings.items():
        for r, c in enumerate(ranked, start=1):
            h = fused.setdefault(c.chunk_id, Hit(c, 0.0))
            h.score += 1.0 / (k + r)
            h.ranks[name] = r
    return sorted(fused.values(), key=lambda h: -h.score)


class Retriever:
    def __init__(self, store, embedder, bm25: BM25Index | None = None, reranker: CrossEncoderReranker | None = None):
        self.store, self.embedder, self.bm25, self.reranker = store, embedder, bm25, reranker

    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 50, rerank: bool | None = None,
               filters: dict | None = None) -> list[Hit]:
        """mode: dense | bm25 | hybrid. `pool` candidates per retriever feed fusion and the reranker."""
        rankings: dict[str, list[Chunk]] = {}
        if mode in ("dense", "hybrid"):
            q = self.embedder.embed_queries([query])[0]
            rankings["dense"] = [c for c, _ in self.store.search(q, pool, filters)]
        if mode in ("bm25", "hybrid"):
            if self.bm25 is None:
                self.bm25 = BM25Index(self.store.all_chunks())
            rankings["bm25"] = [c for c, _ in self.bm25.search(query, pool, filters)]
        if not rankings:
            raise ValueError(f"unknown mode {mode!r}")
        hits = rrf(rankings) if len(rankings) > 1 else [Hit(c, 1.0 / r, {mode: r}) for r, c in
                                                         enumerate(next(iter(rankings.values())), start=1)]
        use_rerank = self.reranker is not None if rerank is None else rerank
        if use_rerank and self.reranker is not None and hits:
            cand = hits[:pool]
            scores = self.reranker.score(query, [h.chunk.text for h in cand])
            for r, (h, s) in enumerate(zip(cand, scores), start=1):
                h.ranks["fused"] = r
                h.score = s
            hits = sorted(cand, key=lambda h: -h.score)
        return hits[:k]
