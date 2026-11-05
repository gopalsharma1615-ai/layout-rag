"""Local embedders (sentence-transformers). No API calls."""
from __future__ import annotations

import hashlib

import numpy as np

# Query-side instructions some families were trained with. Passages are embedded WITHOUT them.
QUERY_PREFIX = {
    "bge-": "Represent this sentence for searching relevant passages: ",
    "e5-": "query: ",
}
PASSAGE_PREFIX = {"e5-": "passage: "}


def _prefix(table: dict[str, str], name: str) -> str:
    low = name.lower()
    if "bge-m3" in low:          # bge-m3 takes questions without an instruction
        return ""
    return next((p for k, p in table.items() if k in low), "")


class Embedder:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", device: str = "cpu", batch_size: int = 16,
                 max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model, device=device, trust_remote_code="gte" in model.lower())
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.qp, self.pp = _prefix(QUERY_PREFIX, model), _prefix(PASSAGE_PREFIX, model)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc([self.pp + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc([self.qp + t for t in texts])

    def _enc(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return v.astype(np.float32)


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens vectors. For tests and offline demos only - not a real model."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hash-{dim}", dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            tok = "".join(c for c in tok if c.isalnum())
            if tok:
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_passages
