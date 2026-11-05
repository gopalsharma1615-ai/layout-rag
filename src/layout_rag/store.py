"""Vector stores: an exact in-memory/numpy store (saved to disk) and Milvus (Lite file or server).

Both accept equality filters on chunk metadata (`{"doc_name": "a.pdf"}`) applied BEFORE ranking, which is what
makes per-document scoping cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]: ...
    def all_chunks(self) -> list[Chunk]: ...


def _match(c: Chunk, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, val in filters.items():
        have = getattr(c, key, None) if hasattr(c, key) else c.metadata.get(key)
        if isinstance(val, (list, tuple, set)):
            if have not in val:
                return False
        elif have != val:
            return False
    return True


class MemoryStore:
    """Exact cosine search over normalized vectors. Fine up to a few hundred thousand chunks on CPU."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vecs = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.vecs = vectors if self.vecs.size == 0 else np.vstack([self.vecs, vectors])
        self.chunks += chunks

    def search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        idx = np.array([i for i, c in enumerate(self.chunks) if _match(c, filters)], dtype=int)
        if idx.size == 0:
            return []
        scores = self.vecs[idx] @ qvec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[idx[i]], float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vecs)
        with open(p / "chunks.jsonl", "w") as f:
            f.writelines(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in self.chunks)

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        p = Path(path)
        s = cls()
        s.vecs = np.load(p / "vectors.npy")
        with open(p / "chunks.jsonl") as f:
            s.chunks = [Chunk(**json.loads(line)) for line in f]
        return s


class MilvusStore:
    """pymilvus MilvusClient. `uri` is a local file for Milvus Lite ("./rag.db") or a server URL; credentials
    come from the MILVUS_TOKEN environment variable, never from code."""

    def __init__(self, uri: str, collection: str, dim: int, recreate: bool = False):
        import os

        from pymilvus import DataType, MilvusClient

        self.client = MilvusClient(uri=uri, token=os.environ.get("MILVUS_TOKEN", ""))
        self.col = collection
        if recreate and self.client.has_collection(collection):
            self.client.drop_collection(collection)
        if not self.client.has_collection(collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            idx = self.client.prepare_index_params()
            idx.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(collection, schema=schema, index_params=idx)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        rows = []
        for c, v in zip(chunks, vectors):
            rows.append({"chunk_id": c.chunk_id, "vector": v.tolist(), "doc_name": c.doc_name, "text": c.text,
                         "payload": json.dumps(c.to_dict(), ensure_ascii=False)})
        for i in range(0, len(rows), 256):
            self.client.insert(self.col, rows[i:i + 256])

    def search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        expr = " and ".join(f'{key} == {json.dumps(val)}' for key, val in (filters or {}).items()) or ""
        res = self.client.search(self.col, data=[qvec.tolist()], limit=k, filter=expr, output_fields=["payload"])
        return [(Chunk(**json.loads(h["entity"]["payload"])), float(h["distance"])) for h in res[0]]

    def all_chunks(self) -> list[Chunk]:
        out, it = [], self.client.query_iterator(self.col, batch_size=1000, output_fields=["payload"])
        while batch := it.next():
            out += [Chunk(**json.loads(r["payload"])) for r in batch]
        it.close()
        return out
