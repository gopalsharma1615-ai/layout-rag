"""layout-rag command line.

  layout-rag chunk  doc.pdf -o chunks.jsonl [--stages out/]       inspect chunking (and every stage)
  layout-rag ingest docs/*.pdf --index ./index                      chunk + embed + store
  layout-rag query  "question" --index ./index [--mode hybrid] [--rerank]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

from .config import ChunkerConfig


def _cfg(a) -> ChunkerConfig:
    return ChunkerConfig(text_source=a.text_source, layout=a.layout, max_tokens=a.max_tokens,
                         min_tokens=a.min_tokens, tokenizer=None if a.tokenizer == "none" else a.tokenizer)


def _add_chunk_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--text-source", default="auto", choices=["auto", "pdf", "ocr"])
    p.add_argument("--layout", default="auto", choices=["auto", "yolo", "heuristic"])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=300)
    p.add_argument("--tokenizer", default="BAAI/bge-large-en-v1.5", help="HF tokenizer name, or 'none'")


def cmd_chunk(a) -> int:
    from .pipeline import DocLayoutChunker

    ch = DocLayoutChunker(_cfg(a))
    with (open(a.output, "w") if a.output else contextlib.nullcontext(sys.stdout)) as out:
        for pdf in a.pdfs:
            r = ch.chunk_pdf(pdf, out_dir=a.stages and Path(a.stages) / Path(pdf).stem)
            for c in r.chunks:
                out.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            print(f"{pdf}: {json.dumps(r.stats)}", file=sys.stderr)
    return 0


def cmd_ingest(a) -> int:
    from .embed import Embedder
    from .pipeline import DocLayoutChunker
    from .store import MemoryStore, MilvusStore

    emb = Embedder(a.embedder)
    ch = DocLayoutChunker(_cfg(a))
    store = (MilvusStore(a.milvus, a.collection, emb.dim, recreate=a.recreate) if a.milvus
             else (MemoryStore.load(a.index) if (Path(a.index) / "vectors.npy").exists() and not a.recreate
                   else MemoryStore()))
    for pdf in a.pdfs:
        r = ch.chunk_pdf(pdf)
        store.add(r.chunks, emb.embed_passages([c.text for c in r.chunks]))
        print(f"{pdf}: {r.stats['chunks']} chunks, coverage {r.stats['coverage']}", file=sys.stderr)
    if not a.milvus:
        store.save(a.index)
        (Path(a.index) / "index.json").write_text(json.dumps({"embedder": a.embedder}))
    return 0


def cmd_query(a) -> int:
    from .embed import Embedder
    from .retrieve import CrossEncoderReranker, Retriever
    from .store import MemoryStore, MilvusStore

    meta = json.loads((Path(a.index) / "index.json").read_text()) if not a.milvus else {}
    emb = Embedder(meta.get("embedder", a.embedder))
    store = MilvusStore(a.milvus, a.collection, emb.dim) if a.milvus else MemoryStore.load(a.index)
    rr = CrossEncoderReranker(a.reranker) if a.rerank else None
    hits = Retriever(store, emb, reranker=rr).search(a.question, k=a.k, mode=a.mode, pool=a.pool,
                                                     filters={"doc_name": a.doc} if a.doc else None)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.chunk.doc_name} p{','.join(str(p + 1) for p in h.chunk.pages)}  score={h.score:.4f}  "
              f"{h.ranks}\n    {h.chunk.text[:300].replace(chr(10), ' / ')}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="layout-rag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chunk", help="chunk PDFs to JSONL")
    c.add_argument("pdfs", nargs="+")
    c.add_argument("-o", "--output")
    c.add_argument("--stages", help="directory for per-stage audit files")
    _add_chunk_args(c)
    c.set_defaults(fn=cmd_chunk)

    for name, fn in (("ingest", cmd_ingest), ("query", cmd_query)):
        s = sub.add_parser(name)
        if name == "ingest":
            s.add_argument("pdfs", nargs="+")
            s.add_argument("--recreate", action="store_true")
            _add_chunk_args(s)
        else:
            s.add_argument("question")
            s.add_argument("-k", type=int, default=5)
            s.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
            s.add_argument("--pool", type=int, default=50)
            s.add_argument("--rerank", action="store_true")
            s.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
            s.add_argument("--doc", help="restrict to one document name")
        s.add_argument("--index", default="./index")
        s.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
        s.add_argument("--milvus", help="Milvus URI (./rag.db for Milvus Lite, or http://host:19530)")
        s.add_argument("--collection", default="layout_rag")
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
