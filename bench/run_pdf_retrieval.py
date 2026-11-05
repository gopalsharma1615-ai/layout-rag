"""PDF -> chunks -> embeddings -> retrieval, scored against gold evidence pages and text.

    python bench/run_pdf_retrieval.py --dataset ohrbench --data-dir data/ohrbench --docs-per-domain 45 \
        --chunker layout --out results/ohrbench_layout
    python bench/run_pdf_retrieval.py --dataset financebench --data-dir data/financebench --limit-docs 17 \
        --chunker recursive --out results/financebench_recursive

Datasets (download them yourself; licences apply):
  ohrbench      qas_v2.json + pdfs/<domain>/<doc>.pdf   (github.com/opendatalab/OHR-Bench)
  financebench  financebench_merged.jsonl + pdfs/<doc>.pdf (github.com/patronus-ai/financebench)

Subsets are WHOLE documents chosen by sha1(doc name) order, so they are reproducible and every question about a
chosen document is asked. Chunks and vectors are cached under --work, keyed by chunker + embedder.
A chunk is relevant when it comes from the gold document AND its pages include a gold evidence page.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import BASELINES
from layout_rag import ChunkerConfig, DocLayoutChunker, MemoryStore, Retriever
from layout_rag.chunker import TokenCounter
from layout_rag.embed import Embedder
from layout_rag.types import Chunk
from metrics import clean_evidence, evidence_recall, first_relevant_rank, hit_at, lcs_ratio, mrr_at, page_relevant

_CJK = re.compile(r"[㐀-鿿]")


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ datasets
def load_ohrbench(d: Path, docs_per_domain: int | None) -> tuple[list[dict], dict[str, Path]]:
    qs = json.loads((d / "qas_v2.json").read_text())
    by_domain: dict[str, set[str]] = defaultdict(set)
    for q in qs:
        by_domain[q["doc_type"]].add(q["doc_name"])
    chosen: set[str] = set()
    for names in by_domain.values():
        ordered = sorted(names, key=_sha)
        chosen |= set(ordered[:docs_per_domain] if docs_per_domain else ordered)
    out, pdfs = [], {}
    for q in qs:
        if q["doc_name"] not in chosen:
            continue
        pdf = d / "pdfs" / f"{q['doc_name']}.pdf"
        pdfs[pdf.name] = pdf
        pages = q["evidence_page_no"]
        pages = pages if isinstance(pages, list) else [pages]
        out.append({"id": q["ID"], "question": q["questions"], "doc": pdf.name, "domain": q["doc_type"],
                    "type": q.get("evidence_source", ""), "pages": {int(p) for p in pages},
                    "evidence": clean_evidence(str(q.get("evidence_context", ""))),
                    "lang": "zh" if _CJK.search(q["questions"]) else "en"})
    return out, pdfs


def load_financebench(d: Path, limit_docs: int | None) -> tuple[list[dict], dict[str, Path]]:
    with open(d / "financebench_merged.jsonl") as f:
        rows = [json.loads(line) for line in f]
    names = sorted({r["doc_name"] for r in rows}, key=_sha)
    chosen = set(names[:limit_docs] if limit_docs else names)
    out, pdfs = [], {}
    for r in rows:
        if r["doc_name"] not in chosen:
            continue
        pdf = d / "pdfs" / f"{r['doc_name']}.pdf"
        if not pdf.exists():
            continue
        pdfs[pdf.name] = pdf
        ev = r["evidence"] if isinstance(r["evidence"], list) else ast.literal_eval(r["evidence"])
        out.append({"id": r["financebench_id"], "question": r["question"], "doc": pdf.name, "domain": "finance",
                    "type": r.get("question_type", ""), "pages": {int(e["evidence_page_num"]) for e in ev},
                    "evidence": " ".join(e.get("evidence_text", "") for e in ev), "lang": "en"})
    return out, pdfs


# ------------------------------------------------------------------------------------------------ indexing
def build_chunks(pdfs: dict[str, Path], chunker: str, tag: str, cfg: ChunkerConfig, work: Path) -> list[Chunk]:
    cache = work / f"chunks_{tag}.jsonl"
    done: dict[str, list[Chunk]] = defaultdict(list)
    if cache.exists():
        with open(cache) as f:
            for line in f:
                c = Chunk(**json.loads(line))
                done[c.doc_name].append(c)
    todo = [p for name, p in sorted(pdfs.items()) if name not in done]
    if todo:
        count = TokenCounter(cfg.tokenizer)
        layout = DocLayoutChunker(cfg, counter=count) if chunker == "layout" else None
        with open(cache, "a") as f:
            for i, pdf in enumerate(todo, 1):
                t0 = time.time()
                try:
                    chunks = (layout.chunk_pdf(pdf).chunks if layout is not None
                              else BASELINES[chunker](pdf, count, cfg.max_tokens))
                except Exception as e:  # noqa: BLE001 - one bad PDF must not stop a multi-hour run
                    print(f"  ! {pdf.name}: {e}", file=sys.stderr)
                    chunks = []
                for c in chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                done[pdf.name] = chunks
                print(f"  [{i}/{len(todo)}] {pdf.name}: {len(chunks)} chunks, {time.time() - t0:.1f}s",
                      file=sys.stderr)
    return [c for name in sorted(pdfs) for c in done.get(name, [])]


def build_store(chunks: list[Chunk], emb: Embedder, work: Path, tag: str) -> MemoryStore:
    vec_path = work / f"vectors_{tag}.npy"
    ids_path = work / f"vectors_{tag}.ids"
    ids = [c.chunk_id for c in chunks]
    if vec_path.exists() and ids_path.exists() and ids_path.read_text().split("\n") == ids:
        vecs = np.load(vec_path)
    else:
        vecs = emb.embed_passages([c.text for c in chunks])
        np.save(vec_path, vecs)
        ids_path.write_text("\n".join(ids))
    store = MemoryStore()
    store.add(chunks, vecs)
    return store


# ------------------------------------------------------------------------------------------------ scoring
def evaluate(questions: list[dict], retriever: Retriever, scope: str, docs_by_domain: dict[str, list[str]],
             k: int, mode: str, rerank: bool) -> list[dict]:
    rows = []
    for q in questions:
        if scope == "doc":
            filters = {"doc_name": q["doc"]}
        elif scope == "category":
            filters = {"doc_name": docs_by_domain[q["domain"]]}
        else:
            filters = None
        hits = retriever.search(q["question"], k=max(k, 10), mode=mode, rerank=rerank, filters=filters)
        flags = [page_relevant(h.chunk.doc_name, h.chunk.pages, q["doc"], q["pages"]) for h in hits]
        rank = first_relevant_rank(flags)
        rel_top2 = [h.chunk.text for h, f in zip(hits[:2], flags[:2]) if f]
        rows.append({
            "id": q["id"], "lang": q["lang"], "domain": q["domain"], "type": q["type"], "rank": rank or "",
            "page_hit@1": hit_at(rank, 1), "page_hit@5": hit_at(rank, 5), "page_hit@10": hit_at(rank, 10),
            "mrr@10": mrr_at(rank, 10),
            "doc_hit@5": float(any(h.chunk.doc_name == q["doc"] for h in hits[:5])),
            "evidence_recall@5": evidence_recall(q["evidence"], [h.chunk.text for h in hits[:5]]),
            "lcs@2": max((lcs_ratio(q["evidence"], t) for t in rel_top2), default=0.0),
            "ctx_words@5": sum(len(h.chunk.text.split()) for h in hits[:5]),
        })
    return rows


METRICS = ["page_hit@1", "page_hit@5", "page_hit@10", "mrr@10", "doc_hit@5", "evidence_recall@5", "lcs@2",
           "ctx_words@5"]


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        out[lang] = {"questions": len(sub)}
        for m in METRICS:
            v = statistics.mean(r[m] for r in sub)
            out[lang][m] = round(v if m in ("mrr@10", "ctx_words@5") else 100 * v, 3 if m == "mrr@10" else 1)
        out[lang]["by_type"] = {
            t: round(100 * statistics.mean(r["page_hit@5"] for r in sub if r["type"] == t), 1)
            for t in sorted({r["type"] for r in sub})}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ohrbench", "financebench"])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--docs-per-domain", type=int, help="ohrbench: documents per domain (sha1 order)")
    ap.add_argument("--limit-docs", type=int, help="financebench: number of filings (sha1 order)")
    ap.add_argument("--chunker", default="layout", choices=["layout", *BASELINES])
    ap.add_argument("--layout", default="auto", choices=["auto", "yolo", "heuristic"])
    ap.add_argument("--text-source", default="auto", choices=["auto", "pdf", "ocr"])
    ap.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--rerank", action="store_true", help="local bge-reranker-v2-m3 over the top --pool")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--scopes", default=None, help="comma list of doc|category|global")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    loader = load_ohrbench if a.dataset == "ohrbench" else load_financebench
    questions, pdfs = loader(a.data_dir, a.docs_per_domain if a.dataset == "ohrbench" else a.limit_docs)
    print(f"{a.dataset}: {len(pdfs)} PDFs, {len(questions)} questions", file=sys.stderr)
    cfg = ChunkerConfig(text_source=a.text_source, layout=a.layout, tokenizer=a.embedder)
    work = a.work / a.dataset
    work.mkdir(parents=True, exist_ok=True)
    tag = f"layout-{a.layout}-{a.text_source}" if a.chunker == "layout" else a.chunker
    tag += f"_{a.embedder.replace('/', '_')}"
    chunks = build_chunks(pdfs, a.chunker, tag, cfg, work)
    emb = Embedder(a.embedder)
    store = build_store(chunks, emb, work, tag)
    reranker = None
    if a.rerank:
        from layout_rag import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    retriever = Retriever(store, emb, reranker=reranker)
    docs_by_domain: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        if q["doc"] not in docs_by_domain[q["domain"]]:
            docs_by_domain[q["domain"]].append(q["doc"])

    scopes = a.scopes.split(",") if a.scopes else (["category", "global"] if a.dataset == "ohrbench"
                                                   else ["doc", "global"])
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": a.dataset, "chunker": tag, "embedder": a.embedder, "mode": a.mode,
               "rerank": a.rerank, "pdfs": len(pdfs), "chunks": len(chunks),
               "pages_chunked": len({(c.doc_name, p) for c in chunks for p in c.pages}),
               "max_chunk_tokens": max((c.token_count for c in chunks), default=0), "scopes": {}}
    for scope in scopes:
        rows = evaluate(questions, retriever, scope, docs_by_domain, a.k, a.mode, a.rerank)
        with open(a.out / f"per_question_{scope}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary["scopes"][scope] = summarize(rows)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
