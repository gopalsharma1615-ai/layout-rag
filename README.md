# layout-rag

[![CI](https://github.com/gopalsharma1615-ai/layout-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/gopalsharma1615-ai/layout-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue) ![cpu](https://img.shields.io/badge/runs%20on-CPU-success) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

**Layout-aware PDF chunking and hybrid retrieval for RAG: CPU-only, with no LLM in the indexing path.**

Most RAG pipelines extract plain text from PDFs and cut it every *N* characters. That throws away exactly the
structure a retriever needs. `layout-rag` keeps it:

- **Real heading hierarchies, from typography alone.** Sizes and weights are measured *relative to the
  document's body text* and resolved with a stack. Numbering such as `3.2.1` anchors the levels. Scanned pages get
  bold and size measured from the image.
- **Chunks budgeted in the embedder's own tokens**, breadcrumb included. A "512-word" chunk is about 650 BERT
  tokens, and the tail an embedder truncates is never searchable.
- **Tables whose rows keep their headers**: `Revenue | 2024 = 4,830 | 2025 = 5,120`, so each row is retrievable on
  its own.
- **Lossless by construction.** Every content word reaches a chunk, and each run reports its `coverage`.
  Running headers, footers, page numbers and tables of contents are removed with guards that protect short
  documents.
- **Hybrid retrieval**: dense (bge / gte) plus BM25 with reciprocal-rank fusion, then an optional **local**
  cross-encoder rerank (`bge-reranker-v2-m3`). No API keys anywhere.

```
PDF ─► words (text layer | Tesseract + image typography) ─► DocLayNet YOLO blocks ─► XY-cut reading order
    ─► furniture / TOC removal ─► run-in heading split ─► heading detection ─► stack-resolved levels
    ─► sections ─► token-budgeted chunks (+ table chunks) ─► embed ─► Milvus Lite | in-memory ─► hybrid + rerank
```

## Quickstart

```bash
pip install "layout-rag[all] @ git+https://github.com/gopalsharma1615-ai/layout-rag"
# OCR needs the tesseract binary: apt-get install tesseract-ocr

layout-rag chunk report.pdf -o chunks.jsonl --stages stages/   # inspect: one audit file per pipeline stage
layout-rag ingest docs/*.pdf --index ./index                     # chunk + embed (bge-large) + store
layout-rag query "What was free cash flow in 2018?" --index ./index --mode hybrid --rerank
```

```python
from layout_rag import ChunkerConfig, DocLayoutChunker, MemoryStore, Retriever, CrossEncoderReranker
from layout_rag.embed import Embedder

result = DocLayoutChunker(ChunkerConfig(max_tokens=512)).chunk_pdf("report.pdf")
print(result.title, result.stats)          # coverage, over_budget_chunks, headings, furniture dropped, ...

emb = Embedder("BAAI/bge-large-en-v1.5")
store = MemoryStore()
store.add(result.chunks, emb.embed_passages([c.text for c in result.chunks]))
hits = Retriever(store, emb, reranker=CrossEncoderReranker()).search("free cash flow", k=5, mode="hybrid")
for h in hits:
    print(h.chunk.heading_path, h.chunk.pages, h.ranks)   # ranks per retriever: dense / bm25 / fused
```

Each chunk carries filterable metadata: `doc_name`, `pages`, `section_id` (a `LIKE '3.2.%'` query selects a
subtree), `h1…hN` sized to the document's own depth, `part` (`2/4` when a section was split),
`absorbed_headings`, and prev/next chunk ids. `MilvusStore("./rag.db", ...)` uses Milvus Lite (a local file)
or a server URI; credentials come from `MILVUS_TOKEN`.

## Results

Retrieval quality on public PDF benchmarks, scored against each benchmark's **gold evidence page**. A chunk is
relevant when it comes from the right document and spans the evidence page. No LLM judge is involved.

> **Provenance.** These numbers come from the reference evaluation run of 2026-09-21, made with the original
> implementation of this design (Tesseract OCR, Table Transformer tables, bge-large-en-v1.5, 512/300-token
> budget). This repository is a clean re-implementation. [`bench/`](bench/) reproduces the protocol on the
> same sha1-ordered subsets, and its output is the way to check parity. The table extractor here is a
> geometric serializer, not Table Transformer.

### OHR-Bench (English), PDF → OCR → chunks → retrieval

313 PDFs / 1,120 pages / 1,298 questions across 7 domains, searching within the question's domain (the
OHR-Bench protocol). The **control** row uses the benchmark's ground-truth page text with the same chunk
budget, so it is the ceiling for any OCR-based pipeline.

| pipeline | page hit@1 | page hit@5 | page hit@10 | MRR@10 | evidence recall@5 | context words@5 |
|---|---|---|---|---|---|---|
| **layout pipeline** (OCR + layout + 512/300) | 65.0 | **85.5** | 89.8 | 0.735 | 70.7 | **1,069** |
| ground-truth text control | 73.7 | 95.5 | 98.4 | 0.831 | 87.7 | 1,584 |

Searching all 7 domains at once instead of one costs about 1 point (page hit@5 84.5 vs 85.5).

By evidence type (page hit@5): text 81.4, **table 88.0**, formula 89.1, chart 88.5, reading order 100.0.
Table and formula questions find their page at least as often as running-text questions do, which fits tables
being kept whole inside one chunk.

By domain (page hit@5): administration 94.8 · law 93.6 · news 93.1 · academic 90.8 · finance 85.2 ·
manual 75.2 · textbook 69.8. Manuals and textbooks hold most of the gap to the control (98.2 and 95.9), and
only 81% of their evidence pages reached any chunk, so chunking loss rather than ranking dominates there.

### FinanceBench, long SEC filings

17 filings / 2,187 pages / 65 questions.

| scope | page hit@1 | page hit@5 | page hit@10 | MRR@10 | doc hit@5 | evidence recall@5 |
|---|---|---|---|---|---|---|
| search within the question's filing | 27.7 | 46.2 | 63.1 | 0.371 | 100.0 | 53.1 |
| search all filings | 18.5 | 38.5 | 52.3 | 0.277 | 80.0 | 42.4 |

FinanceBench is hard by design. A single 10-K often repeats near-identical numbers across the MD&A, the
statements and the notes, so "the right page" can be a needle among lookalikes. Metrics-generated questions reach
page hit@5 100.0, while domain-relevant ones reach 30.6.

### What motivated the token-exact budget

On the Chroma chunking benchmark (472 questions, character-span gold), a chunker that budgets **512 whitespace
words** produced chunks of which **71.8% exceeded bge's 512-token window**, so their tails were silently
truncated and unsearchable. A fixed 256-token chunker had 0% over the window and matched it on recall@5
(84.6 vs 84.6) with **2.2× less context** handed to the generator (996 vs 2,202 words at k=5). Budgets must be
counted in the embedder's tokens, which is what `layout-rag` does (`over_budget_chunks` is reported per
document and is 0 by construction).

### Orientation: published OHR-Bench retrieval numbers

The published numbers use a different stack: all 8,498 questions including Chinese, BGE-M3 and BM25 averaged,
and 1,024-token page-bounded chunks. Ground-truth text scores 70.4 overall, Azure Document Intelligence 60.6,
Marker 57.0 and MinerU 50.5. Compare each system with its own ground-truth control, not across tables.

### Re-running

```bash
python bench/run_pdf_retrieval.py --dataset ohrbench --data-dir data/ohrbench --docs-per-domain 45 \
    --chunker layout --out results/ohrbench_layout
# market baselines, same text + embedder + metrics:
python bench/run_pdf_retrieval.py ... --chunker recursive   # LangChain RecursiveCharacterTextSplitter
python bench/run_pdf_retrieval.py ... --chunker fixed       # fixed 512-token windows
python bench/run_pdf_retrieval.py ... --chunker docling     # Docling HybridChunker
python bench/run_pdf_retrieval.py ... --mode hybrid --rerank
```

The full OHR-Bench subset takes several CPU hours (bge-large embeds about 500 tokens/s on 8 threads). Chunks
and vectors are cached under `--work`.

## Design notes: what broke the naive version

Each rule below exists because a real document broke the simpler approach.

- **Line-box extractors kill the size signal.** `pdftotext -bbox` gives every word on a line the same height.
  Per-character spans (PyMuPDF `rawdict`) carry the declared size and weight.
- **Bold can be the only separator.** In some documents headings and body share a point size. Every style test
  therefore compares against the document's own body text, never an absolute threshold.
- **Detector labels are inconsistent.** DocLayNet models label visually identical headings `Title` on one page
  and `Section-header` on another, so the label is a hint and never part of the level signature.
- **"May only deepen by one" ratchets.** Repeated siblings `A.`, `B.`, `C.` drift one level deeper each time.
  A stack resolver assigns siblings the same level.
- **A centred heading is more prominent, not more indented.** Centring and caps break ties between unnumbered
  headings. They never let "Abstract" become the parent of "1 Introduction".
- **Furniture rules need an absolute floor.** A line appearing on 1 of 3 pages is "on 33% of pages". A
  ratio-only rule deleted whole sections of short documents, so a line must also repeat on at least 3 pages.
- **Promotion needs a ceiling and an edge band.** On a bundle of scanned forms, "larger than body" passed for
  hundreds of blocks. Promotions are capped, and nothing in the top or bottom 7% band is promoted (a bold logo
  once displaced the real title).
- **A masthead is not a title.** The biggest type on a cover is often an issue number such as "No. 6/2023".
- **A heading with no content deletes itself.** Headings reach chunks only as breadcrumbs, so an orphan is
  demoted to text, and a heading absorbed by a small-section merge is folded back in as text.
- **OCR has no fonts.** Stroke width (2·ink area / ink perimeter per line) recovers weight, and x-height (rows
  above 65% of the line's ink-profile peak) recovers size. Both are snapped to document-wide modes, since a smear
  of OCR sizes would invent extra heading tiers.
- **Source line identity beats geometry.** Re-deriving lines from coordinates interleaves two runs that share
  a baseline into word salad.
- **Type3 fonts declare size 0.** The glyph box height stands in, consistently within the document.

## Known limitations

- English-first: Tesseract `eng` plus English embedders. On OHR-Bench's Chinese questions page hit@5 is 14.6.
- Tables are serialized geometrically. Spanning (multi-level) column headers are attributed approximately.
- On OCR'd pages, table column labels outside the detected table box can be promoted to headings.
- Layout weights come from the Hugging Face Space `linhdo/document-layout-analysis` (YOLOv8, DocLayNet) at first
  use and are subject to that Space's terms. `--layout heuristic` needs no model.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev]"
pytest -q        # synthetic PDFs, no model downloads, no network
ruff check .
```

MIT licence. Built by [Gopal Sharma](https://github.com/gopalsharma1615-ai).
