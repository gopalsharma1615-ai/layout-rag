"""Market-baseline chunkers on the SAME text and embedder, so only the chunking differs.

* fixed     - fixed windows of N embedder tokens, no overlap (the common default)
* recursive - LangChain RecursiveCharacterTextSplitter (pip install langchain-text-splitters)
* docling   - Docling DocumentConverter + HybridChunker (pip install docling); its own layout + OCR

Each returns layout_rag Chunk objects with page spans, so all chunkers are scored by one metric code path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from layout_rag.chunker import TokenCounter
from layout_rag.types import Chunk


def _page_texts(pdf: Path) -> list[str]:
    with fitz.open(pdf) as doc:
        return [p.get_text("text") for p in doc]


def _concat(pages: list[str]) -> tuple[str, list[int]]:
    starts, buf, pos = [], [], 0
    for t in pages:
        starts.append(pos)
        buf.append(t)
        pos += len(t) + 1
    return "\n".join(buf), starts


def _pages_of(span: tuple[int, int], starts: list[int]) -> list[int]:
    a, b = span
    out = [i for i, s in enumerate(starts) if s < b and (i + 1 == len(starts) or starts[i + 1] > a)]
    return out or [0]


def _mk(pdf: Path, i: int, text: str, pages: list[int], count: TokenCounter) -> Chunk:
    cid = hashlib.sha1(f"{pdf.name}:{i}".encode()).hexdigest()[:16]
    return Chunk(cid, pdf.stem, pdf.name, text, text, [], pages, count(text))


def fixed_chunks(pdf: Path, count: TokenCounter, max_tokens: int = 512) -> list[Chunk]:
    full, starts = _concat(_page_texts(pdf))
    tok = count.tok
    if tok is None:
        raise ValueError("fixed baseline needs a real tokenizer")
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = enc["offset_mapping"]
    step = max_tokens - 2
    out = []
    for i in range(0, len(offs), step):
        a, b = offs[i][0], offs[min(i + step, len(offs)) - 1][1]
        out.append(_mk(pdf, len(out), full[a:b], _pages_of((a, b), starts), count))
    return out


def recursive_chunks(pdf: Path, count: TokenCounter, max_tokens: int = 512) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    full, starts = _concat(_page_texts(pdf))
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens - 2, chunk_overlap=0,
                                              length_function=lambda s: count(s) - 2)
    out, pos = [], 0
    for piece in splitter.split_text(full):
        a = full.find(piece, pos)
        a = pos if a < 0 else a
        b = a + len(piece)
        pos = b
        out.append(_mk(pdf, len(out), piece, _pages_of((a, b), starts), count))
    return out


def docling_chunks(pdf: Path, count: TokenCounter, max_tokens: int = 512) -> list[Chunk]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(pdf)).document
    chunker = HybridChunker(tokenizer=count.tok, max_tokens=max_tokens - 2, merge_peers=True)
    out = []
    for ch in chunker.chunk(doc):
        text = chunker.contextualize(ch)
        pages = sorted({prov.page_no - 1 for item in ch.meta.doc_items for prov in item.prov}) or [0]
        out.append(_mk(pdf, len(out), text, pages, count))
    return out


BASELINES = {"fixed": fixed_chunks, "recursive": recursive_chunks, "docling": docling_chunks}
