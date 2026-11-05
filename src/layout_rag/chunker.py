"""Sections -> retrieval chunks, budgeted in the EMBEDDER's tokens.

Counting whitespace words is the classic silent bug: 512 words are ~650 BERT tokens, so most "512-word" chunks
are truncated by a 512-token embedder and their tails are never searchable. Here every budget is measured with the
embedder's own tokenizer, breadcrumb and special tokens included.

Guarantees:
* lossless - every content word of every kept section reaches some chunk (checked, reported in stats);
* a heading absorbed by a small-section merge is folded back in as text, never discarded;
* tables become separate chunks whose rows keep their column headers.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import ChunkerConfig
from .tables import serialise_table
from .types import Chunk, Section

_SENT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"\w+|[^\w\s]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


class TokenCounter:
    """Exact counts with a Hugging Face tokenizer, or a conservative estimate when none is configured."""

    def __init__(self, name: str | None):
        self.tok = None
        if name:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(name)
            self.tok.model_max_length = 10**9   # we count long texts on purpose; silence the length warning

    def __call__(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=True, truncation=False)["input_ids"])
        return math.ceil(len(_WORD.findall(text)) * 1.15) + 2


@dataclass
class _Unit:
    text: str
    pages: list[int]
    kind: str = "text"   # "text" | "table"
    header: str = ""


@dataclass
class _Work:
    section: Section
    units: list[_Unit] = field(default_factory=list)


def block_text(lines_text: str) -> str:
    t = _HYPHEN_BREAK.sub(r"\1\2", lines_text)
    return " ".join(t.split())


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text) if p.strip()]
    return parts or [text]


class Chunker:
    def __init__(self, cfg: ChunkerConfig, counter: TokenCounter | None = None):
        self.cfg = cfg
        self.count = counter or TokenCounter(cfg.tokenizer)

    # -------------------------------------------------------------------------------------------- public
    def chunk(self, sections: list[Section], doc_id: str, doc_name: str, title: str | None) -> list[Chunk]:
        for s in sections:
            s.absorbed_headings = []          # chunk() is idempotent: merges are recomputed every call
        works = [self._units(s) for s in sections]
        works = self._merge_small(works, title)
        chunks: list[Chunk] = []
        depth = max((len(w.section.heading_path) for w in works), default=0)
        for w in works:
            chunks += self._pack(w, doc_id, doc_name, title, depth)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["prev_chunk_id"] = chunks[i - 1].chunk_id if i else None
            c.metadata["next_chunk_id"] = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
        return chunks

    # -------------------------------------------------------------------------------------------- internals
    def _units(self, s: Section) -> _Work:
        w = _Work(s)
        for b in s.blocks:
            if b.label == "Table" and self.cfg.table_chunks:
                header, rows = serialise_table(b)
                if rows:
                    for r in rows:
                        w.units.append(_Unit(r, [b.page], "table", header))
                    continue
            t = block_text(b.text)
            if t:
                w.units.append(_Unit(t, [b.page]))
        return w

    def _tokens(self, w: _Work) -> int:
        return self.count("\n".join(u.text for u in w.units))

    def _merge_small(self, works: list[_Work], title: str | None) -> list[_Work]:
        """Absorb the next section into a small one when both share a parent and the result fits the budget.
        The absorbed section's heading is kept as text and recorded in `absorbed_headings`."""
        out: list[_Work] = []
        for w in works:
            prev = out[-1] if out else None
            if prev is not None and self._mergeable(prev, w):
                head = w.section.heading_path[-1] if w.section.heading_path else ""
                if head:
                    prev.units.append(_Unit(head, []))
                    prev.section.absorbed_headings.append(head)
                prev.units += w.units
                continue
            out.append(w)
        return out

    def _mergeable(self, a: _Work, b: _Work) -> bool:
        if any(u.kind == "table" for u in a.units + b.units):
            return False
        pa, pb = a.section.heading_path, b.section.heading_path
        if not pa or not pb or pa[:-1] != pb[: len(pa) - 1]:
            return False                          # b must be a sibling or descendant of a
        if self._tokens(a) >= self.cfg.min_tokens:
            return False
        prefix = self.count(self._crumb(None, pa))
        return prefix + self._tokens(a) + self._tokens(b) + 8 <= self.cfg.max_tokens

    @staticmethod
    def _crumb(title: str | None, path: list[str]) -> str:
        parts = ([title] if title else []) + path
        return " > ".join(parts) + "\n\n" if parts else ""

    def _pieces(self, text: str, budget: int) -> list[str]:
        """Split one unit that exceeds the budget: sentences, then words."""
        if self.count(text) <= budget:
            return [text]
        out: list[str] = []
        cur = ""
        for s in split_sentences(text):
            if self.count(s) > budget:
                if cur:
                    out.append(cur)
                    cur = ""
                out += self._hard_split(s, budget)
                continue
            cand = f"{cur} {s}".strip()
            if self.count(cand) <= budget:
                cur = cand
            else:
                out.append(cur)
                cur = s
        if cur:
            out.append(cur)
        return out

    def _hard_split(self, text: str, budget: int) -> list[str]:
        words = text.split()
        out, cur = [], []
        for wd in words:
            if cur and self.count(" ".join(cur + [wd])) > budget:
                out.append(" ".join(cur))
                cur = []
            cur.append(wd)
        if cur:
            out.append(" ".join(cur))
        return out

    def _pack(self, w: _Work, doc_id: str, doc_name: str, title: str | None, depth: int) -> list[Chunk]:
        path = w.section.heading_path
        crumb = self._crumb(title, path)
        groups: list[tuple[str, list[_Unit]]] = []   # (kind, units)
        for u in w.units:
            if groups and groups[-1][0] == u.kind and (u.kind == "text" or groups[-1][1][0].header == u.header):
                groups[-1][1].append(u)
            else:
                groups.append((u.kind, [u]))

        bodies: list[tuple[str, str, list[int]]] = []   # (kind, body, pages)
        for kind, units in groups:
            header = f"{units[0].header}\n" if kind == "table" and units[0].header else ""
            budget = self.cfg.max_tokens - self.count(crumb + header) - 2
            cur_txt, cur_pages = "", []
            for u in units:
                for piece in self._pieces(u.text, budget):
                    sep = "\n" if kind == "table" else "\n\n"
                    cand = f"{cur_txt}{sep}{piece}" if cur_txt else piece
                    if cur_txt and self.count(header + cand) > budget:
                        bodies.append((kind, header + cur_txt, cur_pages))
                        cur_txt, cur_pages = piece, list(u.pages)
                    else:
                        cur_txt = cand
                        cur_pages += u.pages
            if cur_txt:
                bodies.append((kind, header + cur_txt, cur_pages))

        chunks = []
        n = len(bodies)
        pages_all = sorted({p for u in w.units for p in u.pages})
        for i, (kind, body, pages) in enumerate(bodies):
            pages = sorted(set(pages)) or pages_all
            text = crumb + body
            cid = hashlib.sha1(f"{doc_id}:{w.section.section_id}:{i}:{body[:64]}".encode()).hexdigest()[:16]
            meta = {"section_id": w.section.section_id, "section_depth": len(path), "pages": pages}
            for lvl in range(depth):
                meta[f"h{lvl + 1}"] = path[lvl] if lvl < len(path) else ""
            if n > 1:
                meta["part"] = f"{i + 1}/{n}"
            if w.section.absorbed_headings:
                meta["absorbed_headings"] = list(w.section.absorbed_headings)
            chunks.append(Chunk(cid, doc_id, doc_name, text, body, list(path), pages, self.count(text), kind, meta))
        return chunks


def coverage(sections: list[Section], chunks: list[Chunk]) -> float:
    """Share of content words (multiset) from kept sections that appear in chunk text."""
    want: Counter[str] = Counter()
    for s in sections:
        for b in s.blocks:
            want.update(_WORD.findall(block_text(b.text).lower()))
    have: Counter[str] = Counter()
    for c in chunks:
        have.update(_WORD.findall(c.text.lower()))
    total = sum(want.values())
    if not total:
        return 1.0
    return sum(min(n, have[t]) for t, n in want.items()) / total
