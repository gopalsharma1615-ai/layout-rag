"""Running headers/footers, page numbers and tables of contents: text that must not reach the chunks."""
from __future__ import annotations

import re
from collections import defaultdict

from .types import Block

_DIGITS = re.compile(r"\d+")
_PAGE_NO = re.compile(r"^(page\s*)?[-–—(]?\s*\d{1,4}\s*[-–—)]?(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_ROMAN = re.compile(r"^[ivxlcdm]{1,6}$", re.IGNORECASE)
TOC_RE = re.compile(r"^\s*(table of contents|contents|index|arrangement of (sections|clauses|rules)|"
                    r"list of (figures|tables))\s*$", re.IGNORECASE)
_DOT_LEADER = re.compile(r"(\.{3,}|…|\s{3,})\s*\d{1,4}\s*$")


def _key(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.lower().split()))


def _tokens(text: str) -> set[str]:
    return set(_key(text).split())


def mark_furniture(blocks: list[Block], n_pages: int, band: float = 0.10, ratio: float = 0.30,
                   min_pages: int = 3) -> int:
    """Mark running headers/footers and bare page numbers as dropped. Returns how many blocks were marked.

    A block is furniture when it sits in the top/bottom band of its page AND a near-identical text (digits
    normalised, token overlap >= 0.8) appears in the same band on >= `ratio` of pages AND on >= `min_pages` pages.
    The absolute floor matters: with a ratio alone, one line on a 3-page document is "on 33% of pages" and whole
    sections of short documents get deleted. Matching ignores the detector's label, because running headers are
    often labelled Section-header and would otherwise become a heading level.
    """
    marked = 0
    for b in blocks:
        if b.label in ("Page-header", "Page-footer"):
            b.dropped = "furniture-label"
            marked += 1
        elif _in_band(b, band) and (_PAGE_NO.match(b.text.strip() or "x") or _ROMAN.match(b.text.strip() or "x")):
            b.dropped = "page-number"
            marked += 1

    candidates = [b for b in blocks if b.dropped is None and _in_band(b, band) and b.n_words <= 25]
    clusters: list[tuple[set[str], str, set[int], list[Block]]] = []
    for b in candidates:
        toks, pos = _tokens(b.text), _band_side(b)
        if not toks:
            continue
        for ctoks, cpos, pages, members in clusters:
            if cpos == pos and len(toks & ctoks) / len(toks | ctoks) >= 0.8:
                pages.add(b.page)
                members.append(b)
                break
        else:
            clusters.append((toks, pos, {b.page}, [b]))
    need = max(min_pages, ratio * n_pages)
    for _, _, pages, members in clusters:
        if len(pages) >= need:
            for b in members:
                b.dropped = "furniture-repeated"
                marked += 1
    return marked


def _in_band(b: Block, band: float) -> bool:
    if not b.page_h:
        return False
    return b.bbox[3] <= band * b.page_h or b.bbox[1] >= (1 - band) * b.page_h


def _band_side(b: Block) -> str:
    return "top" if b.bbox[3] <= b.page_h / 2 else "bottom"


def is_toc_heading(text: str) -> bool:
    return bool(TOC_RE.match(text.strip()))


def looks_like_toc_block(b: Block) -> bool:
    """A block whose lines mostly end in dot leaders / page numbers ("2.1 Scope ....... 14")."""
    lines = [ln.text for ln in b.lines if ln.text.strip()]
    if len(lines) < 3:
        return False
    return sum(bool(_DOT_LEADER.search(t)) for t in lines) / len(lines) >= 0.6


def mark_toc_pages(blocks: list[Block]) -> int:
    """Drop TOC blocks: those under a TOC heading on its page(s) plus any dot-leader listing."""
    marked = 0
    toc_pages: dict[int, float] = defaultdict(lambda: -1.0)
    for b in blocks:
        if b.dropped is None and b.n_words <= 6 and is_toc_heading(b.text):
            toc_pages[b.page] = b.bbox[1]
            b.dropped = "toc"
            marked += 1
    for b in blocks:
        if b.dropped is not None:
            continue
        if looks_like_toc_block(b) or (b.page in toc_pages and b.bbox[1] >= toc_pages[b.page]
                                         and _mostly_listing(b)):
            b.dropped = "toc"
            marked += 1
    return marked


def _mostly_listing(b: Block) -> bool:
    lines = [ln.text for ln in b.lines if ln.text.strip()]
    if not lines:
        return False
    ends_num = sum(bool(re.search(r"\d{1,4}\s*$", t)) for t in lines)
    return ends_num / len(lines) >= 0.5
