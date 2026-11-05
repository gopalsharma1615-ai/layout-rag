"""Words with real typography from a PDF text layer (PyMuPDF).

`rawdict` gives every character with its span's font size, font name and flags, so each word carries a declared
size and bold flag. (Line-box based extractors such as `pdftotext -bbox` report the same height for every word on a
line, which kills the size signal headings depend on.)
"""
from __future__ import annotations

import fitz  # PyMuPDF

from .types import Word

_BOLD_FLAG = 1 << 4
_GLYPHLESS = ("glyphless", "glyphlessfont")


def _is_bold(span: dict) -> bool:
    font = span.get("font", "").lower()
    return bool(span.get("flags", 0) & _BOLD_FLAG) or any(k in font for k in ("bold", "black", "heavy", "semibold"))


def _word(chars: list[tuple[str, tuple, dict]], page_no: int, line_id: tuple) -> Word:
    text = "".join(c for c, _, _ in chars)
    spans = [s for _, _, s in chars]
    bold_chars = sum(1 for s in spans if _is_bold(s))
    size = max(s.get("size", 0.0) for s in spans)
    y0, y1 = min(b[1] for _, b, _ in chars), max(b[3] for _, b, _ in chars)
    if size < 1.0:   # Type3 fonts often declare size 0: the glyph box height is a consistent stand-in
        size = (y1 - y0) / 1.2
    return Word(text, min(b[0] for _, b, _ in chars), y0, max(b[2] for _, b, _ in chars), y1, page_no,
                size=round(size, 2) if size >= 1.0 else None, bold=bold_chars * 2 >= len(spans), line_id=line_id)


def page_words(page: fitz.Page) -> list[Word]:
    """All words on one page, in source order, with line identity preserved."""
    out: list[Word] = []
    raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)
    for bno, block in enumerate(raw.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        for lno, line in enumerate(block.get("lines", [])):
            if abs(line.get("dir", (1, 0))[1]) > 0.2:      # skip rotated text (margins, watermarks)
                continue
            cur: list[tuple[str, tuple, dict]] = []
            # spans of one line are contiguous: only whitespace or the line end ends a word
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if c.isspace():
                        if cur:
                            out.append(_word(cur, page.number, (bno, lno)))
                            cur = []
                    else:
                        cur.append((c, tuple(ch["bbox"]), span))
            if cur:
                out.append(_word(cur, page.number, (bno, lno)))
    return out


def text_layer_usable(doc: fitz.Document, min_words_per_page: int = 5) -> bool:
    """False when the text layer is missing, or is an invisible OCR overlay with fake typography.

    Scans re-published with a 'GlyphLessFont' overlay have text but no bold and meaningless sizes; trusting them
    flattens every heading to one level, so they are treated as image-only and OCR'd.
    """
    if doc.page_count == 0:
        return False
    sample = range(min(doc.page_count, 8))
    good = 0
    for i in sample:
        page = doc[i]
        fonts = [f[3].lower() for f in page.get_fonts(full=False)]
        if fonts and all(any(g in f for g in _GLYPHLESS) for f in fonts):
            continue
        if len(page.get_text("words")) >= min_words_per_page:
            good += 1
    return good * 2 > len(sample)
