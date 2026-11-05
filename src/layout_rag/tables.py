"""Serialize a table block so every row carries its column headers.

A row like "Sale proceeds  4,830,214  4,512,007" is meaningless to a retriever once it is cut away from its
header. Each row is emitted as  `row label | <col header> = <value> | ...`  so it can be chunked independently.

Rows are grouped GEOMETRICALLY (some OCR engines report every cell as its own line). Column anchors come from
the data row with the MOST numeric cells - a width-derived gap threshold collapses sparse tables into one column.
"""
from __future__ import annotations

import re
import statistics

from .types import Block, Word

_CURRENCY = {"$", "€", "£", "₹", "¥", "US$", "Rs.", "Rs"}
_CUR_RE = re.compile(r"US\$|Rs\.?|[$€£₹¥]")
_NUM = re.compile(r"^[\(\-–]?\d[\d,]*(\.\d+)?%?\)?$")
_YEAR = re.compile(r"^(FY\s?)?(19|20)\d{2}([/-](19|20)?\d{2})?$", re.IGNORECASE)


def _is_num(text: str) -> bool:
    """An amount: '4,830', '(2,100)', '12.5%', '$ (3,193)'. Bare years, 'FY2024' and '2021/2022' are labels."""
    t = text.strip()
    if _YEAR.match(t):
        return False
    return bool(_NUM.match(_CUR_RE.sub("", t).replace(" ", "")))


def _tidy(text: str) -> str:
    return re.sub(r"(US\$|[$€£₹¥])\s+(?=[\d(])", r"\1", text)


def _rows(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    h = statistics.median([w.y1 - w.y0 for w in words]) or 1.0
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows and abs(statistics.mean(x.cy for x in rows[-1]) - w.cy) <= 0.5 * h:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def _cells(row: list[Word]) -> list[tuple[float, float, str]]:
    """Split a row into cells at gaps wider than ~1 character height."""
    if not row:
        return []
    h = statistics.median([w.y1 - w.y0 for w in row]) or 1.0
    cells: list[list[Word]] = [[row[0]]]
    for w in row[1:]:
        if w.x0 - cells[-1][-1].x1 > 0.9 * h:
            cells.append([w])
        else:
            cells[-1].append(w)
    out = [(c[0].x0, c[-1].x1, _tidy(" ".join(w.text for w in c))) for c in cells]
    merged: list[tuple[float, float, str]] = []
    for cell in out:                         # "$" | "(284)" -> "$(284)": a bare currency sign is not a column
        if merged and merged[-1][2] in _CURRENCY:
            x0, _, sym = merged.pop()
            cell = (x0, cell[1], sym + cell[2])
        merged.append(cell)
    return merged


def serialise_table(b: Block) -> tuple[str, list[str]]:
    """Returns (header_line, row_lines). Falls back to plain lines when no column structure is found."""
    rows = [_cells(r) for r in _rows(b.words)]
    rows = [r for r in rows if r]
    if not rows:
        return "", []
    anchor = max(rows, key=lambda r: (sum(_is_num(t) for _, _, t in r), len(r)))
    n_num = sum(_is_num(t) for _, _, t in anchor)
    if n_num == 0 or len(anchor) < 2:
        return "", [" | ".join(t for _, _, t in r) for r in rows]
    centers = [(x0 + x1) / 2 for x0, x1, _ in anchor]

    def col_of(cell: tuple[float, float, str]) -> int:
        x0, x1, _ = cell
        overl = [max(0.0, min(x1, c + 1) - max(x0, c - 1)) for c in centers]
        if max(overl) > 0:
            return max(range(len(centers)), key=lambda i: overl[i])
        return min(range(len(centers)), key=lambda i: abs((x0 + x1) / 2 - centers[i]))

    grid: list[list[str]] = []
    for r in rows:
        cols = [""] * len(centers)
        for cell in r:
            i = col_of(cell)
            cols[i] = (cols[i] + " " + cell[2]).strip()
        grid.append(cols)

    first_data = next((i for i, g in enumerate(grid) if any(_is_num(c) for c in g[1:])), 0)
    header_rows = grid[:min(first_data, 3)]
    headers = [" ".join(h[i] for h in header_rows if h[i]).strip() for i in range(len(centers))]
    header_line = " | ".join(h for h in headers if h)
    lines: list[str] = []
    for g in grid[first_data:]:
        label = g[0] if g[0] and not _is_num(g[0]) else ""
        parts = [label] if label else []
        for i, v in enumerate(g):
            if not v or (i == 0 and label):
                continue
            parts.append(f"{headers[i]} = {v}" if headers[i] else v)
        if parts:
            lines.append(" | ".join(parts))
    return header_line, lines


def _row_like(b: Block) -> tuple[bool, bool]:
    """(is a one-baseline row with >= 2 separated cells, has a numeric cell)."""
    rows = _rows(b.words)
    if len(rows) != 1 or b.n_words > 30:
        return False, False
    cells = _cells(rows[0])
    return len(cells) >= 2, any(_is_num(t) for _, _, t in cells)


def detect_tables(blocks: list[Block], min_rows: int = 3) -> list[Block]:
    """Merge runs of >= `min_rows` consecutive row-like blocks on one page (each with a numeric cell; a header row
    without numbers may lead the run) into a single Table block. Used when no layout model labels tables."""
    from .layout import build_lines
    from .types import union

    out: list[Block] = []
    i = 0
    while i < len(blocks):
        j = i
        run: list[Block] = []
        while j < len(blocks) and blocks[j].dropped is None and blocks[j].label == "Text" and \
                blocks[j].page == blocks[i].page:
            row, num = _row_like(blocks[j])
            if not row or (not num and run):
                break
            run.append(blocks[j])
            j += 1
        data_rows = sum(_row_like(b)[1] for b in run)
        if data_rows >= min_rows:
            words = [w for b in run for w in b.words]
            first = run[0]
            out.append(Block(page=first.page, bbox=union([b.bbox for b in run]), label="Table", score=1.0,
                             lines=build_lines(words), page_w=first.page_w, page_h=first.page_h))
            i = j
        else:
            out.append(blocks[i])
            i += 1
    return out
