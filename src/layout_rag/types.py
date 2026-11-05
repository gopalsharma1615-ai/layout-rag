"""Plain data types shared by every pipeline stage.

Coordinates are PDF points (1/72 inch), origin top-left, on the page the object came from.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    size: float | None = None      # font size (pt): declared by the PDF, or estimated from x-height for OCR
    bold: bool | None = None       # declared by the PDF, or decided later from stroke width for OCR
    stroke: float | None = None    # OCR only: line stroke width in px (bold evidence)
    line_id: tuple | None = None   # source line identity (PDF block/line, Tesseract block/par/line)
    conf: float = 100.0

    @property
    def bbox(self) -> BBox:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def bbox(self) -> BBox:
        return union([w.bbox for w in self.words])

    @property
    def size(self) -> float | None:
        sizes = [w.size for w in self.words if w.size]
        return statistics.median(sizes) if sizes else None

    @property
    def bold_share(self) -> float:
        n = sum(len(w.text) for w in self.words) or 1
        return sum(len(w.text) for w in self.words if w.bold) / n


@dataclass
class Block:
    page: int
    bbox: BBox
    label: str = "Text"            # DocLayNet label, or "Text" from the heuristic detector
    score: float = 1.0             # detector confidence
    lines: list[Line] = field(default_factory=list)
    is_heading: bool = False
    level: int = 0                 # heading level, 1 = top; 0 for body blocks
    numbering: str | None = None   # "3.2", "IV", "(a)" ... when the heading is numbered
    dropped: str | None = None     # reason the block is excluded from chunks (furniture, toc, ...)
    page_w: float = 0.0
    page_h: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def words(self) -> list[Word]:
        return [w for ln in self.lines for w in ln.words]

    @property
    def n_words(self) -> int:
        return sum(len(ln.words) for ln in self.lines)

    @property
    def size(self) -> float | None:
        sizes = [w.size for w in self.words if w.size]
        return statistics.median(sizes) if sizes else None

    @property
    def bold_share(self) -> float:
        ws = self.words
        n = sum(len(w.text) for w in ws) or 1
        return sum(len(w.text) for w in ws if w.bold) / n

    def to_dict(self) -> dict:
        return {"page": self.page, "bbox": [round(v, 1) for v in self.bbox], "label": self.label,
                "score": round(self.score, 3), "is_heading": self.is_heading, "level": self.level,
                "numbering": self.numbering, "dropped": self.dropped, "size": self.size,
                "bold_share": round(self.bold_share, 2), "text": self.text}


@dataclass
class Section:
    heading_path: list[str]
    section_id: str
    blocks: list[Block] = field(default_factory=list)
    absorbed_headings: list[str] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return sorted({b.page for b in self.blocks})


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str                      # breadcrumb + body: what gets embedded and shown to the reader
    body: str                      # body only
    heading_path: list[str]
    pages: list[int]
    token_count: int
    content_type: str = "text"     # "text" | "table"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def union(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    den = area(a) + area(b) - inter
    return inter / den if den > 0 else 0.0
