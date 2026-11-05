"""Page layout: detect blocks, put words in them, order them for reading.

Two detectors:

* `YoloLayoutDetector` - a YOLOv8 model trained on DocLayNet (11 classes: Title, Section-header, Text, Table,
  List-item, Caption, Footnote, Formula, Picture, Page-header, Page-footer). The weights are downloaded from the
  Hugging Face Space ``linhdo/document-layout-analysis`` at first use; they are not redistributed here.
* `heuristic_blocks` - no model: blocks from the text source's own grouping, split wherever typography changes.

The detector's *label* is used for Table/Picture/Page-header/Page-footer and as a weak heading hint only: it is
applied inconsistently to visually identical headings, so heading levels come from measured typography
(see `headings.py`).
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .types import BBox, Block, Line, Word, area, intersection, iou, union

HF_SPACE = "linhdo/document-layout-analysis"
HF_WEIGHTS = "models/dla-model.pt"


@dataclass
class Detection:
    label: str
    bbox: BBox      # PDF points
    score: float


class YoloLayoutDetector:
    def __init__(self, weights: str | None = None, conf: float = 0.2, iou_thr: float = 0.8, imgsz: int = 1024):
        from ultralytics import YOLO

        if weights is None:
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(HF_SPACE, HF_WEIGHTS, repo_type="space")
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou_thr, imgsz

    def detect(self, image: np.ndarray, scale: float) -> list[Detection]:
        """`image` H x W x 3 (RGB); `scale` converts pixels to points."""
        res = self.model.predict(image[..., ::-1], conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        names = res.names
        out = []
        for xyxy, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            x0, y0, x1, y1 = (v * scale for v in xyxy)
            out.append(Detection(str(names[int(cls)]), (x0, y0, x1, y1), float(score)))
        return dedupe(out)


def dedupe(dets: list[Detection], iou_thr: float = 0.7, contain_thr: float = 0.9) -> list[Detection]:
    """Drop near-duplicate boxes (keep the higher score) and boxes nested inside a same-label box."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        dup = False
        for k in keep:
            if iou(d.bbox, k.bbox) > iou_thr:
                dup = True
                break
            small = area(d.bbox) or 1.0
            if d.label == k.label and intersection(d.bbox, k.bbox) / small > contain_thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


# ------------------------------------------------------------------------------------------------ lines
def build_lines(words: list[Word]) -> list[Line]:
    """Group words into lines by SOURCE line identity; geometry only as a fallback.

    Re-deriving lines from coordinates interleaves two text runs that share a baseline (two columns, or a heading
    beside a paragraph), producing word salad. The text source already knows which words form a line.
    """
    if not words:
        return []
    groups: dict[object, list[Word]] = defaultdict(list)
    loose: list[Word] = []
    for w in words:
        (groups[w.line_id] if w.line_id is not None else loose).append(w)
    lines = [Line(sorted(ws, key=lambda w: w.x0)) for ws in groups.values()]
    if loose:
        lines += _geometric_lines(loose)
    lines.sort(key=lambda ln: (round(ln.bbox[1], 0), ln.bbox[0]))
    return lines


def _geometric_lines(words: list[Word]) -> list[Line]:
    words = sorted(words, key=lambda w: (w.cy, w.x0))
    h = statistics.median([w.y1 - w.y0 for w in words]) or 1.0
    lines: list[list[Word]] = []
    for w in words:
        if lines and abs(lines[-1][-1].cy - w.cy) < 0.5 * h:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [Line(sorted(ws, key=lambda w: w.x0)) for ws in lines]


# ------------------------------------------------------------------------------------------------ blocks
def heuristic_blocks(words: list[Word], page: int, page_w: float, page_h: float) -> list[Block]:
    """Blocks without a layout model: the text source's block grouping, split where typography changes."""
    by_block: dict[object, list[Word]] = defaultdict(list)
    for w in words:
        key = w.line_id[:-1] if isinstance(w.line_id, tuple) and len(w.line_id) > 1 else None
        by_block[key].append(w)
    blocks: list[Block] = []
    for ws in by_block.values():
        cur: list[Line] = []
        for ln in build_lines(ws):
            if cur and _style_break(cur[-1], ln):
                blocks.append(_mk_block(cur, page, page_w, page_h))
                cur = []
            cur.append(ln)
        if cur:
            blocks.append(_mk_block(cur, page, page_w, page_h))
    return blocks


def _style_break(prev: Line, ln: Line) -> bool:
    ps, ls = prev.size or 0, ln.size or 0
    if ps and ls and max(ps, ls) / min(ps, ls) > 1.08:
        return True
    return (prev.bold_share > 0.6) != (ln.bold_share > 0.6)


def _mk_block(lines: list[Line], page: int, page_w: float, page_h: float, label: str = "Text",
              score: float = 1.0) -> Block:
    return Block(page=page, bbox=union([ln.bbox for ln in lines]), label=label, score=score, lines=lines,
                 page_w=page_w, page_h=page_h)


def assign_words(dets: list[Detection], words: list[Word], page: int, page_w: float, page_h: float) -> list[Block]:
    """Put each word in the smallest detection box that holds most of it; leftovers form heuristic blocks."""
    members: dict[int, list[Word]] = defaultdict(list)
    leftover: list[Word] = []
    order = sorted(range(len(dets)), key=lambda i: area(dets[i].bbox))
    for w in words:
        wa = area(w.bbox) or 1.0
        for i in order:
            if intersection(w.bbox, dets[i].bbox) / wa >= 0.5:
                members[i].append(w)
                break
        else:
            leftover.append(w)
    blocks = []
    for i, ws in members.items():
        d = dets[i]
        lines = build_lines(ws)
        blocks.append(_mk_block(lines, page, page_w, page_h, label=d.label, score=d.score))
    blocks += heuristic_blocks(leftover, page, page_w, page_h)
    pictures = [d.bbox for d in dets if d.label == "Picture"]
    for b in blocks:
        if b.label != "Picture" and any(intersection(b.bbox, p) / (area(b.bbox) or 1.0) >= 0.8 for p in pictures):
            b.label = "Picture-text"        # axis labels, legends: content, but never a heading
    return blocks


# ------------------------------------------------------------------------------------------------ reading order
def reading_order(blocks: list[Block], min_gap: float = 6.0) -> list[Block]:
    """Recursive XY-cut: split along the widest whitespace gap (columns or bands), then recurse."""
    if len(blocks) <= 1:
        return list(blocks)

    def gaps(axis: int) -> tuple[float, float]:
        lo, hi = (0, 2) if axis == 0 else (1, 3)
        spans = sorted((b.bbox[lo], b.bbox[hi]) for b in blocks)
        best, cut, reach = 0.0, None, spans[0][1]
        for s, e in spans[1:]:
            if s - reach > best:
                best, cut = s - reach, (reach + s) / 2
            reach = max(reach, e)
        return best, cut

    gx, cx = gaps(0)
    gy, cy = gaps(1)
    if max(gx, gy) < min_gap:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    # Prefer a vertical cut (columns) only when it is clearly wider than the best horizontal band.
    if gx >= min_gap and gx > gy * 1.5:
        left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 < cx]
        right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 >= cx]
    else:
        left = [b for b in blocks if (b.bbox[1] + b.bbox[3]) / 2 < cy]
        right = [b for b in blocks if (b.bbox[1] + b.bbox[3]) / 2 >= cy]
    if not left or not right:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    return reading_order(left, min_gap) + reading_order(right, min_gap)
