"""Tesseract OCR with typography measured from the image.

Tesseract returns no font information, and without it the heading hierarchy of a scanned document collapses.
Two per-LINE measurements recover it (word-level ink depends too much on which letters the word contains):

* weight: stroke width = 2 * ink_area / ink_perimeter of the binarized line crop
* size:   x-height = number of rows whose ink exceeds 65% of the line profile's peak
          (0.5 lets digits and capitals inflate it; above 0.7 short lines collapse)

Bold is decided later, relative to the document's body text (see headings.decide_ocr_bold).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import Word

X_HEIGHT_TO_PT = 1 / 0.5   # x-height is roughly half the em size for Latin text faces


def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best, thresh = -1.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thresh = between, t
    return thresh


def line_typography(ink: np.ndarray) -> tuple[float, float]:
    """(stroke_width_px, x_height_px) of a binarized line crop (True = ink)."""
    area = float(ink.sum())
    if area < 4:
        return 0.0, 0.0
    # perimeter = ink pixels with at least one 4-neighbour of background
    pad = np.pad(ink, 1)
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    perimeter = float((ink & ~interior).sum()) or 1.0
    stroke = 2.0 * area / perimeter
    profile = ink.sum(axis=1)
    x_height = float((profile >= 0.65 * profile.max()).sum())
    return stroke, x_height


def ocr_page(image: np.ndarray, page_no: int, scale: float, lang: str = "eng") -> list[Word]:
    """OCR one page image (H x W x 3 uint8). `scale` converts image pixels to PDF points."""
    import pytesseract

    gray = image if image.ndim == 2 else (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    gray = gray.astype(np.uint8)
    ink_all = gray < _otsu(gray)
    data = pytesseract.image_to_data(gray, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)
    by_line: dict[tuple, list[int]] = defaultdict(list)
    for i, txt in enumerate(data["text"]):
        if txt and txt.strip() and float(data["conf"][i]) >= 0:
            by_line[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)

    words: list[Word] = []
    for lid, idxs in by_line.items():
        x0 = min(data["left"][i] for i in idxs)
        y0 = min(data["top"][i] for i in idxs)
        x1 = max(data["left"][i] + data["width"][i] for i in idxs)
        y1 = max(data["top"][i] + data["height"][i] for i in idxs)
        stroke, xh = line_typography(ink_all[y0:y1, x0:x1])
        size_pt = xh * scale * X_HEIGHT_TO_PT if xh else None
        for i in idxs:
            l, t, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(Word(data["text"][i].strip(), l * scale, t * scale, (l + w) * scale, (t + h) * scale,
                              page_no, size=round(size_pt, 1) if size_pt else None, bold=None,
                              stroke=round(stroke, 3), line_id=lid, conf=float(data["conf"][i])))
    return words
