"""PDF -> chunks: the full layout-aware pipeline.

    words (text layer | OCR) -> layout blocks -> reading order -> furniture/TOC removal -> run-in heading split
    -> heading detection -> levels -> sections -> token-budgeted chunks

Pass `out_dir` to get an audit trail - one file per stage - which is what you need to see WHICH stage broke a
document: 00_run.json, 01_blocks.jsonl, 02_outline.json, 03_sections.jsonl, 04_chunks.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz
import numpy as np

from . import furniture, headings
from .chunker import Chunker, TokenCounter, coverage
from .config import ChunkerConfig
from .layout import YoloLayoutDetector, assign_words, heuristic_blocks, reading_order
from .pdf_text import page_words, text_layer_usable
from .sections import build_sections
from .tables import detect_tables
from .types import Block, Chunk, Section

log = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    doc_id: str
    doc_name: str
    title: str | None
    chunks: list[Chunk]
    blocks: list[Block]
    sections: list[Section]
    stats: dict = field(default_factory=dict)


def _render(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


class DocLayoutChunker:
    def __init__(self, cfg: ChunkerConfig | None = None, detector: YoloLayoutDetector | None = None,
                 counter: TokenCounter | None = None):
        self.cfg = cfg or ChunkerConfig()
        self.detector = detector
        if self.detector is None and self.cfg.layout in ("yolo", "auto"):
            try:
                self.detector = YoloLayoutDetector(conf=self.cfg.layout_conf, iou_thr=self.cfg.layout_iou)
            except Exception as e:
                if self.cfg.layout == "yolo":
                    raise
                log.warning("layout model unavailable (%s); using heuristic blocks", e)
        self.chunker = Chunker(self.cfg, counter)

    def chunk_pdf(self, path: str | Path, out_dir: str | Path | None = None) -> ChunkResult:
        path = Path(path)
        t0 = time.time()
        data = path.read_bytes()
        doc_id = hashlib.sha1(data).hexdigest()[:12]
        doc = fitz.open(stream=data, filetype="pdf")
        cfg = self.cfg
        use_ocr = cfg.text_source == "ocr" or (cfg.text_source == "auto"
                                                and not text_layer_usable(doc, cfg.min_words_per_page))

        blocks: list[Block] = []
        for page in doc:
            pw, ph = page.rect.width, page.rect.height
            image = None
            if use_ocr:
                from .ocr import ocr_page

                image = _render(page, cfg.ocr_dpi)
                words = ocr_page(image, page.number, 72.0 / cfg.ocr_dpi, cfg.ocr_lang)
            else:
                words = page_words(page)
            if self.detector is not None:
                if image is None:
                    image = _render(page, cfg.render_dpi)
                scale = pw / image.shape[1]
                page_blocks = assign_words(self.detector.detect(image, scale), words, page.number, pw, ph)
            else:
                page_blocks = heuristic_blocks(words, page.number, pw, ph)
            blocks += reading_order([b for b in page_blocks if b.lines or b.label in ("Table",)])

        if use_ocr:
            headings.decide_ocr_bold(blocks)
        headings.snap_sizes(blocks)
        n_furn = furniture.mark_furniture(blocks, doc.page_count, cfg.furniture_band, cfg.furniture_ratio,
                                          cfg.furniture_min_pages)
        n_toc = furniture.mark_toc_pages(blocks) if cfg.drop_toc else 0
        if self.detector is None:
            blocks = detect_tables(blocks)
        blocks = headings.split_runin_headings(blocks)
        body = headings.body_style(blocks)
        title = headings.detect_headings(blocks, cfg, body)
        blocks = headings.merge_wrapped_headings(blocks)
        headings.assign_levels(blocks, body, cfg.max_heading_levels)
        n_orphans = headings.demote_orphans(blocks)
        sections = build_sections(blocks, cfg.drop_toc)
        chunks = self.chunker.chunk(sections, doc_id, path.name, title)
        stats = {
            "pages": doc.page_count, "text_source": "ocr" if use_ocr else "pdf",
            "layout": "yolo" if self.detector is not None else "heuristic", "blocks": len(blocks),
            "headings": sum(b.is_heading for b in blocks), "furniture_dropped": n_furn, "toc_dropped": n_toc,
            "orphan_headings_demoted": n_orphans, "sections": len(sections), "chunks": len(chunks),
            "max_chunk_tokens": max((c.token_count for c in chunks), default=0),
            "over_budget_chunks": sum(c.token_count > cfg.max_tokens for c in chunks),
            "coverage": round(coverage(sections, chunks), 4),
            "pages_chunked": len({p for c in chunks for p in c.pages}),
            "seconds": round(time.time() - t0, 2),
        }
        result = ChunkResult(doc_id, path.name, title, chunks, blocks, sections, stats)
        if out_dir is not None:
            self._dump(result, Path(out_dir))
        return result

    def _dump(self, r: ChunkResult, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "00_run.json").write_text(json.dumps({"doc": r.doc_name, "doc_id": r.doc_id, "title": r.title,
                                                     "config": asdict(self.cfg), "stats": r.stats}, indent=2))
        with open(out / "01_blocks.jsonl", "w") as f:
            f.writelines(json.dumps(b.to_dict(), ensure_ascii=False) + "\n" for b in r.blocks)
        outline = [{"level": b.level, "page": b.page, "text": headings.heading_text(b)} for b in r.blocks
                   if b.is_heading]
        (out / "02_outline.json").write_text(json.dumps(outline, indent=2, ensure_ascii=False))
        with open(out / "03_sections.jsonl", "w") as f:
            f.writelines(json.dumps({"section_id": s.section_id, "path": s.heading_path, "pages": s.pages,
                                    "blocks": len(s.blocks), "absorbed": s.absorbed_headings},
                                   ensure_ascii=False) + "\n" for s in r.sections)
        with open(out / "04_chunks.jsonl", "w") as f:
            f.writelines(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in r.chunks)
