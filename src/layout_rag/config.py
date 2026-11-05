"""Every tunable of the chunking pipeline, in one place."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkerConfig:
    # --- text source -------------------------------------------------------------------------------------------
    text_source: str = "auto"          # "pdf" (text layer) | "ocr" (Tesseract) | "auto" (pdf unless unusable)
    ocr_dpi: int = 200                 # 300 dpi was measured slower AND less accurate than 200 for Tesseract
    ocr_lang: str = "eng"
    min_words_per_page: int = 5        # below this on most pages, the text layer is treated as missing

    # --- layout ------------------------------------------------------------------------------------------------
    layout: str = "auto"               # "yolo" (DocLayNet detector) | "heuristic" (no model) | "auto"
    layout_conf: float = 0.2
    layout_iou: float = 0.8
    render_dpi: int = 144              # page image resolution fed to the layout detector

    # --- furniture / front matter ------------------------------------------------------------------------------
    furniture_band: float = 0.10       # top/bottom share of the page where running headers/footers live
    furniture_ratio: float = 0.30      # a repeated line must appear on >= this share of pages ...
    furniture_min_pages: int = 3       # ... AND on at least this many pages (absolute floor: short docs keep text)
    drop_toc: bool = True

    # --- headings ----------------------------------------------------------------------------------------------
    max_heading_levels: int = 8
    max_heading_words: int = 15
    max_question_words: int = 40       # FAQ questions ("...?") may be long headings
    size_ratio: float = 1.12           # a heading is >= this x body size, or bold where body is not
    max_promotion_share: float = 0.25  # promote at most this share of body blocks to headings
    edge_band: float = 0.07            # never promote in the top/bottom 7% of the page (logos, mastheads)

    # --- chunking ----------------------------------------------------------------------------------------------
    max_tokens: int = 512              # measured with the EMBEDDER's tokenizer, breadcrumb included
    min_tokens: int = 300              # sections smaller than this merge with their neighbour when they fit
    tokenizer: str | None = "BAAI/bge-large-en-v1.5"   # None -> fast word-based estimate
    table_chunks: bool = True          # tables become their own chunks with headers repeated per row
