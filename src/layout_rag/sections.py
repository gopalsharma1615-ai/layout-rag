"""Group blocks under their heading path."""
from __future__ import annotations

from .furniture import is_toc_heading
from .headings import heading_text, strip_continued
from .types import Block, Section


def build_sections(blocks: list[Block], drop_toc: bool = True) -> list[Section]:
    sections: list[Section] = [Section([], "0")]
    stack: list[str] = []
    counters: list[int] = []
    for b in blocks:
        if b.dropped is not None:
            continue
        if not b.is_heading:
            sections[-1].blocks.append(b)
            continue
        text, continued = strip_continued(heading_text(b))
        lvl = max(1, b.level)
        if continued and text in stack:
            # "Liabilities (continued)": resume the open section instead of opening a twin
            continue
        stack = stack[:lvl - 1] + [text]
        counters = counters[:lvl]
        while len(counters) < lvl:
            counters.append(0)
        counters[lvl - 1] += 1
        sections.append(Section(list(stack), ".".join(str(c) for c in counters[:len(stack)])))

    out = []
    for s in sections:
        if not s.blocks:
            continue
        # The FULL path is scanned: a TOC's own sub-entries become headings under it and must go too.
        if drop_toc and any(is_toc_heading(h) for h in s.heading_path):
            for b in s.blocks:
                b.dropped = "toc"
            continue
        out.append(s)
    return out
