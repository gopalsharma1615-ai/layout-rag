"""Heading detection and levelling from measured typography - no LLM, no training.

Principles (each one learned from a document that broke the naive version):

1. Signals are RELATIVE to the document's body text. "Bold" means something only when body text is not bold;
   in some documents headings and body share one size and bold is the only separator.
2. The detector's Title/Section-header label is a hint, not a level: it labels visually identical headings
   inconsistently, so it is excluded from the level signature.
3. Levels come from a STACK resolver over style signatures (and numbering depth when headings are numbered),
   not from a global style rank and not from a "may only deepen by one" clamp (which ratchets repeated siblings
   A./B./C. one level deeper each time).
4. Promotion of body blocks to headings is volume-capped and never happens in the page's top/bottom edge band
   (a bold logo lockup must not displace the real title).
5. A numbered body clause is not a heading just because it starts with "3.2".
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass

from .config import ChunkerConfig
from .types import Block, Line, Word

_FAMILIES = [
    ("keyword", re.compile(r"^(chapter|part|section|article|schedule|annex(?:ure)?|appendix)\s+"
                           r"([0-9]+[a-z]?|[ivxlcdm]+|[a-z])\b[.:\-–—]?\s*", re.IGNORECASE)),
    ("decimal", re.compile(r"^(\d{1,2}(?:\.\d{1,3})+)\.?\s+")),
    ("int", re.compile(r"^(\d{1,2})(?:[.)]\s*|\s+)(?=[A-Z])")),
    ("roman", re.compile(r"^([IVXLC]{1,6})[.)]\s+")),
    ("alpha", re.compile(r"^([A-Z])[.)]\s+")),
    ("paren", re.compile(r"^\(([a-z]{1,2}|[ivx]{1,4}|\d{1,2})\)\s+")),
]
_KEYWORD_RANK = {"part": 0, "chapter": 1, "schedule": 1, "appendix": 1, "annex": 1, "annexure": 1,
                 "article": 2, "section": 2}
_CONTINUED = re.compile(r"\s*[\(\[]?\s*(continued|contd\.?|cont'd)\s*[\)\]]?\s*$", re.IGNORECASE)
_TERMINAL = re.compile(r"[.;:,]\s*$")
_RUNIN = re.compile(r"^(\d{1,3}[A-Z]?\.)\s+([A-Z][^.—–]{1,90}?)\s*[.:]\s*[—–-]+\s*")
_HEADING_TEXT = re.compile(r"[A-Za-z]")
_STOP = {"the", "of", "and", "a", "an", "to", "in", "this", "that", "for", "or", "on", "by", "with", "is", "as",
         "at", "be", "it", "from"}


def _has_real_word(text: str) -> bool:
    """A heading names something: at least one word of >= 3 letters that is not a stopword."""
    for tok in re.findall(r"[A-Za-z]{3,}", text):
        if tok.lower() not in _STOP:
            return True
    return False


@dataclass
class Numbering:
    family: str
    value: str
    depth: int


def parse_numbering(text: str) -> Numbering | None:
    t = text.strip()
    for fam, rx in _FAMILIES:
        m = rx.match(t)
        if m:
            if fam == "keyword":
                return Numbering(fam, f"{m.group(1).lower()} {m.group(2)}", 1)
            val = m.group(1)
            return Numbering(fam, val, val.count(".") + 1 if fam == "decimal" else 1)
    return None


def strip_continued(text: str) -> tuple[str, bool]:
    new = _CONTINUED.sub("", text)
    return new, new != text


# ------------------------------------------------------------------------------------------------ typography
def decide_ocr_bold(blocks: list[Block], ratio: float = 1.25) -> None:
    """For OCR words (bold is None): bold when the LINE stroke width is >= ratio x the body median."""
    strokes = [w.stroke for b in blocks if b.n_words >= 8 for w in b.words if w.stroke]
    if not strokes:
        return
    body = statistics.median(strokes)
    for b in blocks:
        for w in b.words:
            if w.bold is None and w.stroke is not None:
                w.bold = w.stroke >= ratio * body


def snap_sizes(blocks: list[Block], tol: float = 0.12, max_modes: int = 6) -> list[float]:
    """Snap word sizes to the document's dominant sizes. OCR x-height gives a smear of values for a handful of
    real sizes; clustering raw values would invent extra heading tiers."""
    counts: Counter[float] = Counter()
    for b in blocks:
        for w in b.words:
            if w.size and w.size >= 1.0:
                counts[round(w.size * 2) / 2] += len(w.text)
    counts.pop(0.0, None)
    if not counts:
        return []
    total = sum(counts.values())
    modes: list[float] = []
    for size, n in counts.most_common():
        if n < 0.01 * total or len(modes) >= max_modes:
            break
        if all(abs(size - m) / m > tol for m in modes):
            modes.append(size)
    for b in blocks:
        for w in b.words:
            if w.size and w.size >= 1.0:
                near = min(modes, key=lambda m: abs(m - w.size))
                if abs(near - w.size) / near <= tol:
                    w.size = near
    return sorted(modes)


@dataclass
class BodyStyle:
    size: float
    bold_share: float


def body_style(blocks: list[Block]) -> BodyStyle:
    chars: Counter[float] = Counter()
    for b in blocks:
        if b.dropped is None and b.n_words >= 8:
            for w in b.words:
                if w.size:
                    chars[w.size] += len(w.text)
    size = chars.most_common(1)[0][0] if chars else 10.0
    body = [b for b in blocks if b.dropped is None and b.n_words >= 8 and b.size and abs(b.size - size) / size < 0.1]
    n = sum(len(w.text) for b in body for w in b.words) or 1
    bold = sum(len(w.text) for b in body for w in b.words if w.bold) / n
    return BodyStyle(size, bold)


def _caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def _centered(b: Block) -> bool:
    if not b.page_w:
        return False
    left, right = b.bbox[0], b.page_w - b.bbox[2]
    return abs(left - right) < 0.08 * b.page_w and (b.bbox[2] - b.bbox[0]) < 0.7 * b.page_w


def prominence(b: Block, body: BodyStyle) -> tuple:
    """Sort key: bigger = more prominent. Size first, then weight, then caps, then centring (a centred heading is
    more prominent, not more indented)."""
    size = round((b.size or body.size) / body.size, 2)
    return (size, b.bold_share >= 0.5, _caps(b.text), _centered(b))


# ------------------------------------------------------------------------------------------------ detection
def split_runin_headings(blocks: list[Block]) -> list[Block]:
    """'12. Priority of claim.—Save as provided ...' -> heading block + body block.

    Applied only when the pattern recurs (>= 3 blocks), so one stray sentence never becomes a heading.
    """
    hits = [(i, m) for i, b in enumerate(blocks) if b.dropped is None and (m := _RUNIN.match(b.text))]
    if len(hits) < 3:
        return blocks
    out: list[Block] = []
    hit_at = dict(hits)
    for i, b in enumerate(blocks):
        m = hit_at.get(i)
        if m is None:
            out.append(b)
            continue
        head, body = _split_block_at(b, m.end())
        if head is None:
            out.append(b)
            continue
        head.is_heading = True
        head.label = "Section-header"
        out.append(head)
        if body is not None:
            out.append(body)
    return out


def _split_block_at(b: Block, char_pos: int) -> tuple[Block | None, Block | None]:
    """Split a block after `char_pos` characters of its (first-line-joined) text, splitting a word if the cut
    falls inside it (OCR glues 'Definitions.—In' into one word)."""
    words = b.words
    pos, cut = 0, None
    for k, w in enumerate(words):
        start = pos
        pos += len(w.text) + 1
        if pos - 1 >= char_pos:
            cut = (k, char_pos - start)
            break
    if cut is None:
        return None, None
    k, off = cut
    head_words = words[:k]
    tail_words = words[k + 1:]
    w = words[k]
    if 0 < off < len(w.text):
        frac = off / len(w.text)
        xm = w.x0 + frac * (w.x1 - w.x0)
        a = Word(w.text[:off], w.x0, w.y0, xm, w.y1, w.page, w.size, w.bold, w.stroke, w.line_id, w.conf)
        c = Word(w.text[off:], xm, w.y0, w.x1, w.y1, w.page, w.size, w.bold, w.stroke, w.line_id, w.conf)
        head_words.append(a)
        tail_words.insert(0, c)
    elif off >= len(w.text):
        head_words.append(w)
    else:
        tail_words.insert(0, w)
    head_words = [x for x in head_words if x.text.strip("—–- ")]
    tail_words = [x for x in tail_words if x.text]
    if not head_words:
        return None, None
    from .layout import build_lines

    def mk(ws: list[Word]) -> Block | None:
        if not ws:
            return None
        lines = build_lines(ws)
        return Block(page=b.page, bbox=_union_words(ws), label=b.label, score=b.score, lines=lines,
                     page_w=b.page_w, page_h=b.page_h)

    return mk(head_words), mk(tail_words)


def _union_words(ws: list[Word]):
    return (min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws))


def detect_headings(blocks: list[Block], cfg: ChunkerConfig, body: BodyStyle) -> str | None:
    """Set `is_heading` on blocks. Returns the document title (excluded from heading paths)."""
    live = [b for b in blocks if b.dropped is None and b.label not in ("Table", "Picture", "Picture-text", "Formula",
                                                                       "Caption", "Footnote")]
    title = _doc_title(live, body)

    def short(b: Block) -> bool:
        t = b.text.strip()
        if not _has_real_word(t) or len(b.lines) > 3:
            return False
        if t.endswith("?"):
            return b.n_words <= cfg.max_question_words
        return b.n_words <= cfg.max_heading_words

    def styled(b: Block) -> bool:
        if (b.size or body.size) < 0.9 * body.size:
            return False                    # smaller than body text: captions, figure labels, footnotes
        size_up = (b.size or 0) >= body.size * cfg.size_ratio
        bold_up = b.bold_share >= 0.6 and body.bold_share < 0.3
        return size_up or bold_up

    def clause(b: Block) -> bool:
        """Numbered body clause, not a heading: long and ends like a sentence, in body style."""
        t = b.text.strip()
        num = parse_numbering(t)
        if t.endswith("?"):
            return False
        if num and num.family in ("decimal", "paren", "int") and b.n_words > 6 and not styled(b):
            return True
        return b.n_words > 8 and bool(_TERMINAL.search(t)) and not styled(b)

    labelled = [b for b in live if (b.label in ("Section-header", "Title") or b.is_heading) and short(b)
                and b is not title and (b.size or body.size) >= 0.9 * body.size]
    # Learn the document's heading style from its labelled headings, then demote labelled blocks lacking it.
    heads_bold = sum(b.bold_share >= 0.5 for b in labelled) / len(labelled) if labelled else 0.0
    for b in labelled:
        if b.is_heading:           # already validated (run-in split, recurrence-checked)
            continue
        if clause(b):
            continue
        keep = styled(b)
        if not keep and heads_bold >= 0.6 and body.bold_share < 0.3:
            keep = False                                   # the doc's headings are bold; this one is not
        elif not keep:
            # detector confidence is an independent witness, trusted only for short one-line blocks
            keep = b.score >= 0.5 and b.n_words <= 6 and len(b.lines) == 1
        b.is_heading = keep

    # Promote missed headings among plain text blocks: short, styled, outside the edge band, volume-capped.
    plain = [b for b in live if not b.is_heading and b is not title and b.label in ("Text", "List-item")]
    cands = [b for b in plain if short(b) and styled(b) and not clause(b) and not _edge(b, cfg.edge_band)
             and len(b.lines) <= 2]
    # share cap stops runaway promotion on noisy scans; the floor lets short documents keep their headings
    cap = max(10, int(cfg.max_promotion_share * max(1, len(plain))))
    cands.sort(key=lambda b: prominence(b, body), reverse=True)
    for b in cands[:cap]:
        b.is_heading = True

    for b in blocks:
        if b.is_heading:
            b.numbering = (n.value if (n := parse_numbering(b.text)) else None)
    if title is None:
        return None
    title.dropped = "title"          # kept as metadata and as the root of every breadcrumb
    return " ".join(title.text.split())


def _edge(b: Block, band: float) -> bool:
    return bool(b.page_h) and (b.bbox[3] <= band * b.page_h or b.bbox[1] >= (1 - band) * b.page_h)


def _doc_title(live: list[Block], body: BodyStyle) -> Block | None:
    """The title is on the first content page: the Title-labelled (or largest) short line block with the MOST
    WORDS - a masthead's issue number ("No. 6/2023") is often the biggest type on the page."""
    if not live:
        return None
    first = min(b.page for b in live)
    page = [b for b in live if b.page == first and b.n_words <= 25 and _HEADING_TEXT.search(b.text)]
    def wordy(b: Block) -> bool:            # "No. 6/2023" (an issue number in a masthead) is not a title
        return sum(1 for w in b.words if sum(c.isalpha() for c in w.text) >= 2) >= 3

    titled = [b for b in page if b.label == "Title" and wordy(b)]
    if not titled:
        titled = [b for b in page if wordy(b) and (b.size or 0) >= body.size * 1.3 and b.bbox[1] < 0.5 * b.page_h]
    if not titled:
        return None
    top = max((b.size or 0) for b in titled)
    return max((b for b in titled if (b.size or 0) >= top * 0.95), key=lambda b: b.n_words)


# ------------------------------------------------------------------------------------------------ levelling
def assign_levels(blocks: list[Block], body: BodyStyle, max_levels: int = 8) -> None:
    """Stack-based level resolution. Siblings share a level; a less prominent heading nests under the open one."""
    stack: list[tuple[tuple, tuple]] = []   # (key, prominence)
    for b in blocks:
        if not b.is_heading:
            continue
        b_text, _ = strip_continued(b.text)
        num = parse_numbering(b_text)
        prom = prominence(b, body)
        if num and num.family in ("decimal", "int"):
            key = ("decimal", num.depth)
        elif num and num.family == "keyword":
            key = ("keyword", _KEYWORD_RANK.get(num.value.split()[0], 3))
        else:
            key = ("style", prom)

        idx = next((i for i, (k, _) in enumerate(stack) if k == key), None)
        if idx is not None:
            del stack[idx + 1:]
            b.level = idx + 1
            continue
        while stack and not _outranks(stack[-1], (key, prom)):
            stack.pop()
        stack.append((key, prom))
        b.level = min(len(stack), max_levels)


def _outranks(parent: tuple[tuple, tuple], child: tuple[tuple, tuple]) -> bool:
    (pk, pp), (ck, cp) = parent, child
    if pk[0] == ck[0] == "decimal":
        return pk[1] < ck[1]
    if pk[0] == ck[0] == "keyword":
        return pk[1] < ck[1]
    if pk[0] == "keyword" and ck[0] != "keyword":
        return pp >= cp                     # "Chapter II" contains same-styled unnumbered headings
    if (pk[0] == "decimal") != (ck[0] == "decimal"):
        return pp[:2] > cp[:2]              # numbered vs unnumbered: size and weight only
    return pp > cp


def demote_orphans(blocks: list[Block]) -> int:
    """A heading with no content before the next heading of the same or higher level would delete its own text
    (headings reach chunks only as breadcrumbs), so it becomes body text."""
    live = [b for b in blocks if b.dropped is None]
    n = 0
    for i, b in enumerate(live):
        if not b.is_heading:
            continue
        nxt = live[i + 1] if i + 1 < len(live) else None
        if nxt is None or (nxt.is_heading and nxt.level <= b.level):
            b.is_heading, b.level, b.label = False, 0, "Text"
            n += 1
    return n


def merge_wrapped_headings(blocks: list[Block]) -> list[Block]:
    """Rejoin a heading split across two blocks by line wrap: short, no terminal punctuation at all, and the next
    block starts lowercase. (A colon is a hard stop here even though it is a soft one for prose joins.)"""
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (prev is not None and prev.is_heading and prev.dropped is None and b.dropped is None and b.page == prev.page
                and prev.n_words <= 15 and not re.search(r"[.;:,?!]\s*$", prev.text)
                and b.text[:1].islower() and b.n_words <= 12 and len(b.lines) <= 2):
            prev.lines = prev.lines + b.lines
            prev.bbox = (min(prev.bbox[0], b.bbox[0]), min(prev.bbox[1], b.bbox[1]),
                         max(prev.bbox[2], b.bbox[2]), max(prev.bbox[3], b.bbox[3]))
            continue
        out.append(b)
    return out


def heading_text(b: Block) -> str:
    t = " ".join(ln.text for ln in b.lines)
    return strip_continued(" ".join(t.split()))[0]


__all__ = ["BodyStyle", "Line", "Numbering", "assign_levels", "body_style", "decide_ocr_bold", "demote_orphans",
           "detect_headings", "heading_text", "merge_wrapped_headings", "parse_numbering", "prominence",
           "snap_sizes", "split_runin_headings", "strip_continued"]
