from layout_rag import ChunkerConfig, DocLayoutChunker
from layout_rag.headings import assign_levels, body_style, parse_numbering, strip_continued
from layout_rag.types import Block, Line, Word


def outline(result):
    return [(b.level, " ".join(b.text.split())) for b in result.blocks if b.is_heading]


def all_text(result):
    return "\n".join(c.text for c in result.chunks)


def test_title_detected_and_used_as_breadcrumb_root(report_result):
    assert report_result.title == "Annual Operations Review 2025"
    assert all(c.text.startswith("Annual Operations Review 2025") for c in report_result.chunks)


def test_heading_levels_from_numbering_and_typography(report_result):
    assert outline(report_result) == [
        (1, "1. Introduction"), (2, "1.1 Scope"), (2, "1.2 Definitions"),
        (1, "2. Results"), (2, "2.1 Regional results"), (1, "3. Outlook"),
    ]


def test_running_header_footer_and_toc_removed(report_result):
    text = all_text(report_result)
    assert "Northwind Logistics - Annual Operations Review" not in text
    assert "Page 3" not in text
    assert "........" not in text and "Contents" not in text


def test_lossless_and_within_budget(report_result):
    s = report_result.stats
    assert s["coverage"] >= 0.99, s
    assert s["over_budget_chunks"] == 0
    assert max(c.token_count for c in report_result.chunks) <= 256


def test_long_section_is_split_into_parts(report_result):
    parts = [c for c in report_result.chunks if c.heading_path[-1:] == ["2.1 Regional results"]]
    assert len(parts) >= 2
    assert all("part" in c.metadata for c in parts)


def test_table_rows_carry_headers(report_result):
    tables = [c for c in report_result.chunks if c.content_type == "table"] or report_result.chunks
    text = "\n".join(c.text for c in tables)
    assert "Revenue | 2024 = 4,830 | 2025 = 5,120" in text


def test_metadata_is_filterable(report_result):
    c = next(c for c in report_result.chunks if c.heading_path == ["1. Introduction", "1.1 Scope"])
    assert c.metadata["h1"] == "1. Introduction" and c.metadata["h2"] == "1.1 Scope"
    assert c.metadata["section_id"].startswith("1.")
    assert c.pages == [1]


def test_short_document_keeps_repeated_line(short_pdf):
    r = DocLayoutChunker(ChunkerConfig(layout="heuristic", tokenizer=None)).chunk_pdf(short_pdf)
    assert "Section 4 applies to all contracts" in all_text(r)


def test_small_sections_merge_and_keep_absorbed_heading(report_pdf):
    r = DocLayoutChunker(ChunkerConfig(layout="heuristic", tokenizer=None, max_tokens=256,
                                       min_tokens=120)).chunk_pdf(report_pdf)
    intro = next(c for c in r.chunks if c.heading_path == ["1. Introduction"])
    assert "1.1 Scope" in intro.metadata.get("absorbed_headings", [])
    assert "1.1 Scope" in intro.body          # the absorbed heading is folded back in as text
    assert r.stats["coverage"] >= 0.99


def test_stage_dump(report_pdf, tmp_path):
    DocLayoutChunker(ChunkerConfig(layout="heuristic", tokenizer=None)).chunk_pdf(report_pdf, out_dir=tmp_path)
    for name in ("00_run.json", "01_blocks.jsonl", "02_outline.json", "03_sections.jsonl", "04_chunks.jsonl"):
        assert (tmp_path / name).stat().st_size > 0


# ---------------------------------------------------------------------------------------------- unit level
def _hb(text, size=12.0, bold=True, y=0.0):
    w = Word(text, 72, y, 200, y + size, 0, size=size, bold=bold, line_id=(int(y), 0))
    b = Block(page=0, bbox=(72, y, 200, y + size), lines=[Line([w])], page_w=595, page_h=842)
    b.is_heading = True
    return b


def _body_block(y):
    ws = [Word(f"word{i}", 72 + i * 30, y, 98 + i * 30, y + 10, 0, size=10.0, bold=False, line_id=(int(y), 1))
          for i in range(12)]
    return Block(page=0, bbox=(72, y, 450, y + 10), lines=[Line(ws)], page_w=595, page_h=842)


def test_repeated_siblings_do_not_ratchet_deeper():
    blocks = [_hb("A. First", y=0), _body_block(20), _hb("B. Second", y=40), _body_block(60),
              _hb("C. Third", y=80), _body_block(100)]
    assign_levels(blocks, body_style(blocks))
    assert [b.level for b in blocks if b.is_heading] == [1, 1, 1]


def test_less_prominent_style_nests():
    blocks = [_hb("Overview", size=16, y=0), _body_block(20), _hb("Background", size=12, y=40),
              _body_block(60), _hb("Method", size=16, y=80), _body_block(100)]
    assign_levels(blocks, body_style(blocks))
    assert [b.level for b in blocks if b.is_heading] == [1, 2, 1]


def test_parse_numbering():
    assert parse_numbering("2.3.1 Liquidity").depth == 3
    assert parse_numbering("Chapter IV General").family == "keyword"
    assert parse_numbering("(a) the borrower").family == "paren"
    assert parse_numbering("Revenue grew") is None


def test_strip_continued():
    assert strip_continued("Liabilities (continued)") == ("Liabilities", True)
    assert strip_continued("Liabilities") == ("Liabilities", False)


def test_currency_sign_joins_its_amount():
    from layout_rag.tables import serialise_table

    def row(y, cells):
        return [Word(t, x, y, x + 6 * len(t), y + 10, 0, size=10.0, bold=False) for x, t in cells]

    words = (row(0, [(72, "(Millions)"), (300, "2018"), (420, "2017")])
             + row(16, [(72, "Dividends"), (300, "$"), (312, "(3,193)"), (420, "$"), (432, "(3,236)")])
             + row(32, [(72, "Buybacks"), (300, "(4,870)"), (420, "(2,068)")]))
    b = Block(page=0, bbox=(72, 0, 480, 42), label="Table", lines=[Line(words)], page_w=595, page_h=842)
    header, rows = serialise_table(b)
    assert header == "(Millions) | 2018 | 2017"
    assert rows[0] == "Dividends | 2018 = $(3,193) | 2017 = $(3,236)"
