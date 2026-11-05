"""Synthetic PDFs with known structure, built with PyMuPDF (no fixtures to download)."""
from __future__ import annotations

import fitz
import pytest

LOREM = ("The committee reviewed the operating results for the year and noted that throughput improved across "
         "all sites while unit costs declined. Maintenance windows were shortened by scheduling work in parallel. ")


def _page(doc: fitz.Document, n: int) -> fitz.Page:
    p = doc.new_page(width=595, height=842)
    p.insert_text((72, 40), "Northwind Logistics - Annual Operations Review", fontsize=8, fontname="helv")
    p.insert_text((280, 815), f"Page {n}", fontsize=8, fontname="helv")
    return p


def _heading(p: fitz.Page, y: float, text: str, size: float) -> float:
    p.insert_text((72, y), text, fontsize=size, fontname="hebo")
    return y + size + 10


def _body(p: fitz.Page, y: float, text: str, height: float = 120) -> float:
    """Write a wrapped paragraph; the box grows until the text fits (measured on a scratch page)."""
    scratch = fitz.open().new_page(width=595, height=842)
    while scratch.insert_textbox(fitz.Rect(72, 0, 523, height), text, fontsize=10, fontname="helv") < 0:
        height += 20
    rc = p.insert_textbox(fitz.Rect(72, y, 523, y + height), text, fontsize=10, fontname="helv")
    assert rc >= 0, "paragraph runs off the page"
    return y + height - rc + 12


def build_report(path) -> None:
    doc = fitz.open()
    p = _page(doc, 1)
    p.insert_text((72, 120), "Annual Operations Review 2025", fontsize=22, fontname="hebo")
    y = _heading(p, 200, "Contents", 14)
    for line in ("1. Introduction ........ 2", "1.1 Scope ........ 2", "2. Results ........ 3",
                 "2.1 Regional results ........ 4", "3. Outlook ........ 5"):
        p.insert_text((72, y), line, fontsize=10, fontname="helv")
        y += 16

    p = _page(doc, 2)
    y = _heading(p, 90, "1. Introduction", 14)
    y = _body(p, y, LOREM * 2, 70)
    y = _heading(p, y + 6, "1.1 Scope", 12)
    y = _body(p, y, "This review covers the warehouses in the northern and southern regions. " + LOREM, 60)
    y = _heading(p, y + 6, "1.2 Definitions", 12)
    _body(p, y, "Throughput means pallets shipped per operating hour. " + LOREM, 60)

    p = _page(doc, 3)
    y = _heading(p, 90, "2. Results", 14)
    y = _body(p, y, LOREM, 50)
    rows = [("Item", "2024", "2025"), ("Revenue", "4,830", "5,120"), ("Operating cost", "2,100", "2,300"),
            ("Pallets shipped", "91,400", "98,250")]
    ty = y + 20
    for r in rows:
        font = "hebo" if r[0] == "Item" else "helv"
        p.insert_text((72, ty), r[0], fontsize=10, fontname=font)
        p.insert_text((300, ty), r[1], fontsize=10, fontname=font)
        p.insert_text((420, ty), r[2], fontsize=10, fontname=font)
        ty += 16

    p = _page(doc, 4)
    y = _heading(p, 90, "2.1 Regional results", 12)
    _body(p, y, ("In the northern region the cold-chain facility exceeded its target. " + LOREM) * 9, 640)

    p = _page(doc, 5)
    y = _heading(p, 90, "3. Outlook", 14)
    _body(p, y, "Next year the network adds two cross-docking hubs. " + LOREM, 60)
    doc.save(path)


def build_short(path) -> None:
    """Two pages that share a top-band line: with a ratio-only furniture rule this line would be deleted."""
    doc = fitz.open()
    for i in range(2):
        p = doc.new_page(width=595, height=842)
        p.insert_text((72, 40), "Section 4 applies to all contracts", fontsize=10, fontname="helv")
        _body(p, 100, LOREM, 60)
    doc.save(path)


@pytest.fixture(scope="session")
def report_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "report.pdf"
    build_report(path)
    return path


@pytest.fixture(scope="session")
def short_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "short.pdf"
    build_short(path)
    return path


@pytest.fixture(scope="session")
def report_result(report_pdf):
    from layout_rag import ChunkerConfig, DocLayoutChunker

    cfg = ChunkerConfig(layout="heuristic", tokenizer=None, max_tokens=256, min_tokens=20)
    return DocLayoutChunker(cfg).chunk_pdf(report_pdf)
