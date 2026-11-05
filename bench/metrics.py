"""Retrieval metrics against benchmark gold evidence. No LLM judge anywhere: every score is a deterministic
comparison of retrieved chunks with the dataset's evidence page and evidence text."""
from __future__ import annotations

import re
from collections import Counter

_W = re.compile(r"[a-z0-9]+")


def clean_evidence(text: str) -> str:
    """OHR-Bench evidence is LaTeX (tables as `\\begin{tabular}{|c|c|}`, `\\multicolumn{2}{c}{X}`, math in `$..$`).
    Strip the structure arguments so word-level metrics compare content, not markup."""
    t = re.sub(r"\\begin\{[^}]*\}(\{[^}]*\})?|\\end\{[^}]*\}", " ", text)
    t = re.sub(r"\\multi(column|row)\{[^}]*\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\(mathrm|text|textbf|mathbf|operatorname)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("&", " ").replace("$", " ").replace("{", " ").replace("}", " ").replace("^", " ")
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return _W.findall(text.lower())


def page_relevant(chunk_doc: str, chunk_pages: list[int], gold_doc: str, gold_pages: set[int]) -> bool:
    return chunk_doc == gold_doc and bool(set(chunk_pages) & gold_pages)


def first_relevant_rank(flags: list[bool]) -> int | None:
    return next((i + 1 for i, f in enumerate(flags) if f), None)


def hit_at(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def mrr_at(rank: int | None, k: int = 10) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def evidence_recall(evidence: str, texts: list[str]) -> float:
    """Share of evidence words (multiset) present in the concatenated retrieved texts."""
    want = Counter(words(evidence))
    if not want:
        return 0.0
    have = Counter(w for t in texts for w in words(t))
    return sum(min(n, have[w]) for w, n in want.items()) / sum(want.values())


def lcs_ratio(evidence: str, text: str) -> float:
    """Word-level longest common subsequence of the evidence inside `text`, over evidence length."""
    a, b = words(evidence), words(text)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(a)
