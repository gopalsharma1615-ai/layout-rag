<!-- Reference evaluation of 2026-09-21, produced by the original implementation of this design. -->
> Reference run, 2026-09-21. Subsets are whole documents in sha1(doc name) order. Reproduce with `bench/run_pdf_retrieval.py` (see the README).

# Chunking + RAG retrieval evaluation

Embedder: BAAI/bge-large-en-v1.5 (cosine, AUTOINDEX). No LLM anywhere: every score compares retrieved chunks against the benchmark's gold evidence.

## Text chunking - Chroma chunking benchmark (5 corpora, 472 questions)

Character-level overlap between retrieved chunks and the highlighted answer excerpts. precision_omega = IoU with perfect retrieval (the ceiling the chunk boundaries allow).

| collection | chunker | chunks | words p50 | recall@1 | recall@5 | recall@10 | precision@5 | IoU@5 | precision_omega | % chunks > embed window |
|---|---|---|---|---|---|---|---|---|---|---|
| word_budget_512_300 | chroma text -> word-budgeted chunker 512/300 (words, not tokens) | 542 | 452 | 52.1 | 84.6 | 91.9 | 1.9 | 1.9 | 10.1 | 71.8 |
| fixed_256_tokens | chroma text -> fixed 256 tokens, overlap 0 | 1246 | 187 | 48.7 | 84.6 | 91.5 | 3.9 | 3.9 | 17.5 | 0.0 |

Equal k is not equal context when chunk sizes differ -- recall against the words actually handed to the generator:

| collection | k=1: recall / words | k=3: recall / words | k=5: recall / words | k=10: recall / words |
|---|---|---|---|---|
| word_budget_512_300 | 52.1 / 439 | 76.9 / 1321 | 84.6 / 2202 | 91.9 / 4383 |
| fixed_256_tokens | 48.7 / 200 | 76.7 / 599 | 84.6 / 996 | 91.5 / 1978 |

Recall@5 by corpus:

| collection | chatlogs | finance | pubmed | state_of_the_union | wikitexts |
|---|---|---|---|---|---|
| word_budget_512_300 | 100.0 | 72.8 | 73.0 | 87.7 | 93.1 |
| fixed_256_tokens | 99.9 | 77.0 | 72.0 | 94.5 | 87.3 |

## OHR-Bench - PDF -> OCR -> chunking -> retrieval (7 domains)

Relevant chunk = right document AND its page span covers the gold evidence page. `ohr_lcs@2` is OHR-Bench's official retrieval metric (LCS of the evidence inside the relevant top-2 chunks); `macro` averages the 7 domains as the paper does.

OHR-Bench is not English-only: 215 of its documents (news, textbook) and 883 questions are Chinese. This stack is English (Tesseract `eng`, bge-large-en), so the **en** rows are the headline and the **zh** rows show what it cannot do by construction.

### scope = category (search the question's domain - OHR-Bench protocol)

| collection | pipeline | lang | questions | page_hit@1 | page_hit@5 | page_hit@10 | MRR@10 | ohr_lcs@2 | ohr_lcs@2 macro | evidence_recall@5 | answer_hit@5 | ctx words@5 | evidence page chunked | GT words kept | pages chunked | chunks | % > embed window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 313 PDFs / 1120 pages (subset) | en | 1298 | 65.0 | 85.5 | 89.8 | 0.735 | 53.0 | 54.5 | 70.7 | 39.7 | 1069 | 94.5 | 70.7 | 87.8 | 3951 | 14.9 |
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 313 PDFs / 1120 pages (subset) | zh | 314 | 3.5 | 14.6 | 24.2 | 0.083 | 1.0 | - | 1.9 | 1.9 | 804 | 95.5 | 4.4 | 87.8 | 3951 | 14.9 |
| gt_text_control | ohrbench-gt text -> word-budgeted chunker 512/300 (words, not tokens) | en | 1298 | 73.7 | 95.5 | 98.4 | 0.831 | 79.8 | 80.6 | 87.7 | 55.2 | 1584 | 100.0 | 95.5 | 95.6 | 1618 | 44.7 |
| gt_text_control | ohrbench-gt text -> word-budgeted chunker 512/300 (words, not tokens) | zh | 314 | 26.8 | 54.5 | 67.5 | 0.382 | 12.4 | - | 15.2 | 8.4 | 633 | 100.0 | 93.8 | 95.6 | 1618 | 44.7 |

### scope = global (search all 7 domains)

| collection | pipeline | lang | questions | page_hit@1 | page_hit@5 | page_hit@10 | MRR@10 | ohr_lcs@2 | ohr_lcs@2 macro | evidence_recall@5 | answer_hit@5 | ctx words@5 | evidence page chunked | GT words kept | pages chunked | chunks | % > embed window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 313 PDFs / 1120 pages (subset) | en | 1298 | 64.2 | 84.5 | 88.4 | 0.725 | 52.5 | 54.0 | 69.9 | 39.0 | 1093 | 94.5 | 70.7 | 87.8 | 3951 | 14.9 |
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 313 PDFs / 1120 pages (subset) | zh | 314 | 1.9 | 7.0 | 11.5 | 0.040 | 0.8 | - | 1.2 | 1.5 | 878 | 95.5 | 4.4 | 87.8 | 3951 | 14.9 |
| gt_text_control | ohrbench-gt text -> word-budgeted chunker 512/300 (words, not tokens) | en | 1298 | 72.6 | 94.5 | 97.5 | 0.820 | 78.7 | 79.5 | 86.7 | 54.8 | 1612 | 100.0 | 95.5 | 95.6 | 1618 | 44.7 |
| gt_text_control | ohrbench-gt text -> word-budgeted chunker 512/300 (words, not tokens) | zh | 314 | 22.3 | 41.4 | 50.3 | 0.300 | 11.8 | - | 13.7 | 7.2 | 482 | 100.0 | 93.8 | 95.6 | 1618 | 44.7 |

### By evidence type (English, scope = category): ohr_lcs@2 / page_hit@5

| collection | text | table | formula | chart | reading_order |
|---|---|---|---|---|---|
| layout_pipeline | 58.8 / 81.4 (n=614) | 51.2 / 88.0 (n=309) | 47.6 / 89.1 (n=239) | 29.8 / 88.5 (n=96) | 74.4 / 100.0 (n=19) |
| gt_text_control | 80.0 / 95.8 (n=614) | 83.9 / 96.4 (n=309) | 80.5 / 96.7 (n=239) | 67.9 / 86.5 (n=96) | 73.8 / 100.0 (n=19) |

### By domain (English, scope = category): ohr_lcs@2 / page_hit@5 / evidence page chunked

| collection | academic | administration | finance | law | manual | news | textbook |
|---|---|---|---|---|---|---|---|
| layout_pipeline | 52.2 / 90.8 / 100.0 (n=153) | 61.4 / 94.8 / 100.0 (n=153) | 46.6 / 85.2 / 100.0 (n=278) | 70.8 / 93.6 / 100.0 (n=157) | 52.3 / 75.2 / 81.4 (n=113) | 58.6 / 93.1 / 97.5 (n=202) | 39.9 / 69.8 / 81.4 (n=242) |
| gt_text_control | 75.5 / 95.4 / 100.0 (n=153) | 83.5 / 99.4 / 100.0 (n=153) | 73.2 / 92.1 / 100.0 (n=278) | 84.1 / 96.2 / 100.0 (n=157) | 86.6 / 98.2 / 100.0 (n=113) | 79.7 / 95.0 / 100.0 (n=202) | 81.7 / 95.9 / 100.0 (n=242) |

### Published OHR-Bench retrieval results (for orientation only)

Different stack: all 8,498 questions incl. Chinese, multilingual retrievers (averaged over BGE-M3 and BM25), 1024-token page-bounded chunks, full-size domain indexes. Compare against this suite's own ground-truth control for a like-for-like reading.

| OCR | TXT | TAB | FOR | CHA | RO | ALL |
|---|---|---|---|---|---|---|
| Ground-truth text | 81.6 | 69.8 | 75.2 | 70.3 | 9.8 | 70.4 |
| Azure Document Intelligence | 78.0 | 59.4 | 55.2 | 45.2 | 5.8 | 60.6 |
| Qwen2.5-VL-72B | 75.1 | 60.0 | 60.0 | 38.2 | 5.3 | 59.6 |
| Nanonets-OCR-s | 71.8 | 59.8 | 57.4 | 43.7 | 4.4 | 58.3 |
| Marker-1.2.3 | 75.5 | 58.2 | 55.5 | 20.0 | 5.9 | 57.0 |
| olmOCR-7B | 72.5 | 58.4 | 55.4 | 24.8 | 5.0 | 56.6 |
| MinerU-0.9.3 | 68.1 | 48.6 | 51.3 | 16.5 | 5.9 | 50.5 |
| GOT | 62.5 | 41.1 | 49.0 | 17.4 | 3.7 | 45.8 |
| Nougat | 59.5 | 32.8 | 44.3 | 11.3 | 4.4 | 41.2 |

## FinanceBench - long SEC filings (the benchmark: 84 filings, 150 questions)

### scope = doc (search only the question's filing)

| collection | pipeline | questions | page_hit@1 | page_hit@5 | page_hit@10 | MRR@10 | doc_hit@5 | evidence_recall@5 | evidence_recall@10 | ctx words@5 | evidence page chunked | pages chunked | chunks | % > embed window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 17 PDFs / 2187 pages (subset) | 65 | 27.7 | 46.2 | 63.1 | 0.371 | 100.0 | 53.1 | 66.3 | 1280 | 84.6 | 93.0 | 6476 | 27.1 |

### scope = global (search every filing in the collection)

| collection | pipeline | questions | page_hit@1 | page_hit@5 | page_hit@10 | MRR@10 | doc_hit@5 | evidence_recall@5 | evidence_recall@10 | ctx words@5 | evidence page chunked | pages chunked | chunks | % > embed window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layout_pipeline | layout pipeline, ocr/tesseract, tables=tatr, budget 512/300; 17 PDFs / 2187 pages (subset) | 65 | 18.5 | 38.5 | 52.3 | 0.277 | 80.0 | 42.4 | 54.2 | 1380 | 84.6 | 93.0 | 6476 | 27.1 |

### By question type (scope = doc): page_hit@5 / evidence_recall@5

| collection | domain-relevant | metrics-generated | novel-generated |
|---|---|---|---|
| layout_pipeline | 30.6 / 42.4 | 100.0 / 57.5 | 64.3 / 66.6 |

