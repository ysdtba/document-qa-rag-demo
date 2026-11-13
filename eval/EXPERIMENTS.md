# Experiment log

What each run changed, what happened, and why. All numbers come from `eval/results/*.json`; the full table is in `RESULTS.md`. Every run is on the 30-question dev split, so one question equals 3.3 points.

## 2026-11-13 · baseline

Stage 1 configuration: per-page pypdf text, SentenceSplitter 384/48, bge-small-zh-v1.5, hybrid BM25 + vector with RRF (top_k=10), bge-reranker-base, top_n=3, deepseek-flash with thinking off as generator and judge.

Hit@5 88.9%, MRR@10 0.815, number 84.2%, text 75.0% / 12.5% partial, not-found 100%, faithfulness 100%.

Of the five wrong answers, four are refusals after a retrieval miss (h02, h08, h10, a05) and one is a partial text answer (a13). The model never made up a figure.

## 2026-11-13 · citation_fix

Change: the system prompt's citation example `(HKEX FY2025 p.12)` became `(<company> FY<year> p.<page>)`.

Before the fix, 8 AIA answers cited "HKEX"; after it, none did. Every score stayed the same and no question flipped.

## 2026-11-13 · topn5, topn8

Change: RERANK_TOP_N, the number of chunks sent to the LLM, went 3 → 5 → 8.

| | top_n=3 | 5 | 8 |
|---|---|---|---|
| Number / text accuracy | 84.2% / 75.0% | same | same |
| Citation precision | 51.9% | 40.0% | 29.6% |
| Prompt tokens per question | 1,054 | 1,570 | 2,255 |
| Cost per question | $0.00037 | $0.00049 | $0.00070 |

No question changed. More context roughly doubled the cost and diluted citation precision, with no gain in accuracy. **Keep top_n=3.**

Why h02 (HKEX profit attributable to shareholders) still failed even though its gold pages p.83 and p.75 reached the model at top_n=5:

- The p.83 chunk that was retrieved is the footnote half of the page. The 10-year table holding 17,754 is in a different chunk, but the page still counts as a gold hit.
- The p.75 chunk does contain 17,754, but it's the tail of a quarterly table. Its column headers ("Q1 2025 … Total 2025") ended up in the previous chunk, and the next table's 2024 headers follow straight after. The model can't tell which year the number belongs to, so it correctly refuses.

Takeaways:

1. **The bottleneck is how tables are chunked, not how many chunks are sent.** Splitting tables away from their headers makes the numbers unreadable. This is the case for the "tables as their own chunks" experiment, which should come next.
2. **Page-level Hit@5 overstates retrieval quality.** A hit only means some chunk from a gold page was retrieved, not the chunk holding the answer. Worth reporting alongside answer accuracy rather than on its own.
