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

## 2026-11-13 · retrieval ablations: vector_only, bm25_only, rerank_none, topk20, topk50

Code changes before these runs:

- **Bug fix:** BM25 was loaded with the `similarity_top_k` saved when the index was built (10), so changing TOP_K only widened the vector side. BM25 now uses the current TOP_K. Runs at TOP_K=10 are unaffected.
- New switches `RETRIEVAL_MODE` (hybrid / vector / bm25) and `RERANK_MODE=none`, plus `RERANK_DEVICE`.

Each run changes one thing relative to citation_fix (hybrid, top_k=10, bge-reranker-base, top_n=3).

| Run | Hit@5 | MRR@10 | Cite prec. | Number | Text | False refusals | p50 / p95 s |
|---|---|---|---|---|---|---|---|
| citation_fix (hybrid, k=10) | 88.9% | 0.815 | 51.9% | 84.2% | 75.0% | 14.8% | 1.8 / 2.2 |
| vector_only | 51.9% | 0.468 | 28.4% | 42.1% | 50.0% | 48.1% | 1.7 / 2.4 |
| bm25_only | 88.9% | 0.843 | 53.1% | 89.5% | 100% | 7.4% | 1.7 / 2.2 |
| rerank_none | 81.5% | 0.552 | 39.5% | 63.2% | 62.5% | 33.3% | 1.3 / 1.6 |
| **topk20** | **92.6%** | 0.830 | 55.6% | **94.7%** | **100%** | **3.7%** | 2.1 / 2.8 |
| topk50 | 92.6% | 0.902 | 56.8% | 94.7% | 100% | 3.7% | 5.4 / 6.5 * |

\* topk50 ran the reranker on CPU because MPS ran out of memory at 50 candidates, so its latency can't be compared with the other runs.

Findings:

1. **The vector side is weak.** Vector-only gets Hit@5 51.9% and 42.1% number accuracy. The likely cause is that `bge-small-zh-v1.5` is a Chinese embedding model and the reports are in English. That's the next thing to test (an English or multilingual embedding).
2. **BM25 alone beats the hybrid at top_k=10** (number 89.5% vs 84.2%, text 100% vs 75%). With only 10 fused slots, weak vector hits take places that good BM25 hits would otherwise get. The hybrid still contributes something: a06 is correct with vector-only and wrong with BM25-only.
3. **The reranker matters a lot.** Turning it off drops MRR from 0.815 to 0.552 and number accuracy from 84.2% to 63.2%.
4. **TOP_K=20 fixes the hybrid.** It gives 4 more correct answers than top_k=10 (h02, a05, h10, a13) and cuts false refusals from 4 to 1. More candidates let the reranker reach the right page: for h02 it now sends p.10 (Financial Highlights) instead of two copies of p.232. TOP_K=50 matches TOP_K=20 on accuracy (MRR 0.902 vs 0.830), but the reranker takes about 2.5 times as long and needs more memory than MPS had. **Adopt TOP_K=20.**
5. The only question still wrong is h08 (HKEX total assets, which is only on the balance sheet p.145). It isn't retrieved even at TOP_K=50, so it's a table-chunking problem, not a candidate-count one.

Caveats: on 30 questions each question is worth 3.3 points (5.3 points on the 19 number questions), so +10.5 points of number accuracy means 2 questions. h10 was answered from p.92, which states the chairman but isn't in its gold_pages, so gold page lists can still miss valid sources. Per plan.md, top_n=3 was tuned at top_k=10, so it should be rechecked at top_k=20.

## 2026-11-13 · embed_en, en_vector_only

Change: EMBED_MODEL `BAAI/bge-small-zh-v1.5` → `BAAI/bge-small-en-v1.5`. Both are the same size (small), the chunks are identical (1,540), and LlamaIndex adds each model's own BGE query instruction. The English index was built in `storage/en/`, so the original index is untouched.

| Run | Embedding | Retrieval | Hit@5 | MRR@10 | Number | Text | False refusals |
|---|---|---|---|---|---|---|---|
| vector_only | zh | vector, k=10 | 51.9% | 0.468 | 42.1% | 50.0% | 48.1% |
| en_vector_only | **en** | vector, k=10 | **85.2%** | **0.768** | **84.2%** | **100%** | **11.1%** |
| topk20 | zh | hybrid, k=20 | 92.6% | 0.830 | 94.7% | 100% | 3.7% |
| embed_en | **en** | hybrid, k=20 | 92.6% | **0.886** | 94.7% | 87.5% | 3.7% |

Findings:

1. **The Chinese embedding model was the weak point.** On the English reports, switching to the English model raises vector-only Hit@5 from 51.9% to 85.2% and number accuracy from 42.1% to 84.2%. On its own, vector search now nearly matches BM25.
2. **In the hybrid, ranking improves but answers don't change.** MRR rises from 0.830 to 0.886 and latency is the same. Number accuracy stays at 94.7% because BM25 plus the reranker had already rescued those questions at k=20.
3. The drop in text accuracy (100% → 87.5%) is a single question, a13, and it's **judge noise, not a retrieval change**. The answers are the same in substance (75% of net FSG, without the "regular review" clause). In one run the judge called that "correct … though it omits the regular review"; in another it called it "partially_correct". So the judge is inconsistent on partial answers at temperature 0 across slightly different contexts.

Decision: adopt the English embedding. Ranking is better at no extra cost, and the vector side is no longer a liability if BM25 misses (such as paraphrased questions or the planned Chinese-question support).

Follow-up on eval reliability: a13 shows the judge isn't stable on partial answers. Options are a clearer rubric (list the required facts), a narrower reference answer, or reporting text accuracy as correct + 0.5 × partial. This should be settled before small text-accuracy differences are taken seriously.
