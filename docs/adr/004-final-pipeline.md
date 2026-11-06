# ADR 004: The final pipeline is hybrid + soft QU, with nothing on top

*Drafted by Claude Code from the dev results; see `docs/AI_WORKFLOW.md`.*

**Decision.** `configs/final.yaml` is field-weighted BM25 + dense, fused with RRF, then a soft QU boost.
Everything else that was tried stays out, because on dev none of it beat this pipeline by more than noise.

| Tried on dev | Result | Report |
|---|---|---|
| Cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) | Hurts alone, and is noise when blended by RRF. It was trained on web passages, not product titles, and it adds latency for no gain. | [rerank_dev](../../results/rerank_dev.md), [latency](../../results/latency.md) |
| Popularity prior (`rating_count`) | Noise at every weight tried. | [popularity_dev](../../results/popularity_dev.md) |
| Learning to rank (LightGBM) | Ties when unjudged rows count as gain 0, and loses when they are dropped. | [ltr_dev](../../results/ltr_dev.md) |
| Query rewriting | Under the plan's rule almost nothing changes. Rewriting every query fixes some typos and damages others, for no net gain. | [rewrite_dev](../../results/rewrite_dev.md), [rewrite_all_dev](../../results/rewrite_all_dev.md) |
| Prompt v2 | No better than v1, and it adds a fallback. | [prompts/CHANGELOG.md](../../prompts/CHANGELOG.md) |

**Why a tie doesn't justify LTR.** The tuned pipeline has three weights a person can read, and a deterministic
gate. LTR adds a trained artefact that has to be retrained, and its dev score needs cross-validation
because dev is its training set. It also learns from a biased pool: the variant that treats unjudged
products as irrelevant raises judged@10, which suggests it partly learned what annotators were shown
rather than what is relevant. With no significant gain, the simpler system stays. LTR still gets one
row in the test table, so that this decision can be checked.

**Consequences.** The system ships no reranker, so a claim about reranking gains would be false. The
plan's resume template mentions one in brackets; that part is dropped.
