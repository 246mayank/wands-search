# ADR 002: Reciprocal rank fusion over a tuned weighted sum

*Drafted by Claude Code from the dev results; see `docs/AI_WORKFLOW.md`.*

**Context.** BM25 and dense retrieval fail differently: BM25 misses vocabulary mismatches, and dense
retrieval misses exact model names. Their scores aren't comparable, since BM25 is unbounded and depends
on the query, while cosine similarity sits in a narrow band.

**Options.** Reciprocal rank fusion uses ranks only. A weighted sum min-max normalises each list's
top 200, then needs an alpha tuned on dev, logged in
[`results/fusion_grid_dev.csv`](../../results/fusion_grid_dev.csv).

**Decision.** RRF with k = 60. In [`results/fusion_dev.md`](../../results/fusion_dev.md) the two tie on
NDCG@10, and RRF has no parameter tuned on these 320 queries, so it has less room to overfit. Given a
tie, the method with fewer tuned numbers wins.

**When normalisation breaks.** Min-max uses each list's own best and worst score. One outlier at the top
of a BM25 list, such as an exact model-name hit, squashes every other product towards 0, and the same
raw score means different things on different queries. RRF only sees order, so it can't suffer this. It
pays by throwing away how far ahead a result was.

**Consequences.** The weighted sum had the better capped recall in that table, so a recall-first stage
might prefer it. The QU boost needs scores in a known range, so it min-max normalises the RRF scores
per query; that is safe because RRF scores have no outliers.
