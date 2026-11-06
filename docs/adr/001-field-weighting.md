# ADR 001: Field-weighted BM25, tuned on graded NDCG

*Drafted by Claude Code from the dev results; see `docs/AI_WORKFLOW.md`.*

**Context.** A product has five text fields of very different length and quality: a short name, a
class, a category path, a long features string and a free-text description. A query word in the name
should count for more than the same word deep in a description.

**Decision.** One bm25s index per field, and a query's score is a weighted sum of the per-field BM25
scores. Weights come from a 324-row grid on the dev split only, every row logged in
[`results/bm25_grid_dev.csv`](../../results/bm25_grid_dev.csv). The winner is in
[`configs/bm25f.yaml`](../../configs/bm25f.yaml), and the rows compared are in
[`results/bm25_dev.md`](../../results/bm25_dev.md).

**This is not BM25F proper.** True BM25F pools term frequencies across fields *before* saturating
them, so a word repeated across fields saturates once. A weighted sum saturates per field, so a word
matching in three fields scores three times. The class is still called `BM25FRetriever` to match the
design, and the README says "field-weighted BM25".

**What the grid showed.**

- The name does carry the most weight, but not by much: the best row gives the name 4x a description's
  weight, and the grid's top rows are within noise of each other.
- The best `description` weight sits at the top edge of its grid. The grid was left alone, because
  widening it after seeing the result is how 320 queries get overfit.
- Graded NDCG and Exact-only NDCG disagree. Plain concatenation (`bm25_all`) loses on graded NDCG but
  beats the tuned weights on Exact-only NDCG and MRR. Tuning on graded gain favours filling the top 10
  with Partial matches. The primary metric stays graded NDCG as planned, and this trade-off is reported.
- Stemming is the largest single lexical gain; see the `_nostem` rows.

**Feature text.** Keys and yes/no flags were removed from the features field, because `mirror : no`
shouldn't match the query "mirror". That argument is about correctness, not score: the
`bm25f_rawfeatures` row shows how little it moved NDCG.

**Consequences.** Five indexes to build and five score arrays per query, which is cheap at this
catalogue size. Weights are tuned to one catalogue and one label distribution, so they aren't portable.
