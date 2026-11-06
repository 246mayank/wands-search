# ADR 003: A soft boost from an offline LLM, not a filter, and not in the request path

*Drafted by Claude Code from the dev results; see `docs/AI_WORKFLOW.md`.*

## The LLM stays out of the request path

**Context.** An LLM call costs seconds, can fail, and costs money per query. Search can't wait for it.

**Decision.** Query understanding runs offline, once per query, and its validated output is committed in
`cache/qu_cache.jsonl`. The online path only does a dictionary lookup. A query that isn't in the cache,
or whose answer failed validation, gets no boost and a reason code, and the rest of the pipeline runs
unchanged. This mirrors how head queries can be precomputed, and tail queries simply degrade to
hybrid retrieval.

**What this buys.** Latency doesn't depend on the LLM ([`results/latency.md`](../../results/latency.md)).
The CI gate is deterministic and needs no API key. A prompt change alters the cache key, so it can't
slip through: it misses the cache, the fallback count rises and the gate fails until someone re-records.

**What it costs.** A tail query seen for the first time gets nothing. Serving those would need a
background job that fills the cache, or a small local model. Neither is built here.

## A soft boost, not a filter

**Context.** Query understanding is sometimes wrong. Any-of-3 class accuracy is high but not perfect
([`results/qu_dev.md`](../../results/qu_dev.md)), and some labelled classes are themselves odd.

**Decision.** Add to a product's score for a class or attribute match, and drop nothing. Low confidence
halves the bonus. The weights come from a logged grid on dev.

**Evidence.** The filter row in [`results/qu_dev.md`](../../results/qu_dev.md) loses recall@100 and
gains less NDCG than the boost. A wrong class in a filter removes correct products for good; in a boost
it only moves them down, and they can still be found. When a filter would empty the page, the prediction
is ignored.

**Why the LLM can't invent a class.** It chooses from a per-query candidate list of about 18 catalogue
classes, built without labels, and any other answer is refused. Candidate recall on dev is reported
with the scores, because it is the ceiling for the LLM.

**Where it helps and hurts.** See [`results/qu_breakdown_dev.md`](../../results/qu_breakdown_dev.md).
