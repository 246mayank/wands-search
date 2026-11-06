# Prompt changelog

Dev split only, model `gemini-3.5-flash-lite` unless stated. The test split runs once, after the freeze.
Scores are in the linked reports, which are generated.

| version | date | change | outcome | report |
|---|---|---|---|---|
| `qu_v1` | 2026-11-06 | First prompt: classes from the candidates, attributes the query states, intent, confidence. | **Frozen.** | [v1](../results/qu_gemini-3.5-flash-lite_v1_dev.md), [ranking](../results/qu_dev.md) |
| `qu_v2` | 2026-11-06 | Written from v1's misses: a brand isn't an attribute (`style: stoneford`), normalise values (`blk` to `black`), dimensions aren't attributes, prefer a specific class. | Class accuracy moved by one query, ranking was no better, and it added one fallback. Rejected, and the reason cleaner attributes didn't help ranking wasn't investigated. | [v2](../results/qu_gemini-3.5-flash-lite_v2_dev.md), [ranking](../results/qu_dev.md) |
| `judge_v1` | 2026-11-06 | Rubric taken from WANDS's annotation guidelines, including its examples. | Calibrated once on 600 dev pairs; not tuned. | [report](../results/judge_gemini-3.5-flash-lite_v1_dev.md) |
| `rewrite_v1` | 2026-11-06 | Fix spelling and abbreviations, keep every detail and brand. | No gain, so not in the final pipeline. | [unsure queries](../results/rewrite_dev.md), [all queries](../results/rewrite_all_dev.md) |
