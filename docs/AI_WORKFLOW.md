# How AI tools were used on this project

This project was built with Claude Code. This file records who did what, and every time a test, a
review or a metric caught a mistake the AI made. Entries are real events, written when they happened.

## Who did what

| Piece | Designed by | Written by |
|---|---|---|
| Project plan: eval protocol, split, schema, gate rule, wording rules | Mayank | Mayank (`plan.md`, kept outside the repo) |
| Code, tests, configs | plan by Mayank | Claude Code, reviewed by Mayank |
| Data decisions forced by the profile (duplicate labels, undefined metrics, capped recall) | Claude Code proposed, Mayank reviewed | Claude Code |
| ADRs in `docs/adr/` | Decisions follow the plan and the dev results | Drafted by Claude Code; each file says so |
| Demo page in `web/` | Plan by Mayank | Drafted by Claude Code, not hand-written |

## Mistakes caught

### 2026-11-06: wrong CSV quoting on `query.csv`

- **What was wrong:** Claude read all three WANDS files with quoting off, and wrote a code comment
  saying descriptions contain stray double quotes. They contain none. Three queries came through with
  literal CSV escaping in the text, such as `"writing desk 48"""`, which would have reached the
  tokeniser and the LLM prompt.
- **How it was caught:** a random sample of queries printed during the M1 data profile showed the
  query, and a comparison with pandas' default parsing (what WANDS's own notebook uses) found exactly
  three rows that differed.
- **Fix:** quoting on for `query.csv`, off for `product.csv`, and the false comment removed.

### 2026-11-06: a test that could not fail

- **What was wrong:** Claude wrote two assertions with an escape hatch: one ended `... or True`, and
  another ended `... or stats[OK] >= 1` after the exact comparison it wanted had failed. Both would pass
  whatever the code did.
- **How it was caught:** the first failed honestly, and reading why showed the test's script was wrong,
  not the code. Re-reading the test file before running it caught the second.
- **Fix:** both tests rewritten to assert exact values, before either reached a commit.

### 2026-11-06: learning to rank scored worse for a reason unrelated to ranking

- **What was wrong:** Claude trained LightGBM on judged candidates only, with a sensible-sounding comment:
  an unjudged product isn't known to be irrelevant. The model then pushed unjudged products up, and the
  metric scores those as 0, so the cross-validated score came out well below the tuned pipeline. Taken
  at face value, that would have been reported as "LTR doesn't work here".
- **How it was caught:** `judged@10` fell from 0.89 to 0.85 in the same report. That metric is in the
  harness because the plan asked for it, for exactly this kind of bias.
- **Fix:** both training variants are reported in `results/ltr_dev.md`, with the caveat that the better
  one may partly learn the annotation pool. Commit `423bc9d` onwards.

## How the tests and the gate make AI-written changes safe to merge

- The metrics are checked against hand-computed cases and against `ranx`, so a wrong formula fails
  a test rather than quietly changing every number.
- Every result names its config and commit, and reports are generated, so a claim can be traced to
  the code that produced it.
