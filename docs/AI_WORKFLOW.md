# How AI tools were used on this project

This project was built with Claude Code. This file records who did what, and every time a test, a
review or a metric caught a mistake the AI made. Entries are real events, written when they happened.

## Who did what

| Piece | Designed by | Written by |
|---|---|---|
| Project plan: eval protocol, split, schema, gate rule, wording rules | Mayank | Mayank (`plan.md`, kept outside the repo) |
| Code, tests, configs | plan by Mayank | Claude Code, reviewed by Mayank |
| Data decisions forced by the profile (duplicate labels, undefined metrics, capped recall) | Claude Code proposed, Mayank reviewed | Claude Code |

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

## How the tests and the gate make AI-written changes safe to merge

- The metrics are checked against hand-computed cases and against `ranx`, so a wrong formula fails
  a test rather than quietly changing every number.
- Every result names its config and commit, and reports are generated, so a claim can be traced to
  the code that produced it.
