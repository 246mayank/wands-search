"""Error analysis of the final pipeline on test. It reads results only, so it can't affect them.

A worksheet shows the evidence for the worst queries. A person then gives each one a cause in
docs/error_tags.yaml, and the report joins those tags with the generated numbers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from wands.eval.qrels import GAINS
from wands.eval.report import Scored
from wands.pipeline import World
from wands.qu.cache import QUTable

TAGS_PATH = Path("docs/error_tags.yaml")
REPORT_PATH = Path("docs/ERROR_ANALYSIS.md")
CAUSES = (
    "vocabulary mismatch",
    "wrong class predicted",
    "attribute ignored",
    "ambiguous query",
    "label looks inconsistent",
    "unjudged results",
)
LABEL_OF = {gain: label for label, gain in GAINS.items()}


def worst(scored: Scored, n: int = 20) -> list[int]:
    """Positions of the n lowest NDCG@10 scores, skipping queries with no score."""
    ndcg = scored.scores["ndcg@10"]
    defined = np.flatnonzero(~np.isnan(ndcg))
    return defined[np.argsort(ndcg[defined], kind="stable")][:n].tolist()


def worksheet(world: World, scored: Scored, ranked: dict[int, list[int]], table: QUTable) -> str:
    positions = {int(pid): i for i, pid in enumerate(world.catalog.product_ids)}

    def product(pid: int) -> str:
        i = positions[pid]
        return f"{world.catalog.fields['name'][i]} [{world.catalog.fields['class'][i]}]"

    lines = ["# Worst queries on test: evidence for tagging", ""]
    for at in worst(scored):
        qid = scored.query_ids[at]
        judged = world.qrels.get(qid, {})
        qu, reason = table.lookup(world.queries[qid])
        lines += [
            f"## {qid}: {world.queries[qid]}",
            "",
            f"ndcg@10 {scored.scores['ndcg@10'][at]:.3f}, judged@10 {scored.scores['judged@10'][at]:.1f}, "
            f"labelled class `{world.query_classes[qid]}`, labels: {dict(Counter(judged.values()))}",
            "",
            f"QU: {qu.model_dump() if qu else reason}",
            "",
            "Top 5:",
        ]
        for pid in ranked[qid][:5]:
            lines.append(f"- {LABEL_OF.get(judged.get(pid), 'unjudged')}: {product(pid)}")
        missed = [
            pid
            for pid, gain in judged.items()
            if gain == max(judged.values()) and pid not in ranked[qid][:10]
        ]
        lines += ["", "Best-labelled products not in the top 10:"]
        lines += [f"- {product(pid)}" for pid in missed[:3]]
        lines.append("")
    return "\n".join(lines)


def load_tags(path: Path = TAGS_PATH) -> dict[int, dict]:
    tags = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = {t["cause"] for t in tags.values()} - set(CAUSES)
    if bad:
        raise ValueError(f"unknown causes: {sorted(bad)}")
    return tags


def per_class(world: World, scored: Scored, min_size: int = 3) -> list[tuple[str, int, float]]:
    groups: dict[str, list[float]] = {}
    for at, qid in enumerate(scored.query_ids):
        value = scored.scores["ndcg@10"][at]
        if not np.isnan(value):
            groups.setdefault(world.query_classes[qid] or "(none)", []).append(float(value))
    rows = [(name, len(v), float(np.mean(v))) for name, v in groups.items() if len(v) >= min_size]
    return sorted(rows, key=lambda row: row[2])


def render_analysis(world: World, scored: Scored, sha: str, tags: dict[int, dict]) -> str:
    causes = Counter(t["cause"] for t in tags.values())
    lines = [
        "# Error analysis",
        "",
        f"Final pipeline on the {len(scored.query_ids)} test queries, commit `{sha}`. Numbers are generated "
        "by `wands error-analysis`. The cause of each query is a judgement in `docs/error_tags.yaml`, made "
        "after the single test run, and changed nothing about it. Tags were drafted by Claude Code from the "
        "evidence shown; see `docs/AI_WORKFLOW.md`.",
        "",
        "## NDCG@10 by labelled class, 3+ queries, worst first",
        "",
        "| class | queries | ndcg@10 |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {n} | {value:.4f} |" for name, n, value in per_class(world, scored)]
    lines += ["", "## Causes among the 20 worst", "", "| cause | queries |", "|---|---|"]
    lines += [f"| {cause} | {count} |" for cause, count in causes.most_common()]
    lines += ["", "## The 20 worst", "", "| query | ndcg@10 | cause | evidence |", "|---|---|---|---|"]
    for at in worst(scored):
        qid = scored.query_ids[at]
        tag = tags[qid]
        text = world.queries[qid].replace("|", "\\|")
        lines.append(f"| {text} | {scored.scores['ndcg@10'][at]:.3f} | {tag['cause']} | {tag['evidence']} |")
    return "\n".join(lines) + "\n"
