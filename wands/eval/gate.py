"""The CI relevance gate: compare a run on dev with the committed baseline.

LLM answers come from the committed cache, so a re-run reproduces the same ranking and the gate needs
no noise band. The tolerance only absorbs float differences between machines.
"""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_PATH = Path("results/baseline.json")
MAX_NDCG_DROP = 0.002


def make_report(
    config: str, split: str, sha: str, metrics: dict[str, float], fallbacks: dict[str, int]
) -> dict:
    return {"config": config, "split": split, "sha": sha, "metrics": metrics, "fallbacks": fallbacks}


def check(report: dict, baseline: dict, max_drop: float = MAX_NDCG_DROP) -> tuple[bool, list[str]]:
    """Fail if dev NDCG@10 drops by more than max_drop, or if any fallback count rises."""
    lines = []
    ok = True
    now, then = report["metrics"]["ndcg@10"], baseline["metrics"]["ndcg@10"]
    drop = then - now
    passed = drop <= max_drop
    ok &= passed
    lines.append(
        f"{'ok  ' if passed else 'FAIL'} ndcg@10 {now:.4f} vs baseline {then:.4f} "
        f"(drop {drop:+.4f}, allowed {max_drop})"
    )
    reasons = sorted(set(report["fallbacks"]) | set(baseline["fallbacks"]))
    for reason in reasons:
        count, allowed = report["fallbacks"].get(reason, 0), baseline["fallbacks"].get(reason, 0)
        passed = count <= allowed
        ok &= passed
        lines.append(f"{'ok  ' if passed else 'FAIL'} fallback {reason}: {count} vs baseline {allowed}")
    return ok, lines


def write_baseline(report: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
