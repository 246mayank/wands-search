"""Fill the marked blocks of README.md from generated reports, so no number there is typed by hand."""

from __future__ import annotations

import re
from pathlib import Path

RESULTS = Path("results")
README = Path("README.md")
NOT_RUN = "*Not run yet: the test split runs once, after the `freeze-v1` tag.*"


def tables(path: Path) -> list[str]:
    """Every markdown table in a report, in order."""
    found, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            found.append("\n".join(current))
            current = []
    if current:
        found.append("\n".join(current))
    return found


def row(table: str, name: str) -> list[str]:
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == f"`{name}`":
            return cells
    raise KeyError(name)


def column(table: str, header: str) -> int:
    cells = [cell.strip() for cell in table.splitlines()[0].strip("|").split("|")]
    return cells.index(header)


def cell(table: str, name: str, header: str) -> str:
    return row(table, name)[column(table, header)]


def commit_of(path: Path) -> str:
    match = re.search(r"commit `([^`]+)`", path.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def linked(path: Path) -> str:
    return f"[`{path.name}`]({path.as_posix()})"


def latest_test() -> Path | None:
    found = sorted(RESULTS.glob("test_*.md"))
    return found[-1] if found else None


def headline() -> str:
    path = latest_test()
    if path is None:
        return NOT_RUN
    table = tables(path)[0]
    start, end = cell(table, "bm25_name", "ndcg@10"), cell(table, "final", "ndcg@10")
    queries = re.search(r"(\d+) queries", path.read_text(encoding="utf-8")).group(1)
    hybrid_table = tables(path)[1]
    qu_delta = row(hybrid_table, "final")[-1]
    return (
        f"**NDCG@10 {start} (BM25 on product names) to {end} (final pipeline) on {queries} held-out WANDS "
        f"queries, run once after freezing.** Query understanding's share of that, against hybrid retrieval "
        f"alone: {qu_delta}, 95% CI in brackets. Commit `{commit_of(path)}`, report {linked(path)}."
    )


def test_table() -> str:
    path = latest_test()
    if path is None:
        return NOT_RUN
    return f"{tables(path)[0]}\n\nFrom {linked(path)}. `*`: the 95% paired-bootstrap CI excludes 0."


def dev_table(name: str) -> str:
    path = RESULTS / name
    return f"{tables(path)[0]}\n\nDev split, commit `{commit_of(path)}`, from {linked(path)}."


def qu_scores() -> str:
    lines = []
    for model, label in (("gemini-3.5-flash-lite", "Gemini"), ("qwen2.5-3b", "Qwen 3B")):
        path = RESULTS / f"qu_{model}_v1_dev.md"
        if path.exists():
            scores = dict(
                [cells.strip() for cells in line.strip("|").split("|")][:2]
                for line in tables(path)[0].splitlines()[2:]
            )
            lines.append(
                f"| {label} | {scores['candidate recall (ceiling)']} | {scores['top-1 class correct']} "
                f"| {scores['any of up to 3 classes correct']} | {scores['fallbacks']} | {linked(path)} |"
            )
    head = "| model | candidate recall | top-1 | any of 3 | fallbacks | report |\n|---|---|---|---|---|---|"
    return head + "\n" + "\n".join(lines) + "\n\nDev split, prompt `v1`, vs labelled query classes."


def judge_scores() -> str:
    path = RESULTS / "judge_gemini-3.5-flash-lite_v1_dev.md"
    return "\n\n".join(tables(path)[:2]) + f"\n\nFrom {linked(path)}."


def latency_table() -> str:
    path = RESULTS / "latency.md"
    return f"{tables(path)[0]}\n\nOn a laptop CPU, from {linked(path)}."


BLOCKS = {
    "headline": headline,
    "test_table": test_table,
    "bm25": lambda: dev_table("bm25_dev.md"),
    "fusion": lambda: dev_table("fusion_dev.md"),
    "qu_scores": qu_scores,
    "qu_ranking": lambda: dev_table("qu_dev.md"),
    "rerank": lambda: dev_table("rerank_dev.md"),
    "judge": judge_scores,
    "latency": latency_table,
}


def render(text: str) -> str:
    for name, make in BLOCKS.items():
        pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.DOTALL)
        text, count = pattern.subn(lambda m, make=make: f"{m.group(1)}{make()}\n{m.group(2)}", text)
        if count != 1:
            raise ValueError(f"README has {count} blocks named {name}")
    return text


def update(path: Path = README) -> None:
    path.write_text(render(path.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
