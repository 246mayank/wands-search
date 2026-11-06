"""`wands` command line."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Fetch and inspect the WANDS dataset.")
app.add_typer(data_app, name="data")

FREEZE_TAG = "freeze-v1"
RUNS_DIR = Path("runs")


@data_app.command("fetch")
def data_fetch() -> None:
    """Download WANDS at the pinned commit and verify its checksums."""
    from wands import data

    for name, digest in data.fetch().items():
        typer.echo(f"ok {name} sha256={digest}")


@data_app.command("profile")
def data_profile() -> None:
    """Regenerate docs/DATA_NOTES.md."""
    from wands.profile import write_notes

    typer.echo(f"wrote {write_notes()}")


@data_app.command("split")
def data_split() -> None:
    """Create splits.json. Refuses to change an existing split."""
    from wands import data
    from wands.eval.split import SPLITS_PATH, load_split, make_split, write_split

    queries = data.load().queries.select("query_id", "query_class").iter_rows()
    split = make_split(queries)
    if SPLITS_PATH.exists():
        if load_split("dev") != split["dev"] or load_split("test") != split["test"]:
            raise typer.BadParameter("splits.json differs from what this code generates; it is frozen")
        typer.echo("splits.json is unchanged")
        return
    write_split(split)
    typer.echo(f"wrote {SPLITS_PATH}: {len(split['dev'])} dev, {len(split['test'])} test")


@data_app.command("random-run")
def data_random_run(out: Path = Path("runs/random_run.tsv"), seed: int = 7) -> None:
    """Write a run that ranks 200 random products per query: the floor for every metric."""
    from wands import data
    from wands.eval.runfile import write_run

    wands = data.load()
    ids = wands.products["product_id"].to_list()
    rng = random.Random(seed)
    rows = {
        qid: [(pid, 200.0 - rank) for rank, pid in enumerate(rng.sample(ids, 200))]
        for qid in wands.queries["query_id"].to_list()
    }
    write_run(rows, out)
    typer.echo(f"wrote {out}")


def require_freeze() -> None:
    tags = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.split()
    if FREEZE_TAG not in tags:
        raise typer.BadParameter(f"the test split runs once, at tag {FREEZE_TAG}; HEAD is not tagged")


@app.command("eval")
def evaluate(
    config: Path = typer.Option(None, help="Pipeline YAML to run."),
    run: Path = typer.Option(None, help="Existing run file to score instead."),
    split: str = typer.Option("dev", help="dev or test."),
    depth: int = typer.Option(200, help="Results kept per query."),
) -> None:
    """Score a pipeline or a run file on a split and write a report."""
    from wands.eval.provenance import git_sha
    from wands.eval.report import render_report, score_run, write_report
    from wands.eval.runfile import read_run, write_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline

    if (config is None) == (run is None):
        raise typer.BadParameter("give exactly one of --config and --run")
    if split == "test":
        require_freeze()

    world = World.load()
    query_ids = load_split(split)
    sha = git_sha()
    if config is not None:
        cfg = load_config(config)
        name = cfg["name"]
        hits = run_pipeline(build(cfg["pipeline"], world, name), world, query_ids, depth)
        rows = {qid: [(h.product_id, h.score) for h in found] for qid, found in hits.items()}
        run_path = RUNS_DIR / f"{name}_{split}.tsv"
        write_run(rows, run_path)
        ranked = {qid: [pid for pid, _ in found] for qid, found in rows.items()}
        source = str(config).replace("\\", "/")
    else:
        name = run.stem
        ranked = read_run(run)
        source = str(run).replace("\\", "/")

    scored = score_run(ranked, world.qrels, query_ids, name, split)
    text = render_report(scored, sha, source, world.queries)
    # A dirty tree can't be reproduced from a commit, so its report stays out of results/.
    path = write_report(scored, text, sha, RUNS_DIR if sha.endswith("-dirty") or run else Path("results"))
    typer.echo(text.split("## Per query")[0].rstrip())
    typer.echo(f"\nreport: {path}")


@app.command("tune-bm25")
def tune_bm25_cmd() -> None:
    """Grid-search field weights on dev, log every config, and write configs/bm25f.yaml."""
    from wands import tune
    from wands.eval.split import load_split
    from wands.pipeline import World

    rows = tune.tune_bm25(World.load(), load_split("dev"))
    tune.write_grid(rows, Path("results/bm25_grid_dev.csv"))
    tune.write_bm25_config(rows[0], Path("configs/bm25f.yaml"))
    for row in rows[:5]:
        typer.echo(row)
    typer.echo(f"... worst {rows[-1]}")


@app.command("compare")
def compare(
    configs: list[Path] = typer.Argument(..., help="Pipeline YAMLs; the first is the base row."),
    out: Path = typer.Option(..., help="Markdown table to write under results/."),
    split: str = "dev",
    title: str = "Comparison",
    depth: int = 200,
) -> None:
    """Run several configs and write a table with a bootstrap CI against the first."""
    from wands.eval.metrics import METRICS
    from wands.eval.provenance import git_sha
    from wands.eval.report import compare_table, score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline

    if split == "test":
        require_freeze()
    world = World.load()
    query_ids = load_split(split)
    rows = []
    for config in configs:
        cfg = load_config(config)
        hits = run_pipeline(build(cfg["pipeline"], world, cfg["name"]), world, query_ids, depth)
        ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
        rows.append(score_run(ranked, world.qrels, query_ids, cfg["name"], split))
        typer.echo(f"{cfg['name']}: ndcg@10 {rows[-1].mean('ndcg@10'):.4f}")
    table = compare_table(rows, rows[0], METRICS)
    sources = ", ".join(f"`{str(c).replace(chr(92), '/')}`" for c in configs)
    text = (
        f"# {title}\n\nSplit `{split}`, {len(query_ids)} queries, commit `{git_sha()}`, configs {sources}. "
        "Generated by `wands compare`. `*` marks a delta whose 95% paired-bootstrap CI excludes 0.\n\n"
        + table
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


if __name__ == "__main__":
    app()
